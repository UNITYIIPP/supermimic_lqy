"""Multi-task differentiable world model env for concurrent encoder training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.history import append_history

if TYPE_CHECKING:
  from supermimic.models import Dynamics
  from supermimic.replay_buffer import ReplayBuffer
  from supermimic.representations import StateRepresentation
  from supermimic.tasks.task_spec import TaskSpec


class MultiTaskWorldModelEnv(WorldModelEnv):
  """WM env for motion tracking with per-slot goal modalities.

  Reward is motion-tracking for every slot; tasks differ only in what
  ``goal_<name>`` key the actor's encoder consumes. ``get_observations``
  emits all per-task goal keys full-batch and the actor slices its own
  slot range. The obs-history ring (history_length > 1) lives in the base.
  """

  def __init__(
    self,
    dynamics: Dynamics,
    representation: StateRepresentation,
    env: Any,
    batch_size: int,
    replay_buffer: ReplayBuffer,
    task_specs: list[TaskSpec],
    history_length: int,
    termination_softness: float,
    device: str = "cpu",
  ) -> None:
    self.task_specs = task_specs
    assert task_specs[0].name == "motion", "task_specs[0] must be the motion task"
    super().__init__(
      dynamics=dynamics,
      representation=representation,
      env=env,
      batch_size=batch_size,
      replay_buffer=replay_buffer,
      history_length=history_length,
      termination_softness=termination_softness,
      device=device,
      goal_history_length=max(ts.history_length for ts in task_specs),
    )

  def _build_task_goals(self) -> None:
    # The anchor task's goal is the actor goal already built in _update_goal; reuse it.
    anchor, *rest = self.task_specs
    self._task_goal_frames: dict[str, torch.Tensor] = {
      anchor.name: (
        anchor.extract_training_goal(self.motion, self.motion_times, self.aux_state)
        if anchor.history_length
        else self._actor_goal_obs
      )
    }
    for ts in rest:
      self._task_goal_frames[ts.name] = ts.extract_training_goal(
        self.motion, self.motion_times, self.aux_state
      ).clone()

  def _on_reset(self, batch) -> None:
    super()._on_reset(batch)
    self._task_goal_obs: dict[str, torch.Tensor] = {}
    for ts in self.task_specs:
      frame = self._task_goal_frames[ts.name]
      if ts.history_length:
        # The newest frame is rebuilt from the reset state. Earlier frames keep
        # the relative orientations observed in the actual replay trajectory.
        past = (
          batch["goal_window"][:, -ts.history_length :]
          if ts.history_length > 1
          else None
        )
        self._task_goal_obs[ts.name] = (
          torch.cat([past[:, :-1], frame.unsqueeze(1)], dim=1)
          if past is not None
          else frame.unsqueeze(1)
        )
      else:
        self._task_goal_obs[ts.name] = frame

  def _advance_task_history(self) -> None:
    for ts in self.task_specs:
      frame = self._task_goal_frames[ts.name]
      self._task_goal_obs[ts.name] = (
        append_history(self._task_goal_obs[ts.name], frame, ts.history_length)
        if ts.history_length
        else frame
      )

  def get_observations(self) -> TensorDict:
    return build_actor_obs_td(
      self._task_goal_obs, self._sensor_history(), critic=self._critic_obs()
    )
