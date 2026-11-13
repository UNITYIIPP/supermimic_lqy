from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import torch
import torch.nn as nn
import tyro
import wandb
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from wandb.util import json_friendly_val

from supermimic.envs.motion_tracking_env import MotionTrackingEnv
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.learners.learner_utils import Execution, Timings, resolve_execution
from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import Dynamics, MultiEncoderAE, build_mlp
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import StateRepresentation, create_representation
from supermimic.robots.control import control_snapshot
from supermimic.runners.exporter import export_actor_as_onnx
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.sim_context import SimContext
from supermimic.tasks.task_spec import (
  TasksCfg,
  build_actor_obs_td,
  resolve_task_specs,
)
from supermimic.utils import distributed as distributed
from supermimic.utils.helpers import (
  extract_action_scale_offset,
  unpack_obs_dims,
)
from supermimic.utils.logger import Logger
from supermimic.utils.wandb_artifacts import download_model_dir, log_model_artifact


@dataclass
class BaseRunnerCfg:
  """Shared configuration fields for all runners."""

  wandb_enabled: bool = False
  """Record metrics and upload models to W&B in addition to local TensorBoard."""
  wandb_project: str = "supermimic"
  run_name: str | None = None
  log_root: str = "logs"
  wandb_run_path: str | None = None
  headless: bool = True
  buffer_size: int = 1000
  representation: str = "gencoord"  # "gencoord" or "maxcoord"
  actor_obs: str = "proprio"  # "rep_state" or "proprio"
  log_interval: int = 10
  save_interval: int = 1000
  execution: Execution = "auto"
  """auto: full graph on one GPU; split forward/backward graphs on multiple GPUs."""
  profile: bool = False
  """Synchronize stage boundaries and write per-iteration performance.jsonl."""
  profile_warmup_iterations: int = 20
  resume: str | None = None
  """Path to training_state.pt; replay and real environments are re-warmed."""
  check_sync_interval: int = 0
  """Expensive cross-rank model/statistics consistency check; 0 disables."""
  export_onnx: bool = True
  history_length: int = 1
  termination_softness: float = 0.05
  dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)


BpttCfg = Annotated[BpttPolicyLearnerCfg, tyro.conf.subcommand(name="bptt")]
ActorCriticCfg = Annotated[
  ActorCriticPolicyLearnerCfg, tyro.conf.subcommand(name="actor-critic")
]


@dataclass
class SuperMimicRunnerCfg(BaseRunnerCfg):
  max_iterations: int = 1000000
  replay_ratio: int = 16
  latent_dim: int = 32
  actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  critic_hidden_dims: tuple[int, ...] = (256, 128)
  latent_bc_weight: float = 1e-1
  """Weight on the latent BC loss pulling non-anchor encoders toward the
  anchor encoder's latent on the same state. 0 disables."""
  tasks: TasksCfg = field(default_factory=TasksCfg)
  """Per-task hyperparameters."""
  slot_mix: dict[str, float] = field(
    default_factory=lambda: {"motion": 1.0, "velocity": 0.0}
  )
  """Fractional slot allocation per task. Must sum to 1.0."""
  policy: BpttCfg | ActorCriticCfg = field(default_factory=ActorCriticCfg)

  sampling_mode: Literal["uniform", "adaptive", "start", "return"] = "return"
  """RSI reset distribution: 'return' (concentrate resets on the bins where the
  policy earns low return; see ReturnSampler) or one of mjlab's native modes.
  train.py resolves this into the motion command's cfg before the env is built
  ('return' drives mjlab's 'external' mode)."""
  return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)


@dataclass
class WorldModel:
  """The world model plus the env dimensions that sized it."""

  representation: StateRepresentation
  dynamics: Dynamics
  sim_context: SimContext
  num_goals: int
  num_actions: int
  robot_control: dict | None = None

  def bundle(self) -> dict:
    """The ``world_model.pt`` contents."""
    return {
      "dynamics": self.dynamics.state_dict(),
      "sim_context": self.sim_context.state_dict(),
      "robot_control": self.robot_control,
    }

  def load_bundle(self, bundle: dict) -> None:
    """Load a ``world_model.pt`` bundle."""
    saved_control = bundle.get("robot_control")
    if (saved_control is None) != (self.robot_control is None):
      raise ValueError("Checkpoint robot does not match this task (EVT/G1)")
    if saved_control is not None:
      assert self.robot_control is not None
      for key in saved_control.keys() - {"parameters"}:
        if saved_control[key] != self.robot_control.get(key):
          raise ValueError(
            f"Checkpoint robot control differs: {key}; restore its control "
            "snapshot before creating the environment"
          )
    self.dynamics.load_state_dict(bundle["dynamics"])
    self.sim_context.load_state_dict(bundle["sim_context"])


def build_world_model(
  cfg: BaseRunnerCfg, env, device: str, model_seed: int | None = None
) -> WorldModel:
  """Construct the representation and dynamics for ``env`` sized by ``cfg``.

  Resets ``env`` to read its observation dimensions.
  """
  if cfg.history_length > 1 and cfg.actor_obs != "proprio":
    raise RuntimeError(
      "history_length > 1 requires actor_obs='proprio' (--runner.actor-obs proprio)"
    )

  obs, _ = env.reset()
  num_full_state, num_goals = unpack_obs_dims(obs)
  num_actions = env.num_actions

  print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

  sim_context = SimContext.from_env(env.unwrapped)
  sim_ctx_dim = sim_context.output_dim
  print(
    f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})"
  )

  default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0].to(device)
  representation = create_representation(
    cfg.representation,
    num_full_state,
    num_actions,
    env.unwrapped.step_dt,
    env.unwrapped.cfg.decimation,
    default_joint_pos,
    actor_obs_mode=cfg.actor_obs,
  ).to(device)

  representation.bind(env.unwrapped, sim_context)

  print(
    f"[INFO] Representation: {cfg.representation} "
    f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
    f"pred={representation.prediction_dim})"
  )

  action_scale, action_offset = extract_action_scale_offset(env)
  with distributed.model_initialization(model_seed):
    dynamics = Dynamics(
      representation,
      num_actions,
      sim_ctx_dim,
      action_scale,
      action_offset,
      hidden_dims=cfg.dynamics_hidden_dims,
    ).to(device)

  return WorldModel(
    representation=representation,
    dynamics=dynamics,
    sim_context=sim_context,
    num_goals=num_goals,
    num_actions=num_actions,
    robot_control=control_snapshot(env.unwrapped),
  )


def build_actor(
  cfg: SuperMimicRunnerCfg, wm: WorldModel, device: str
) -> MultiEncoderAE:
  """Actor sized by ``cfg``; the resolved task specs live on ``actor.task_specs``."""
  representation = wm.representation
  obs_dim = (
    representation.rep_state_dim
    if cfg.actor_obs == "rep_state"
    else representation.proprio_obs_dim(wm.num_actions)
  )
  return MultiEncoderAE(
    obs_dim=obs_dim,
    action_dim=wm.num_actions,
    task_specs=resolve_task_specs(cfg.tasks, representation, wm.num_goals, device),
    slot_mix=cfg.slot_mix,
    latent_dim=cfg.latent_dim,
    hidden_dims=cfg.actor_hidden_dims,
    latent_bc_weight=cfg.latent_bc_weight,
    history_length=cfg.history_length,
  ).to(device)


def load_wandb_checkpoint(wandb_run_path: str, wm: WorldModel, device: str) -> None:
  """Warm-start ``wm`` from another W&B run's world-model bundle."""
  print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
  path = None
  if distributed.rank() == 0:
    run = wandb.Api().run(wandb_run_path)
    path = str(download_model_dir(run) / "world_model.pt")
  path = distributed.broadcast_object(path)
  assert isinstance(path, str)
  wm.load_bundle(torch.load(path, map_location=device))


class Ratio:
  def __init__(self, ratio: float) -> None:
    self._ratio = ratio
    self._prev: int | None = None

  def __call__(self, step: int) -> int:
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class BaseRunner[CfgT: BaseRunnerCfg]:
  """Base class with shared initialization for all runners.

  Handles: env dimension extraction, representation creation, world model
  setup, replay buffer, action manager, and logger. Subclasses implement
  ``_init_policy()`` for their specific policy setup.
  """

  cfg: CfgT

  def __init__(
    self,
    env,
    cfg: CfgT,
    task_id: str,
    registry_name: str | None = None,
    device: str = "cuda:0",
    model_seed: int | None = None,
  ):
    self.env = env
    self.cfg = cfg
    self.task_id = task_id
    self.registry_name = registry_name
    self.device = device
    self.model_seed = model_seed
    self.startup_seconds = 0.0
    self.training_wall_seconds = 0.0
    self.execution: Execution = resolve_execution(cfg.execution, cfg.profile)

    self._init_common()
    self._init_policy()

    self.logger = Logger(
      run_name=cfg.run_name,
      num_envs=env.num_envs,
      max_episode_length=env.max_episode_length,
      device=device,
      wandb_enabled=cfg.wandb_enabled,
    )

  def _init_common(self) -> None:
    wm = build_world_model(self.cfg, self.env, self.device, self.model_seed)
    self._wm = wm
    self._representation = wm.representation
    self._dynamics = wm.dynamics
    self._sim_context = wm.sim_context
    self._num_goals = wm.num_goals
    self._num_actions = wm.num_actions

    if self.cfg.wandb_run_path is not None:
      load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)

    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion

    self.replay_buffer = ReplayBuffer(
      schema=self._buffer_schema(),
      buffer_size=self.cfg.buffer_size,
      num_envs=self.env.num_envs,
      device=self.device,
    )
    self.world_model_learner = WorldModelLearner(
      self._dynamics,
      self._representation,
      self.cfg.wm,
      device=self.device,
      execution=self.execution,
    )

    self._pending_norm_updates: list[tuple] = []
    self._sim_history: torch.Tensor | None = None

  def _seed_sim_history(self) -> None:
    """Seed the real-sim obs-history ring with the noised current sensor."""
    if self.cfg.history_length == 1:
      return
    seed = self._representation.add_proprio_noise(
      self.env.get_observations()["actor"].to(self.device)
    )
    self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

  def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
    """The ``(B, history_length, sensor_dim)`` window for a real-sim actor step.
    history_length > 1 needs ``_seed_sim_history()`` first."""
    if self.cfg.history_length == 1:
      return sensor.unsqueeze(1)
    assert self._sim_history is not None
    return self._sim_history

  def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
    """Append the noised next sensor; rows whose episode just reset repeat that
    single realization, matching the warm-start padding."""
    if self.cfg.history_length == 1:
      return
    assert self._sim_history is not None
    next_sensor = self._representation.add_proprio_noise(next_actor)
    shifted = torch.cat([self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1)
    reset = dones.bool().view(-1, 1, 1)
    self._sim_history = torch.where(reset, next_sensor.unsqueeze(1), shifted)

  def _policy_input_dims(self) -> tuple[int, int]:
    """Compute actor and critic input dimensions from the representation."""
    actor_dim = self._representation.actor_obs_dim(
      self._num_actions, self._num_goals, self.cfg.history_length
    )
    critic_dim = (
      self._representation.rep_state_dim
      + self._num_goals
      + self._sim_context.output_dim
    )
    return actor_dim, critic_dim

  def _build_mlp(
    self,
    name: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
    **kwargs: object,
  ) -> nn.Module:
    """Build an MLPModel with input obs normalization."""
    obs = TensorDict(
      {name: torch.zeros(1, input_dim, device=self.device)},
      batch_size=[1],
      device=self.device,
    )
    return MLPModel(
      obs=obs,
      obs_groups={name: [name]},
      obs_set=name,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      obs_normalization=True,
      **kwargs,
    ).to(self.device)

  def _create_wm_env(self, batch_size: int) -> MotionTrackingEnv:
    """Create a :class:`MotionTrackingEnv` bound to this runner's world model and buffer."""
    return MotionTrackingEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=batch_size,
      replay_buffer=self.replay_buffer,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )

  @property
  def _stores_actor_obs(self) -> bool:
    """The buffer carries the real sensor only for its sole reader, the
    history warm-start (``WorldModelEnv.sample_initial_states``)."""
    return (
      self._representation.actor_obs_mode == "proprio" and self.cfg.history_length > 1
    )

  def _buffer_schema(self) -> dict[str, FieldSpec]:
    """Override to customize the replay buffer schema (e.g. extra fields)."""
    schema = {
      "state": FieldSpec(self._representation.full_state_dim),
      "action": FieldSpec(self._num_actions),
      "prev_action": FieldSpec(self._num_actions),
      "motion_time": FieldSpec(dtype=torch.long),
      "terminated": FieldSpec(dtype=torch.bool),
      "truncated": FieldSpec(dtype=torch.bool),
    }
    if self._stores_actor_obs:
      schema["actor_obs"] = FieldSpec(
        self._representation.proprio_obs_dim(self._num_actions)
      )
    schema["sim_ctx"] = FieldSpec(self._sim_context.output_dim)
    return schema

  def _init_policy(self) -> None:
    """Subclasses implement this to create their policy learner."""
    raise NotImplementedError

  def _step_and_store(
    self,
    actions: torch.Tensor,
    curr_states: torch.Tensor,
    actor_td: TensorDict | None = None,
    extra_fields: dict[str, torch.Tensor] | None = None,
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Step the env and write the transition to the replay buffer.

    When given, ``actor_td`` is queued for the actor-normalizer update.
    """
    motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    motion_times = motion_cmd.time_steps.clone()

    raw_actions = actions.detach().clone()
    prev_action = self.action_manager.action.clone()

    next_obs, rewards, dones, extras = self.env.step(actions)

    truncated = extras.get(
      "time_outs",
      torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
    )
    terminated = dones.bool() & ~truncated

    fields: dict[str, torch.Tensor] = {
      "state": curr_states,
      "action": raw_actions,
      "prev_action": prev_action,
      "motion_time": motion_times,
      "terminated": terminated,
      "truncated": truncated,
    }
    if extra_fields:
      fields.update(extra_fields)
    self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))

    if actor_td is not None:
      self._pending_norm_updates.append(
        (
          curr_states.clone(),
          raw_actions.clone(),
          actor_td.detach().clone(),
        )
      )
    return next_obs, rewards, dones, extras

  def _flush_normalization(self) -> None:
    """Apply queued normalizer updates. Call outside ``inference_mode()``."""
    if not self._pending_norm_updates:
      return
    states, actions, actor_inputs = zip(*self._pending_norm_updates, strict=True)
    self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
    self._update_actor_normalization(actor_inputs)
    self._pending_norm_updates.clear()

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    """Subclasses override to update the actor obs normalizer."""
    raise NotImplementedError

  def _create_log_dir(self, *extra_parts: str) -> Path:
    """Create and return a timestamped log directory."""
    timestamp = distributed.broadcast_object(
      datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
      if distributed.rank() == 0
      else None
    )
    assert isinstance(timestamp, str)
    log_dir = Path(self.cfg.log_root, self.task_id, *extra_parts, timestamp)
    if distributed.rank() == 0:
      log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Logging to {log_dir}")
    return log_dir

  def _init_logging(self, log_dir: Path, name_suffix: str = "") -> None:
    """Save local logs/config, optionally starting a W&B run for this training."""
    self._wandb_run = None
    self._run_name = self.cfg.run_name or (
      f"{self.task_id}{name_suffix}-{log_dir.name}"
    )
    if distributed.rank() != 0:
      return
    self.logger.init_tensorboard(log_dir)
    run_config = {
      "env_cfg": asdict(self.env.cfg),
      "runner_cfg": asdict(self.cfg),
      "task_id": self.task_id,
      "robot_control": self._wm.robot_control,
      "distributed": {
        "world_size": distributed.world_size(),
        "execution": self.execution,
        "global_num_envs": self.env.num_envs * distributed.world_size(),
        "model_seed": self.model_seed,
      },
    }
    if self._wm.robot_control:
      (log_dir / "robot_control.json").write_text(
        json.dumps(self._wm.robot_control, indent=2) + "\n"
      )

    # Use the same value normalization as W&B without starting a W&B run.
    (log_dir / "config.json").write_text(
      json.dumps(json_friendly_val(run_config), indent=2) + "\n"
    )
    if not self.cfg.wandb_enabled:
      print("[INFO] W&B recording disabled; using local TensorBoard and checkpoints")
      return

    self._wandb_run = wandb.init(
      project=self.cfg.wandb_project,
      name=self._run_name,
      config=run_config,
    )
    self._run_name = self._wandb_run.name

    if self.registry_name:
      api = wandb.Api()
      for name in self.registry_name.split(","):
        name = name.strip()
        if ":" not in name:
          name = name + ":latest"
        print(f"[INFO] Linking registry artifact: {name}")
        artifact = api.artifact(name)
        self._wandb_run.use_artifact(artifact)

  def _cleanup(self) -> None:
    """Flush local metrics, close the environment, and finish the W&B run."""
    try:
      self.logger.close()
    finally:
      try:
        env_close_fn = getattr(self.env, "close", None)
        if env_close_fn:
          env_close_fn()
      finally:
        if getattr(self, "_wandb_run", None) is not None:
          wandb.finish()


class SuperMimicRunner(BaseRunner[SuperMimicRunnerCfg]):
  """Trains WM + multi-encoder actor end-to-end via BPTT/AC.

  Slots are contiguous blocks of the env batch; each block routes through its
  task's encoder. Motion-tracking reward is the only reward path; tasks differ
  only in what goal modality their encoder consumes.
  """

  def _init_policy(self) -> None:
    with distributed.model_initialization(
      self.model_seed + 1 if self.model_seed is not None else None
    ):
      actor = build_actor(self.cfg, self._wm, self.device)
    self._active_task_specs = [
      ts for ts in actor.task_specs if self.cfg.slot_mix.get(ts.name, 0.0) > 0.0
    ]

    print("[INFO] Sim slot ranges:")
    for ts, (lo, hi) in zip(
      actor.task_specs, actor.slot_ranges(self.env.num_envs), strict=True
    ):
      print(f"  {ts.name}: [{lo}, {hi}) — goal_dim={ts.goal_dim}")
    for size in (self.env.num_envs, self.cfg.policy.batch_size):
      for name, (lo, hi) in zip(
        actor._task_names, actor.slot_ranges(size), strict=True
      ):
        if self.cfg.slot_mix.get(name, 0) > 0 and hi <= lo:
          raise ValueError(f"Local batch {size} leaves active task {name} empty")
    _, critic_input_dim = self._policy_input_dims()
    with distributed.model_initialization(
      self.model_seed + 2 if self.model_seed is not None else None
    ):
      critic = build_mlp(
        critic_input_dim,
        self.cfg.critic_hidden_dims,
        output_dim=1,
      ).to(self.device)

    policy_cfg = self.cfg.policy

    self.wm_env = MultiTaskWorldModelEnv(
      dynamics=self._dynamics,
      representation=self._representation,
      env=self.env,
      batch_size=policy_cfg.batch_size,
      replay_buffer=self.replay_buffer,
      task_specs=self._active_task_specs,
      history_length=self.cfg.history_length,
      termination_softness=self.cfg.termination_softness,
      device=self.device,
    )
    if torch.cuda.is_available():
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)

    self._configure_reset_sampling()

    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(
        actor,
        critic,
        self.wm_env,
        policy_cfg,
        device=self.device,
        execution=self.execution,
        profile=self.cfg.profile,
      )
    else:
      self.policy_learner = BpttPolicyLearner(
        actor,
        self.wm_env,
        policy_cfg,
        device=self.device,
        execution=self.execution,
        profile=self.cfg.profile,
      )

    modules = self._shared_modules()
    distributed.check_same(
      [
        (key, tuple(t.shape), str(t.dtype))
        for m in modules
        for key, t in m.state_dict().items()
      ],
      "shared model layout",
    )
    sources = self.motion_cmd.cfg.motion_file
    sources = [sources] if isinstance(sources, (str, Path)) else sources
    self._motion_manifest = []
    for pattern in sources:
      for name in sorted(glob.glob(os.path.expanduser(str(pattern)))):
        path = Path(name).resolve()
        stat = path.stat()
        self._motion_manifest.append((str(path), stat.st_size, stat.st_mtime_ns))
    distributed.check_same(
      (self._motion_manifest, self.motion.time_step_total, self.motion_cmd.bin_count),
      "motion library layout",
    )
    distributed.broadcast_modules(modules)

  def _shared_modules(self) -> list[nn.Module]:
    modules = [self._dynamics, self.policy_learner.actor]
    critic = getattr(self.policy_learner, "critic", None)
    if critic is not None:
      modules.append(critic)
    return modules

  def _optimizers(self) -> list[torch.optim.Optimizer]:
    return [
      *self.world_model_learner.optimizers,
      *self.policy_learner.actor_optimizers,
      *getattr(self.policy_learner, "critic_optimizers", []),
    ]

  def _save_training_state(
    self, path: Path, step: int, updates: int, collected: int
  ) -> None:
    critic = getattr(self.policy_learner, "critic", None)
    state = {
      "version": 1,
      "task_id": self.task_id,
      "runner_cfg": asdict(self.cfg),
      "world_size": distributed.world_size(),
      "model_seed": self.model_seed,
      "motion_manifest": self._motion_manifest,
      "world_model": self._wm.bundle(),
      "actor": self.policy_learner.actor.state_dict(),
      "critic": critic.state_dict() if critic is not None else None,
      "optimizers": [opt.state_dict() for opt in self._optimizers()],
      "sampler": self._sampler.state_dict() if self._sampler is not None else None,
      "next_iteration": step + 1,
      "updates": updates,
      "collected": collected,
      "training_wall_s": self.training_wall_seconds,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

  def _load_training_state(self) -> tuple[int, int, int]:
    if self.cfg.resume is None:
      return 0, 0, 0
    ckpt = torch.load(self.cfg.resume, map_location=self.device, weights_only=False)
    if ckpt["version"] != 1 or ckpt["task_id"] != self.task_id:
      raise ValueError("Checkpoint version/task does not match")
    if ckpt.get("motion_manifest", self._motion_manifest) != self._motion_manifest:
      raise ValueError("Motion files/order changed; cannot restore the return sampler")
    self.training_wall_seconds = ckpt.get("training_wall_s", 0.0)
    previous = ckpt["runner_cfg"]
    for key in (
      "representation",
      "actor_obs",
      "history_length",
      "slot_mix",
      "tasks",
      "sampling_mode",
      "return_sampler",
      "latent_bc_weight",
      "termination_softness",
    ):
      if previous[key] != asdict(self.cfg)[key]:
        raise ValueError(f"Resume configuration mismatch: {key}")
    # Batch sizes may change on resume; other learner hyperparameters must match.
    for key in ("wm", "policy"):
      before = {k: v for k, v in previous[key].items() if k != "batch_size"}
      after = {k: v for k, v in asdict(self.cfg)[key].items() if k != "batch_size"}
      if before != after:
        raise ValueError(f"Resume learner configuration mismatch: {key}")
    self._wm.load_bundle(ckpt["world_model"])
    self.policy_learner.actor.load_state_dict(ckpt["actor"])
    if ckpt["critic"] is not None:
      critic = getattr(self.policy_learner, "critic", None)
      if critic is None:
        raise ValueError("Actor-critic checkpoint cannot resume as pure BPTT")
      critic.load_state_dict(ckpt["critic"])
    for opt, state in zip(self._optimizers(), ckpt["optimizers"], strict=True):
      opt.load_state_dict(state)
    if self._sampler is not None:
      self._sampler.load_state_dict(ckpt["sampler"])
      # Compute probabilities without counting an additional sampling update.
      if self._sampler.effort.sum() > 0:
        self.motion_cmd.set_sampling_distribution(self._sampler._distribution(False))
    print(
      "[INFO] Restored models/optimizers/sampler; re-warming local replay (not bitwise resume)"
    )
    return ckpt["next_iteration"], ckpt["updates"], ckpt["collected"]

  def _check_sync(self) -> None:
    tensors = [
      t
      for module in self._shared_modules()
      for t in [*module.parameters(), *module.buffers()]
    ]
    if self._sampler is not None:
      tensors += list(self._sampler.state_dict().values())
    for opt in self._optimizers():
      tensors += [
        v
        for state in opt.state.values()
        for v in state.values()
        if isinstance(v, torch.Tensor) and v.is_cuda
      ]
    distributed.assert_synced(tensors)

  def _build_sim_obs_td(
    self,
    rep_state: torch.Tensor,
    aux_state: torch.Tensor,
    sim_actor_obs: torch.Tensor,
    motion_times: torch.Tensor,
    sim_ctx: torch.Tensor,
  ) -> TensorDict:
    proprio = self.cfg.actor_obs == "proprio"
    sensor = self._representation.compute_actor_sensor(
      rep_state,
      sim_ctx,
      aux_state=aux_state,
      sim_actor_obs=sim_actor_obs if proprio else None,
      noisy=proprio,
    )
    goals = {
      ts.name: ts.extract_training_goal(self.motion, motion_times, aux_state)
      for ts in self._active_task_specs
    }
    return build_actor_obs_td(goals, self._collection_history(sensor))

  def _configure_reset_sampling(self) -> None:
    self._sampler: ReturnSampler | None = None
    env_mode = self.motion_cmd.cfg.sampling_mode
    expected = (
      "external" if self.cfg.sampling_mode == "return" else self.cfg.sampling_mode
    )
    if env_mode != expected:
      raise RuntimeError(
        f"runner sampling_mode='{self.cfg.sampling_mode}' needs the motion command "
        f"in '{expected}' mode but the env was built with '{env_mode}'; resolve it "
        "into the env cfg before env creation ('external' also needs an mjlab with "
        "MotionCommand external-mode support)"
      )
    if self.cfg.sampling_mode != "return":
      return
    self._sampler = ReturnSampler(
      cfg=self.cfg.return_sampler,
      bin_count=self.motion_cmd.bin_count,
      frame_count=self.motion.time_step_total,
      num_envs=self.env.num_envs,
      max_step_reward=self.wm_env.max_step_reward,
      device=self.device,
    )

  def step(self) -> int:
    with torch.inference_mode():
      obs = self.env.get_observations()
      curr_states = obs["full_state"].to(self.device)
      sim_actor_obs = obs["actor"].to(self.device)

      # Pre-step frames; env.step() advances time_steps in place.
      motion_times = self.motion_cmd.time_steps.to(self.device).clone()

      rep_state = self._representation.represent(curr_states)
      aux_state = self._representation.extract_aux_state(curr_states)
      sim_ctx = self._sim_context.read(self.env.unwrapped)
      actor_td = self._build_sim_obs_td(
        rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx
      )

      actions = self.policy_learner.actor(actor_td)
      actions = actions + torch.randn_like(actions) * self.policy_learner.cfg.noise

      extra_fields = {"sim_ctx": sim_ctx}
      if self._stores_actor_obs:
        extra_fields["actor_obs"] = sim_actor_obs

      next_obs, rewards, dones, extras = self._step_and_store(
        actions,
        curr_states,
        actor_td=actor_td,
        extra_fields=extra_fields,
      )
      self._advance_sim_history(next_obs["actor"].to(self.device), dones)
      terminated = self.env.unwrapped.termination_manager.terminated
      self.logger.process_env_step(rewards, terminated, extras)

      if self._sampler is not None:
        # A failure on a step that also times out counts as the time-out.
        manager = self.env.unwrapped.termination_manager
        truncated = manager.time_outs.to(self.device, torch.bool)
        clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
        weights = self._sampler.record(
          motion_times,
          rewards,
          terminated.to(self.device, torch.bool) & ~truncated,
          clip_end,
          truncated & ~clip_end,
        )
        if weights is not None:
          self.motion_cmd.set_sampling_distribution(weights)

    self._flush_normalization()
    return self.env.num_envs

  def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
    self.policy_learner.actor.train()
    for actor_td in actor_inputs:
      self.policy_learner.actor.update_normalization(actor_td)

  def _save_outputs(
    self, log_dir: Path, step: int, updates: int, collected: int
  ) -> None:
    if distributed.rank() == 0:
      self._save_training_state(log_dir / "training_state.pt", step, updates, collected)
      world_model_path = str(log_dir / "world_model.pt")
      actor_path = str(log_dir / "actor.pt")
      torch.save(self._wm.bundle(), world_model_path)
      torch.save(self.policy_learner.actor.state_dict(), actor_path)
      files = [world_model_path, actor_path]
      critic = getattr(self.policy_learner, "critic", None)
      if critic is not None:
        critic_path = str(log_dir / "critic.pt")
        torch.save(critic.state_dict(), critic_path)
        files.append(critic_path)
      if self._wm.robot_control:
        files.append(str(log_dir / "robot_control.json"))
      if (
        self.cfg.export_onnx
        and self._representation.actor_obs_mode == "proprio"
        and self.motion.time_step_total <= 15_000
      ):
        files += export_actor_as_onnx(
          actor=self.policy_learner.actor,
          env=self.env.unwrapped,
          path=str(log_dir),
          run_name=self._run_name,
          term_widths=self._representation.proprio_term_widths(self._num_actions),
        )
      if self.cfg.wandb_enabled:
        log_model_artifact(files)
    # All ranks resume collection together after potentially slow exports.
    distributed.broadcast_object(None)

  def run(self) -> None:
    setup_started = time.perf_counter()
    profile_file = None
    try:
      log_dir = self._create_log_dir()
      self._init_logging(log_dir)
      first_step, updates, total_collected = self._load_training_state()
      self._check_sync()
      self.logger.tot_timesteps = total_collected
      self.env.reset()
      self._seed_sim_history()
      if self.cfg.profile and distributed.rank() == 0:
        profile_file = (log_dir / "performance.jsonl").open("w")
      timings = Timings(self.cfg.profile, self.device)
      started = time.perf_counter()
      self.startup_seconds += started - setup_started
      prior_wall = self.training_wall_seconds
      print(
        f"[INFO] Training iterations {first_step}..{self.cfg.max_iterations}; execution={self.execution}"
      )
      should_train = Ratio(
        self.env.num_envs
        / self.cfg.wm.num_batches
        / self.cfg.wm.batch_size
        / self.cfg.wm.horizon
        * self.cfg.replay_ratio
      )
      # Resume starts with an empty buffer; update scheduling restarts after warmup.
      num_collected_steps = 0
      wm_learn_time = policy_learn_time = collect_time = 0.0
      loss_dict: dict[str, torch.Tensor] = {}
      for step in range(first_step, self.cfg.max_iterations):
        iteration_start = time.perf_counter()
        t0 = time.perf_counter()
        with timings.measure("collect"):
          collected = self.step()
        collect_time += time.perf_counter() - t0
        num_collected_steps += collected
        total_collected += collected * distributed.world_size()
        iteration_updates = 0
        for _ in range(should_train(step - first_step)):
          ready = self.replay_buffer.num_transitions >= self.cfg.wm.horizon
          data = []
          if ready:
            try:
              data = [
                self.replay_buffer.sample(self.cfg.wm.horizon, self.cfg.wm.batch_size)
                for _ in range(self.cfg.wm.num_batches)
              ]
            except RuntimeError as exc:
              if "no valid sequences" not in str(exc):
                raise
              ready = False
          if not distributed.all_ready(ready, self.device):
            continue
          t0 = time.perf_counter()
          with timings.measure("wm"):
            wm_logs = self.world_model_learner.update(data)
          wm_learn_time += time.perf_counter() - t0
          t0 = time.perf_counter()
          policy_logs = self.policy_learner.update()
          policy_learn_time += time.perf_counter() - t0
          loss_dict.update(wm_logs)
          loss_dict.update(policy_logs)
          updates += 1
          iteration_updates += 1

        if self.cfg.profile:
          stages = {"collect": 0.0, "wm": 0.0, "actor": 0.0, "critic": 0.0}
          stages.update(timings.drain())
          stages.update(self.policy_learner.timings.drain())
          stages["iteration"] = time.perf_counter() - iteration_start
          stages = distributed.max_values(stages, self.device)
          row = {
            "iteration": step,
            "phase": (
              "warmup"
              if step - first_step < self.cfg.profile_warmup_iterations
              else "steady"
            ),
            "world_size": distributed.world_size(),
            "execution": self.execution,
            "global_env_steps": collected * distributed.world_size(),
            "total_env_steps": total_collected,
            "updates": iteration_updates,
            "total_updates": updates,
            "elapsed_s": time.perf_counter() - started,
            "startup_s": self.startup_seconds,
            **{f"{k}_s": v for k, v in stages.items()},
          }
          if profile_file is not None:
            profile_file.write(json.dumps(row) + "\n")
            profile_file.flush()

        if (
          self.cfg.check_sync_interval
          and (step + 1) % self.cfg.check_sync_interval == 0
        ):
          self._check_sync()
        if (
          step + 1
        ) % self.cfg.log_interval == 0 or step + 1 == self.cfg.max_iterations:
          self.logger.log(
            it=step + 1,
            total_it=self.cfg.max_iterations,
            collect_time=collect_time,
            wm_learn_time=wm_learn_time,
            policy_learn_time=policy_learn_time,
            loss_dict=loss_dict,
            num_collected_steps=num_collected_steps,
          )
          collect_time = wm_learn_time = policy_learn_time = 0.0
          num_collected_steps = 0
        if (
          step + 1
        ) % self.cfg.save_interval == 0 or step + 1 == self.cfg.max_iterations:
          self.training_wall_seconds = (
            prior_wall + self.startup_seconds + time.perf_counter() - started
          )
          self._save_outputs(log_dir, step, updates, total_collected)
    finally:
      if profile_file is not None:
        profile_file.close()
      self._cleanup()
