"""Export checkpoint policies without motion files or a simulation environment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import onnx
import torch
from torch import nn

from supermimic.models import MultiEncoderAE
from supermimic.representations import (
  GOAL_FRAME_STRIDE,
  GOAL_REL_POS_DIM,
  NUM_GOAL_FRAMES,
  create_representation,
)
from supermimic.runners.exporter import _export_module
from supermimic.tasks.task_spec import resolve_task_specs

if TYPE_CHECKING:
  from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg


class PolicyOnnx(nn.Module):
  """One encoder + shared decoder; input history is temporal, newest frame last."""

  def __init__(self, actor: MultiEncoderAE, task: str):
    super().__init__()
    self.obs_normalizer = actor.obs_normalizer
    self.goal_normalizer = actor.goal_normalizers[task]
    self.encoder = actor.encoders[task]
    self.decoder = actor.decoder

  def forward(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    history_n = self.obs_normalizer(history)
    z = torch.tanh(self.encoder(self.goal_normalizer(goal)))
    return self.decoder(torch.cat([z, history_n.flatten(start_dim=1)], dim=-1))


def _validate_control(control: dict | None, task_id: str) -> dict:
  if not control:
    raise ValueError(
      "Motion-free export requires robot_control in world_model.pt to preserve "
      "the trained PD/action mapping. This checkpoint has no control snapshot; "
      "current robot defaults will not be substituted."
    )
  if control.get("robot", "").lower() != task_id.rsplit("-", 1)[-1].lower():
    raise ValueError("Checkpoint robot does not match the selected task")
  names = control["joint_names"]
  if not names or len(set(names)) != len(names):
    raise ValueError("Control snapshot must have unique joint_names")
  for key in (
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "action_offset",
    "action_scale",
    "joint_effort_limits",
    "joint_velocity_limits",
  ):
    values = control.get(key, [])
    if len(values) != len(names) or not all(math.isfinite(v) for v in values):
      raise ValueError(f"Invalid control snapshot field: {key}")
  for key in ("control_dt", "physics_dt"):
    if not math.isfinite(control[key]) or control[key] <= 0:
      raise ValueError(f"Invalid control snapshot field: {key}")
  return control


def _history_terms(mode: str, representation: str, joints: int, bodies: int) -> dict:
  if mode == "proprio":
    return {
      "projected_gravity": 3,
      "base_ang_vel": 3,
      "joint_pos_minus_default": joints,
      "joint_vel": joints,
      "previous_actions": joints,
    }
  if mode != "rep_state":
    raise ValueError(f"Unsupported actor observation mode: {mode}")
  if representation == "gencoord":
    return {
      "root_height": 1,
      "projected_gravity": 3,
      "root_lin_vel_b": 3,
      "root_ang_vel_b": 3,
      "joint_pos_absolute": joints,
      "joint_vel": joints,
    }
  return {
    "local_body_pos": 3 * bodies,
    "local_body_rot6d": 6 * bodies,
    "local_body_lin_vel": 3 * bodies,
    "local_body_ang_vel": 3 * bodies,
    "root_height": 1,
    "projected_gravity": 3,
  }


def export_policy_checkpoint(
  state_dict: dict[str, torch.Tensor],
  cfg: SuperMimicRunnerCfg,
  control: dict | None,
  *,
  task_id: str,
  body_names: list[str],
  anchor_body_name: str,
  path: Path,
  run_path: str,
  task: Literal["motion", "velocity", "all"] = "motion",
) -> list[Path]:
  """Rebuild only the actor, retaining checkpoint control values at full precision.

  Each graph takes ``history[1,H,D]`` and ``goal[1,G]``, and returns raw
  ``actions[1,J]``. Neither reference arrays nor the world model are exported.
  A JSON sidecar documents the input layout and the complete control snapshot.
  """
  control = _validate_control(control, task_id)
  joints, bodies = len(control["joint_names"]), len(body_names)
  representation = create_representation(
    cfg.representation,
    13 + 2 * joints if cfg.representation == "gencoord" else 13 * bodies,
    joints,
    control["control_dt"],
    round(control["control_dt"] / control["physics_dt"]),
    torch.tensor(control["default_joint_pos"]),
    cfg.actor_obs,
  )
  terms = _history_terms(cfg.actor_obs, cfg.representation, joints, bodies)
  goal_dim = state_dict["goal_normalizers.motion._mean"].numel()
  if goal_dim != NUM_GOAL_FRAMES * (2 * joints + 6):
    raise ValueError("Checkpoint motion goal differs from the current goal layout")
  actor = MultiEncoderAE(
    obs_dim=sum(terms.values()),
    action_dim=joints,
    task_specs=resolve_task_specs(
      cfg.tasks, representation, goal_dim + GOAL_REL_POS_DIM
    ),
    slot_mix=cfg.slot_mix,
    latent_dim=cfg.latent_dim,
    hidden_dims=cfg.actor_hidden_dims,
    latent_bc_weight=cfg.latent_bc_weight,
    history_length=cfg.history_length,
  ).cpu()
  # Strict loading checks architecture, history width and observation mode.
  actor.load_state_dict(state_dict, strict=True)
  actor.eval()
  selected = (
    actor.task_specs
    if task == "all"
    else [spec for spec in actor.task_specs if spec.name == task]
  )
  if not selected:
    raise ValueError(f"Checkpoint has no encoder for task: {task}")
  paths = []
  for spec in selected:
    fraction = cfg.slot_mix.get(spec.name, 0.0)
    if fraction == 0:
      print(
        f"[WARN] '{spec.name}' had zero training slots; this encoder may be untrained."
      )
    contract = {
      "schema_version": 1,
      "format": "supermimic_policy_external_goal",
      "run_path": run_path,
      "task_id": task_id,
      "task": spec.name,
      "training_slot_fraction": fraction,
      "actor_obs_mode": cfg.actor_obs,
      "representation": cfg.representation,
      "history_shape": [1, actor.history_length, actor.obs_dim],
      "history_layout": "frame_major_oldest_to_newest",
      "history_terms": terms,
      "goal_shape": [1, spec.goal_dim],
      "actions_shape": [1, joints],
      "dtype": "float32",
      "normalization": "included_in_graph; supply unnormalized inputs",
      "action_mapping": "joint_target = action_offset + action_scale * actions",
      "action_clipping": "none_in_graph",
      "anchor_body_name": anchor_body_name,
      "body_names": body_names,
      "robot_control": control,
    }
    if spec.name == "motion":
      contract["goal_layout"] = {
        "order": "flatten([joint_pos, joint_vel] per frame), then flatten(relative_anchor_rot6d per frame)",
        "frames": NUM_GOAL_FRAMES,
        "stride_control_steps": GOAL_FRAME_STRIDE,
        "orientation": "reference anchor relative to current robot anchor; first two rotation matrix columns, row-major",
        "joint_positions": "absolute radians in robot_control.joint_names order",
        "clip_end": "clamp future frames to the current clip's final frame",
      }
    else:
      contract["goal_layout"] = {
        "terms": ["vx_b", "vy_b", "wz_b"],
        "units": ["m/s", "m/s", "rad/s"],
        "ranges": [
          cfg.tasks.velocity.lin_vel_x_range,
          cfg.tasks.velocity.lin_vel_y_range,
          cfg.tasks.velocity.ang_vel_z_range,
        ],
      }
    onnx_path = Path(
      _export_module(
        PolicyOnnx(actor, spec.name),
        (torch.zeros(*contract["history_shape"]), torch.zeros(*contract["goal_shape"])),
        ["history", "goal"],
        ["actions"],
        str(path),
        f"policy_{spec.name}.onnx",
      )
    )
    # Keep convenient CSV keys, but avoid the legacy metadata helper's rounding.
    metadata = {
      key: ",".join(map(str, value)) if isinstance(value, list) else str(value)
      for key, value in control.items()
      if key != "parameters"
    }
    metadata.update(
      {
        "robot_control": json.dumps(control, allow_nan=False),
        "policy_contract": json.dumps(contract, allow_nan=False),
        "run_path": run_path,
        "actor_obs_mode": cfg.actor_obs,
        "history_length": str(actor.history_length),
      }
    )
    model = onnx.load(onnx_path)
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    sidecar = onnx_path.with_suffix(".json")
    sidecar.write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    paths.extend([onnx_path, sidecar])
  return paths
