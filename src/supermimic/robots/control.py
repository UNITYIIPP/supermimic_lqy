"""Capture the effective robot control settings and restore EVT run snapshots."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import torch

from supermimic.robots.evt import EvtEntityCfg, EvtRobotCfg, JointCfg


def control_snapshot(env) -> dict | None:
  """Read gains/scales from the running env, never from a second parameter table."""
  cfg = env.cfg.scene.entities.get("robot")
  if not isinstance(cfg, EvtEntityCfg):
    return None
  robot = env.scene["robot"]
  model = env.sim.mj_model
  action = env.action_manager.get_term("joint_pos")
  names = list(robot.joint_names)
  by_name = {j.name: j for j in cfg.parameters.joints}
  ids = {
    model.actuator(i).name.split("/")[-1]: i for i in robot.indexing.ctrl_ids.tolist()
  }
  ctrl = [ids[name] for name in names]

  def row(value):
    if not isinstance(value, torch.Tensor):
      return [float(value)] * len(names)
    value = value.detach().cpu()
    return (
      value.expand(len(names)).tolist()
      if value.ndim == 0
      else value.reshape(-1, len(names))[0].tolist()
    )

  scale = row(action.scale)
  offset = row(action.offset)
  parameters = asdict(cfg.parameters)
  parameters["xml_path"] = str(parameters["xml_path"])
  parameters["mesh_dir"] = str(parameters["mesh_dir"])
  return {
    "robot": "evt",
    "parameters": parameters,
    "joint_names": names,
    "joint_stiffness": model.actuator_gainprm[ctrl, 0].tolist(),
    "joint_damping": (-model.actuator_biasprm[ctrl, 2]).tolist(),
    "joint_effort_limits": [float(model.actuator_forcerange[i, 1]) for i in ctrl],
    "joint_velocity_limits": [by_name[n].velocity_limit for n in names],
    "velocity_limits_enforced": False,
    "default_joint_pos": row(robot.data.default_joint_pos),
    "action_offset": offset,
    "action_scale": scale,
    "control_dt": float(env.step_dt),
    "physics_dt": float(model.opt.timestep),
    "delay_min_lag": cfg.parameters.delay_min_lag,
    "delay_max_lag": cfg.parameters.delay_max_lag,
  }


def restore_control(env_cfg, snapshot: dict | None) -> None:
  """Rebuild EVT actuators and action mapping from a recorded run configuration."""
  if not snapshot:
    return
  is_evt = isinstance(env_cfg.scene.entities.get("robot"), EvtEntityCfg)
  if snapshot.get("robot") != "evt" or not is_evt:
    raise ValueError("Robot task does not match the run's control snapshot")
  from supermimic.tasks.evt_env_cfgs import configure_robot

  raw = dict(snapshot["parameters"])
  raw["xml_path"] = Path(raw["xml_path"])
  raw["mesh_dir"] = Path(raw["mesh_dir"])
  raw["root_pos"] = tuple(raw["root_pos"])
  names = snapshot["joint_names"]
  effective = {name: i for i, name in enumerate(names)}
  raw["joints"] = tuple(
    replace(
      JointCfg(**j),
      kp=snapshot["joint_stiffness"][effective[j["name"]]],
      kd=snapshot["joint_damping"][effective[j["name"]]],
      effort_limit=snapshot["joint_effort_limits"][effective[j["name"]]],
      action_scale=snapshot["action_scale"][effective[j["name"]]],
      default_pos=snapshot["default_joint_pos"][effective[j["name"]]],
    )
    for j in raw["joints"]
  )
  configure_robot(env_cfg, EvtRobotCfg(**raw))
  action = env_cfg.actions["joint_pos"]
  action.use_default_offset = False
  action.offset = dict(zip(names, snapshot["action_offset"], strict=True))
  env_cfg.sim.mujoco.timestep = snapshot["physics_dt"]
  env_cfg.decimation = round(snapshot["control_dt"] / snapshot["physics_dt"])
