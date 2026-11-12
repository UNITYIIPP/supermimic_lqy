"""EVT mechanical model and the single source of control parameters.

Edit ``EVT_CFG`` / ``default_joints`` for new runs. Velocity limits are nominal
specifications, not hard velocity constraints in MuJoCo. Passive joint dynamics
come from the XML; ``kd`` below is the separate active PD derivative gain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

TRACKED_BODIES = (
  "pelvis",
  "hip_roll_l_link",
  "knee_pitch_l_link",
  "ankle_roll_l_link",
  "hip_roll_r_link",
  "knee_pitch_r_link",
  "ankle_roll_r_link",
  "waist_pitch_link",
  "shoulder_roll_l_link",
  "elbow_pitch_l_link",
  "wrist_roll_l_link",
  "shoulder_roll_r_link",
  "elbow_pitch_r_link",
  "wrist_roll_r_link",
)
FOOT_GEOM_RE = r"^foot_(left|right)_.*$"
_ASSET_DIR = (
  Path(__file__).resolve().parents[4] / "allmimic" / "assets" / "robots" / "evt"
)


@dataclass(frozen=True)
class JointCfg:
  name: str
  kp: float
  kd: float
  effort_limit: float
  velocity_limit: float
  default_pos: float = 0.0
  action_scale: float = 0.25

  def __post_init__(self) -> None:
    # Tyro validates nested dataclass defaults strictly, including numeric types.
    for name in (
      "kp",
      "kd",
      "effort_limit",
      "velocity_limit",
      "default_pos",
      "action_scale",
    ):
      object.__setattr__(self, name, float(getattr(self, name)))


def default_joints() -> tuple[JointCfg, ...]:
  """Canonical kinematic order; values supplied for this EVT experiment."""
  joints = []
  for side in ("l", "r"):
    for name, kp, kd, effort, velocity, q0 in (
      ("hip_pitch", 300, 15, 235, 16.755160819145562, -0.25),
      ("hip_roll", 300, 15, 235, 16.755160819145562, 0.0),
      ("hip_yaw", 150, 7.5, 150, 13.82300767579509, 0.0),
      ("knee_pitch", 350, 15, 400, 11.100294042683936, 0.5),
      ("ankle_pitch", 30, 3.75, 55, 14.137166941154069, -0.25),
      ("ankle_roll", 16.8, 2.1, 55, 14.137166941154069, 0.0),
    ):
      joints.append(JointCfg(f"{name}_{side}_joint", kp, kd, effort, velocity, q0))
  for name, kd in (("yaw", 7.5), ("roll", 15), ("pitch", 15)):
    joints.append(JointCfg(f"waist_{name}_joint", 400, kd, 91, 9.42477796076938))
  for side in ("l", "r"):
    for name, kp, kd, effort, velocity, q0 in (
      ("shoulder_pitch", 150, 5, 90, 7.218332720398149, 0.0),
      ("shoulder_roll", 50, 2.5, 90, 7.218332720398149, 0.3 if side == "l" else -0.3),
      ("shoulder_yaw", 50, 2.5, 50, 11.455294012539582, 0.0),
      ("elbow_pitch", 150, 5, 50, 11.455294012539582, -0.3),
      ("elbow_yaw", 50, 5, 36, 9.739, 0.0),
      ("wrist_pitch", 20, 2, 36, 9.739, 0.0),
      ("wrist_roll", 20, 2, 36, 9.739, 0.0),
    ):
      joints.append(JointCfg(f"{name}_{side}_joint", kp, kd, effort, velocity, q0))
  return tuple(joints)


@dataclass(frozen=True)
class EvtRobotCfg:
  xml_path: Path = _ASSET_DIR / "urdf" / "evt2_simple.xml"
  mesh_dir: Path = _ASSET_DIR / "meshes"
  root_pos: tuple[float, float, float] = (0.0, 0.0, 0.95)
  soft_joint_pos_limit_factor: float = 0.9
  delay_min_lag: int = 0
  delay_max_lag: int = 0
  joints: tuple[JointCfg, ...] = field(default_factory=default_joints)

  def __post_init__(self) -> None:
    names = [j.name for j in self.joints]
    if len(names) != 29 or len(set(names)) != 29:
      raise ValueError("EVT requires exactly 29 unique joint configurations")
    if not 0 < self.soft_joint_pos_limit_factor <= 1:
      raise ValueError("soft_joint_pos_limit_factor must be in (0, 1]")
    if not 0 <= self.delay_min_lag <= self.delay_max_lag:
      raise ValueError("invalid actuator delay range")
    for j in self.joints:
      values = (
        j.kp,
        j.kd,
        j.effort_limit,
        j.velocity_limit,
        j.default_pos,
        j.action_scale,
      )
      if not all(math.isfinite(x) for x in values):
        raise ValueError(f"non-finite control parameter: {j.name}")
      if min(j.kp, j.effort_limit, j.velocity_limit, j.action_scale) <= 0 or j.kd < 0:
        raise ValueError(f"invalid control parameter: {j.name}")


EVT_CFG = EvtRobotCfg()


@dataclass
class EvtEntityCfg(EntityCfg):
  """Keep resolved parameters on the environment for logging and export."""

  parameters: EvtRobotCfg = field(default_factory=lambda: EVT_CFG)


def get_spec(cfg: EvtRobotCfg) -> mujoco.MjSpec:
  """Prepare a private MjSpec; never edit the source XML or mesh files."""
  spec = mujoco.MjSpec.from_file(str(cfg.xml_path))
  assets = {}
  for mesh in spec.meshes:
    path = cfg.xml_path.parent / spec.meshdir / mesh.file
    if not path.is_file():
      path = cfg.mesh_dir / Path(mesh.file).name
    assets[mesh.file] = path.read_bytes()
  spec.assets = assets
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  spec.option.disableactuator = 0
  for geom in list(spec.geoms):
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
      spec.delete(geom)  # The tracking scene supplies its own ground.
  names = [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
  if names != [j.name for j in cfg.joints]:
    raise ValueError(f"EVT model joint order does not match configuration: {names}")
  for j in cfg.joints:
    joint = spec.joint(j.name)
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.actfrcrange[:] = np.array([-j.effort_limit, j.effort_limit])
  return spec


def get_robot_cfg(cfg: EvtRobotCfg | None = None) -> EvtEntityCfg:
  cfg = EVT_CFG if cfg is None else cfg
  return EvtEntityCfg(
    parameters=cfg,
    spec_fn=partial(get_spec, cfg),
    init_state=EntityCfg.InitialStateCfg(
      pos=cfg.root_pos,
      joint_pos={j.name: j.default_pos for j in cfg.joints},
      joint_vel={".*": 0.0},
    ),
    sort_actuators=True,
    articulation=EntityArticulationInfoCfg(
      soft_joint_pos_limit_factor=cfg.soft_joint_pos_limit_factor,
      actuators=tuple(
        BuiltinPositionActuatorCfg(
          target_names_expr=(j.name,),
          stiffness=j.kp,
          damping=j.kd,
          effort_limit=j.effort_limit,
          delay_min_lag=cfg.delay_min_lag,
          delay_max_lag=cfg.delay_max_lag,
        )
        for j in cfg.joints
      ),
    ),
  )


def action_scales(cfg: EvtRobotCfg) -> dict[str, float]:
  return {j.name: j.action_scale for j in cfg.joints}
