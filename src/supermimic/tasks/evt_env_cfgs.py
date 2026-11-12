"""EVT tracking tasks; the original G1 environment factories stay unchanged."""

from typing import cast

import mjlab.envs.mdp as mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from supermimic.robots.evt import (
  FOOT_GEOM_RE,
  TRACKED_BODIES,
  EvtRobotCfg,
  action_scales,
  get_robot_cfg,
)
from supermimic.tasks import env_cfgs as obs


def gencoord_evt_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_tracking_env_cfg()
  robot = get_robot_cfg()
  cfg.scene.entities = {"robot": robot}
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )
  configure_robot(cfg, robot.parameters)
  motion = cast(MotionCommandCfg, cfg.commands["motion"])
  motion.anchor_body_name = "pelvis"
  motion.body_names = TRACKED_BODIES
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOM_RE
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_pitch_link",)
  cfg.events = {
    k: v
    for k, v in cfg.events.items()
    if k in {"foot_friction", "base_com", "encoder_bias"}
  }
  terms = {}
  obs._add_term(terms, "projected_gravity", mdp.projected_gravity)
  obs._add_term(
    terms, "base_ang_vel", mdp.builtin_sensor, sensor_name="robot/angular-velocity"
  )
  obs._add_term(terms, "joint_pos", mdp.joint_pos_rel, biased=True)
  obs._add_term(terms, "joint_vel", mdp.joint_vel_rel)
  obs._add_term(terms, "actions", mdp.last_action)
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )
  critic = cfg.observations["critic"].terms
  critic["base_ang_vel"].params["sensor_name"] = "robot/angular-velocity"
  critic["base_lin_vel"] = ObservationTermCfg(func=mdp.base_lin_vel)
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      name: ObservationTermCfg(func=getattr(obs, name))
      for name in (
        "root_pos",
        "root_ori",
        "joint_pos",
        "root_lin_vel",
        "root_ang_vel",
        "joint_vel",
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  cfg.observations["goal"] = obs._goal_obs_group()
  cfg.rewards.pop("self_collisions", None)
  cfg.rewards["joint_limit"].weight = -2.0
  cfg.terminations.pop("ee_body_pos", None)
  cfg.viewer.body_name = "waist_pitch_link"
  cfg.sim.nconmax = 128
  cfg.sim.njmax = 1024
  if play:
    cfg.episode_length_s = int(1e9)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.sampling_mode = "start"
  return cfg


def configure_robot(cfg: ManagerBasedRlEnvCfg, parameters: EvtRobotCfg) -> None:
  """Apply one parameter set to both actuators and policy-action mapping."""
  cfg.scene.entities["robot"] = get_robot_cfg(parameters)
  cast(JointPositionActionCfg, cfg.actions["joint_pos"]).scale = action_scales(
    parameters
  )


def maxcoord_evt_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = gencoord_evt_env_cfg(play)
  cfg.observations["full_state"] = ObservationGroupCfg(
    terms={
      name: ObservationTermCfg(
        func=getattr(obs, name), params={"command_name": "motion"}
      )
      for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg
