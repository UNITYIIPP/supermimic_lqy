"""Causal input, replay boundaries, imagination gradients and export parity."""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from tensordict import TensorDict

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.models import MultiEncoderAE
from supermimic.replay_buffer import FieldSpec, ReplayBuffer
from supermimic.representations import (
  _assemble_goal,
  create_representation,
  motion_anchor_window,
  motion_goal_frame,
  motion_goal_window,
)
from supermimic.runners.policy_export import export_policy_checkpoint
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.tasks.task_spec import (
  MotionTaskCfg,
  TasksCfg,
  build_actor_obs_td,
  resolve_task_specs,
)
from supermimic.utils.history import append_history


def motion_fixture():
  q = torch.arange(20.0).view(-1, 1).expand(-1, 29).clone()
  motion = SimpleNamespace(
    joint_pos=q,
    joint_vel=q * 7 + 3,
    root_pos_w=torch.stack([torch.arange(20.0), torch.zeros(20), torch.ones(20)], -1),
    root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(20, 1),
    motion_ends=torch.tensor([10, 20]),
  )
  motion.body_state = lambda t: (
    motion.root_pos_w[t, None],
    motion.root_quat_w[t, None],
    torch.zeros(len(t), 1, 3),
    torch.zeros(len(t), 1, 3),
  )
  return motion


def actor_fixture(mode="causal_history", sensor_h=5, goal_h=5, mix=None):
  cfg = SuperMimicRunnerCfg(
    history_length=sensor_h,
    actor_hidden_dims=(24, 16),
    latent_dim=8,
    tasks=TasksCfg(motion=MotionTaskCfg(mode=mode, history_length=goal_h)),
    slot_mix=mix or {"motion": 1.0, "velocity": 0.0},
  )
  rep = create_representation("maxcoord", 182, 29, 0.02, 4, torch.zeros(29), "proprio")
  specs = resolve_task_specs(cfg.tasks, rep, 670)
  actor = MultiEncoderAE(93, 29, specs, cfg.slot_mix, 8, (24, 16), 0.1, sensor_h)
  return cfg, rep, specs, actor


def test_history_padding_resets_and_gradients():
  x = torch.randn(2, 3, requires_grad=True)
  h = append_history(None, x, 5)
  y = torch.randn(2, 3, requires_grad=True)
  new = append_history(h, y, 5, torch.tensor([False, True]))
  torch.testing.assert_close(new[0, :-1], x[0].expand(4, -1))
  torch.testing.assert_close(new[1], y[1].expand(5, -1))
  torch.testing.assert_close(h, x[:, None].expand(-1, 5, -1))
  new.sum().backward()
  torch.testing.assert_close(x.grad[0], torch.full((3,), 4.0))
  torch.testing.assert_close(x.grad[1], torch.zeros(3))
  torch.testing.assert_close(y.grad[1], torch.full((3,), 5.0))


def test_actor_is_causal_and_critic_still_uses_future():
  _, rep, _, actor = actor_fixture()
  motion = motion_fixture()
  aux = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
  goal_history = None
  for t in range(5):
    frame = motion_goal_frame(motion, torch.tensor([t]), aux)
    torch.testing.assert_close(frame[:, 29:58], motion.joint_vel[t : t + 1])
    goal_history = append_history(goal_history, frame, 5)
  history = torch.randn(1, 5, 93)
  actor.eval()
  before = actor(build_actor_obs_td({"motion": goal_history}, history))
  critic_before, _ = rep.compute_wm_goal(motion, torch.tensor([4]), aux)
  # Change everything strictly after t, including supplied velocities.
  motion.joint_pos[5:] += 100
  motion.joint_vel[5:] -= 200
  motion.root_pos_w[5:] += 10
  motion.root_quat_w[5:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
  rebuilt = None
  for t in range(5):
    rebuilt = append_history(
      rebuilt, motion_goal_frame(motion, torch.tensor([t]), aux), 5
    )
  after = actor(build_actor_obs_td({"motion": rebuilt}, history))
  torch.testing.assert_close(rebuilt, goal_history, atol=0, rtol=0)
  torch.testing.assert_close(after, before, atol=0, rtol=0)
  critic_after, _ = rep.compute_wm_goal(motion, torch.tensor([4]), aux)
  assert critic_before.shape == (1, 670)
  assert not torch.equal(critic_before, critic_after)


def test_normalization_is_per_frame_and_aux_loss_supports_mixed_tasks():
  _, _, _, actor = actor_fixture(mix={"motion": 0.5, "velocity": 0.5})
  goals = torch.randn(4, 5, 64)
  td = build_actor_obs_td(
    {"motion": goals, "velocity": torch.randn(4, 3)}, torch.randn(4, 5, 93)
  )
  actor.update_normalization(td)
  torch.testing.assert_close(
    actor.goal_normalizers["motion"].mean, goals[:2, -1].mean(0)
  )
  assert actor.goal_normalizers["motion"]._mean.shape == (1, 64)
  actor(td).square().mean().backward()
  for name in ("motion", "velocity"):
    assert actor.encoders[name][0].weight.grad is not None
  assert actor.encoders["motion"][0].in_features == 320
  assert actor.decoder[0].in_features == 465 + 8


class TrackingHarness(MultiTaskWorldModelEnv):
  """Small differentiable dynamics; exercise production reset/step/goal plumbing."""

  def _wm_step(self, rep, aux, actions, last, context):
    next_rep = rep + actions.mean(-1, keepdim=True) * 0.01
    angle = next_rep[:, :1]
    quat = torch.cat(
      [
        torch.cos(angle / 2),
        torch.zeros_like(angle).expand(-1, 2),
        torch.sin(angle / 2),
      ],
      -1,
    )
    next_aux = torch.cat([aux[:, :3], quat], -1)
    return (
      next_aux,
      next_rep,
      next_rep[:, :58],
      actions[:, 0] * 0,
      -next_rep.square().mean(-1),
    )


def replay_fixture():
  schema = {
    "state": FieldSpec(221),
    "prev_action": FieldSpec(29),
    "sim_ctx": FieldSpec(33),
    "motion_time": FieldSpec(dtype=torch.long),
    "actor_obs": FieldSpec(93),
    "motion_goal": FieldSpec(64),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
  }
  replay = ReplayBuffer(schema, 8, 2)
  motion = motion_fixture()
  for t in range(12):  # wrap buffer and cross a terminal boundary
    f = t % 4
    state = torch.zeros(2, 221)
    state[:, 3] = 1
    state[:, 7:] = f
    goal = motion_goal_frame(motion, torch.full((2,), f), state[:, :7])
    replay.insert(
      TensorDict(
        {
          "state": state,
          "prev_action": torch.zeros(2, 29),
          "sim_ctx": torch.zeros(2, 33),
          "motion_time": torch.full((2,), f),
          "actor_obs": torch.full((2, 93), float(f)),
          "motion_goal": goal,
          "terminated": torch.full((2,), f == 3),
          "truncated": torch.zeros(2, dtype=torch.bool),
        },
        [2],
      )
    )
  return replay, motion


@pytest.mark.parametrize(
  ("sensor_h", "goal_h"), [(5, 5), (1, 5), (5, 1), (3, 5), (5, 3)]
)
def test_replay_and_rollout_histories(sensor_h, goal_h):
  _, _, specs, _ = actor_fixture(sensor_h=sensor_h, goal_h=goal_h)
  replay, motion = replay_fixture()

  def wm_goal(m, t, aux):
    pos, quat = motion_anchor_window(m, t)
    return _assemble_goal(motion_goal_window(m, t), aux[:, :3], aux[:, 3:7], pos, quat)

  rep = SimpleNamespace(
    represent=lambda s: s[:, 7:],
    extract_aux_state=lambda s: s[:, :7],
    joint_state=lambda s: s[:, :58],
    add_proprio_noise=lambda x: x,
    compute_actor_sensor=lambda s, ctx, **kw: s[:, :93],
    compute_wm_goal=wm_goal,
  )
  cfg = SimpleNamespace(
    rewards={"track": SimpleNamespace(weight=1.0)},
    terminations={
      k: SimpleNamespace(params={"threshold": 1.0})
      for k in ("anchor_pos", "anchor_ori")
    },
  )
  command = SimpleNamespace(motion=motion, motion_anchor_body_index=0)
  env = SimpleNamespace(
    cfg=cfg,
    num_actions=29,
    step_dt=0.02,
    command_manager=SimpleNamespace(get_term=lambda _: command),
  )
  env.unwrapped = env
  wm = TrackingHarness(None, rep, env, 32, replay, specs[:1], sensor_h, 0.05)
  batch = wm.sample_initial_states()
  obs = wm.reset(batch)
  times = batch["motion_time"]
  expected = (times[:, None] + torch.arange(1 - goal_h, 1)).clamp(min=0)
  torch.testing.assert_close(obs["goal_motion"][..., 0], expected.float())
  expected_s = (times[:, None] + torch.arange(1 - sensor_h, 1)).clamp(min=0)
  torch.testing.assert_close(obs["history"][..., 0], expected_s.float())
  assert obs["critic"].shape == (32, 917)
  # Repeated reads neither advance the window nor change cached orientations.
  saved = obs["goal_motion"].clone()
  torch.testing.assert_close(wm.get_observations()["goal_motion"], saved)
  actions = torch.ones(32, 29, requires_grad=True)
  next_obs, reward, _, _ = wm.step(actions)
  torch.testing.assert_close(next_obs["goal_motion"][:, :-1], saved[:, 1:])
  expected_frame = motion_goal_frame(motion, times + 1, wm.aux_state)
  torch.testing.assert_close(next_obs["goal_motion"][:, -1], expected_frame)
  next_obs["goal_motion"][:, -1, -6:].sum().add(reward.sum()).backward()
  assert actions.grad is not None and torch.isfinite(actions.grad).all()
  assert actions.grad.abs().sum() > 0


def test_real_collection_uses_same_goal_window_and_resets():
  cfg, rep, specs, _ = actor_fixture()
  runner = SuperMimicRunner.__new__(SuperMimicRunner)
  runner.cfg, runner._representation, runner.motion = cfg, rep, motion_fixture()
  runner._active_task_specs = specs[:1]
  runner._sim_goal_histories = {}
  runner._goal_history_reset = None
  runner._sim_history = torch.zeros(2, 5, 93)
  aux = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).expand(2, -1)
  for t in range(5):
    td = runner._build_sim_obs_td(
      torch.zeros(2, 214),
      aux,
      torch.zeros(2, 93),
      torch.full((2,), t),
      torch.zeros(2, 33),
    )
  torch.testing.assert_close(td["goal_motion"][0, :, 0], torch.arange(5.0))
  runner._advance_sim_history(torch.zeros(2, 93), torch.tensor([True, False]))
  td = runner._build_sim_obs_td(
    torch.zeros(2, 214),
    aux,
    torch.zeros(2, 93),
    torch.tensor([0, 5]),
    torch.zeros(2, 33),
  )
  torch.testing.assert_close(td["goal_motion"][0, :, 0], torch.zeros(5))
  torch.testing.assert_close(td["goal_motion"][1, :, 0], torch.arange(1.0, 6.0))


def test_reset_sensor_padding_shares_current_noise():
  wm = TrackingHarness.__new__(TrackingHarness)
  wm.num_envs, wm.device = 2, "cpu"
  wm.representation = SimpleNamespace(add_proprio_noise=lambda x: x + 0.5)
  wm._latest_sensor = torch.full((2, 93), 0.75)
  padded = torch.tensor(
    [[True, True, True, True, False], [True, True, False, False, False]]
  )
  past = wm._warmstart_past_sensors(torch.zeros(2, 5, 93), padded)
  torch.testing.assert_close(past[0], wm._latest_sensor[0].expand(4, -1))
  torch.testing.assert_close(past[1], torch.full((4, 93), 0.5))


@pytest.mark.parametrize("mode", ["future_window", "causal_history"])
def test_checkpoint_export_parity_and_legacy_config(tmp_path, mode):
  cfg, _, _, actor = actor_fixture(mode)
  raw = asdict(cfg)
  if mode == "future_window":
    del raw["tasks"]["motion"]  # old runs did not serialize this field
  restored = load_runner_cfg({"runner_cfg": raw})
  assert restored.tasks.motion.mode == mode
  control = {
    key: [1.0] * 29
    for key in (
      "joint_stiffness",
      "joint_damping",
      "default_joint_pos",
      "action_offset",
      "action_scale",
      "joint_effort_limits",
      "joint_velocity_limits",
    )
  }
  control.update(
    robot="evt",
    joint_names=[f"j{i}" for i in range(29)],
    control_dt=0.02,
    physics_dt=0.005,
  )
  export_policy_checkpoint(
    actor.state_dict(),
    restored,
    control,
    task_id="SuperMimic-MaxCoord-EVT",
    body_names=[f"b{i}" for i in range(14)],
    anchor_body_name="b0",
    path=tmp_path,
    run_path="local-test",
  )
  model = onnx.load(tmp_path / "policy_motion.onnx")
  onnx.checker.check_model(model, full_check=True)
  causal = mode == "causal_history"
  assert [x.name for x in model.graph.input] == [
    "history",
    "goal_history" if causal else "goal",
  ]
  reference = ReferenceEvaluator(model)
  actor.eval()
  for _ in range(3):
    history = torch.randn(1, 5, 93)
    goal = torch.randn(1, 5, 64) if causal else torch.randn(1, 640)
    with torch.no_grad():
      expected = actor(build_actor_obs_td({"motion": goal}, history)).numpy()
    actual = reference.run(
      None,
      {"history": history.numpy(), "goal_history" if causal else "goal": goal.numpy()},
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
  copied = deepcopy(actor)
  copied.load_state_dict(actor.state_dict(), strict=True)
