"""Evaluate a local training_state.pt with a fixed seed and no exploration noise.

Use the same motion set, seed, environment count and step budget across runs.
This is a fixed-budget motion-tracking evaluation, not a per-clip completion test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import tyro
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.robots.control import restore_control
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.train_utils import create_env, setup_device
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.distributed import seed_all


def main(
  checkpoint: Path,
  motion_file: str,
  output: Path | None = None,
  num_envs: int = 128,
  steps: int = 1000,
  seed: int = 1234,
  device: str = "cuda:0",
) -> None:
  if num_envs < 1 or steps < 1:
    raise ValueError("num_envs and steps must be positive")
  setup_device(device)
  torch.cuda.set_device(device)
  seed_all(seed)
  ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
  cfg = load_runner_cfg({"runner_cfg": ckpt["runner_cfg"]})
  if cfg.slot_mix.get("motion", 0) <= 0:
    raise ValueError("Checkpoint has no trained motion encoder")
  cfg.slot_mix = {"motion": 1.0}
  env_cfg = load_env_cfg(ckpt["task_id"], play=True)
  env_cfg.seed = seed
  env_cfg.commands["motion"].motion_file = motion_file
  env_cfg.commands["motion"].sampling_mode = "start"
  restore_control(env_cfg, ckpt["world_model"].get("robot_control"))
  env = create_env(env_cfg, num_envs, device)
  started = time.perf_counter()
  try:
    wm = build_world_model(cfg, env, device)
    wm.load_bundle(ckpt["world_model"])
    actor = build_actor(cfg, wm, device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    rep = wm.representation
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    spec = next(ts for ts in actor.task_specs if ts.name == "motion")
    obs, _ = env.reset()
    history = None
    previous_done = torch.ones(num_envs, device=device, dtype=torch.bool)
    reward_sum = torch.zeros((), device=device)
    failures = torch.zeros((), device=device, dtype=torch.long)
    episodes = torch.zeros_like(failures)
    with torch.inference_mode():
      for _ in range(steps):
        state = obs["full_state"]
        rep_state, aux = rep.represent(state), rep.extract_aux_state(state)
        context = wm.sim_context.read(env.unwrapped)
        sensor = rep.compute_actor_sensor(
          rep_state,
          context,
          aux_state=aux,
          sim_actor_obs=obs["actor"] if cfg.actor_obs == "proprio" else None,
          noisy=False,
        )
        if history is None:
          history = sensor.unsqueeze(1).repeat(1, cfg.history_length, 1)
        else:
          history = torch.cat([history[:, 1:], sensor.unsqueeze(1)], dim=1)
          history[previous_done] = sensor[previous_done, None, :]
        goal = spec.extract_training_goal(motion_cmd.motion, motion_cmd.time_steps, aux)
        actions = actor(build_actor_obs_td({"motion": goal}, history))
        obs, reward, done, _ = env.step(actions)
        previous_done = done.bool()
        reward_sum += reward.sum()
        failures += env.unwrapped.termination_manager.terminated.sum()
        episodes += done.sum()
    result = {
      "checkpoint": str(checkpoint),
      "task": ckpt["task_id"],
      "motion_file": motion_file,
      "seed": seed,
      "num_envs": num_envs,
      "steps": steps,
      "mean_step_reward": reward_sum.item() / (num_envs * steps),
      "failures": failures.item(),
      "completed_episodes": episodes.item(),
      "failures_per_1000_env_steps": failures.item() * 1000 / (num_envs * steps),
      "training_wall_s": ckpt.get("training_wall_s"),
      "training_env_steps": ckpt["collected"],
      "training_updates": ckpt["updates"],
      "evaluation_wall_s": time.perf_counter() - started,
    }
    output = output or checkpoint.parent / "evaluations.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as f:
      f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  tyro.cli(main)
