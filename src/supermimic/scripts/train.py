"""Script to train World Model using SuperMimicRunner."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Literal

import mjlab.tasks  # noqa: F401
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import (
  create_env,
  download_registry_motions,
  setup_device,
)
from supermimic.utils import distributed as distributed


@dataclass
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  runner: SuperMimicRunnerCfg

  # Experiment & Environment
  num_envs: int = 4096
  device: str | None = None
  registry_name: str | None = "supermimic/wandb-registry-motions/lafan_run1_subject5"
  motion_file: str | None = None
  seed: int | None = None
  """Optional reproducible model seed; multi-GPU defaults to the environment seed."""
  batch_size_scope: Literal["per_rank", "global"] = "per_rank"
  """Interpret num_envs and learner batch sizes per rank (legacy) or globally."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id, play=False)
    return TrainConfig(
      env=env_cfg,
      runner=SuperMimicRunnerCfg(),
      registry_name=None if task_id.endswith("-EVT") else TrainConfig.registry_name,
    )


def main() -> None:
  # Parse first argument to choose the task.
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    config=mjlab.TYRO_FLAGS,
  )

  device = distributed.initialize(args.device)
  try:
    run_train(chosen_task, args, device)
  finally:
    distributed.shutdown()


def run_train(chosen_task: str, args: TrainConfig, device: str) -> None:
  setup_start = time.perf_counter()
  distributed.check_same(
    (
      chosen_task,
      asdict(args.runner),
      args.num_envs,
      args.motion_file,
      args.seed,
      args.batch_size_scope,
    ),
    "training configuration",
  )
  setup_device(device)
  base_seed = args.seed if args.seed is not None else args.env.seed
  if base_seed is None and distributed.world_size() > 1:
    base_seed = 42
  model_seed = (
    base_seed if distributed.world_size() > 1 or args.seed is not None else None
  )
  if model_seed is not None:
    args.env.seed = model_seed + 1000 * distributed.rank()
    distributed.seed_all(args.env.seed)
  layout = distributed.resolve_batch_layout(
    args.num_envs,
    args.runner.wm.batch_size,
    args.runner.policy.batch_size,
    args.batch_size_scope,
    distributed.world_size(),
  )
  args.num_envs = layout["local"]["num_envs"]
  args.runner.wm.batch_size = layout["local"]["wm_batch"]
  args.runner.policy.batch_size = layout["local"]["policy_batch"]
  if (
    args.num_envs < 1
    or args.runner.wm.batch_size < 1
    or args.runner.policy.batch_size < 1
  ):
    raise ValueError("Environment and learner batch sizes must be positive")
  if args.runner.wm.horizon < 2 or args.runner.buffer_size < args.runner.wm.horizon:
    raise ValueError("Require buffer_size >= wm.horizon >= 2")
  if args.runner.sampling_mode == "adaptive" and distributed.world_size() > 1:
    raise ValueError("Multi-GPU supports return/uniform/start sampling; use return")
  if args.runner.log_interval < 1 or args.runner.save_interval < 1:
    raise ValueError("Log/save intervals must be positive")
  if args.runner.wm.num_batches < 1 or args.runner.policy.num_batches < 1:
    raise ValueError("Learner num_batches must be positive")
  if args.runner.policy.horizon < 1 or args.runner.max_iterations < 1:
    raise ValueError("Policy horizon and max_iterations must be positive")
  if (
    isinstance(args.runner.policy, ActorCriticPolicyLearnerCfg)
    and args.runner.policy.batch_size * args.runner.policy.horizon % 4
  ):
    raise ValueError("Policy batch_size * horizon must be divisible by 4")
  for size in (args.num_envs, args.runner.policy.batch_size):
    active = [f for f in args.runner.slot_mix.values() if f > 0]
    if any(round(size * fraction) == 0 for fraction in active):
      raise ValueError("Each active task needs at least one slot on every rank")
  print(
    f"[INFO] rank={distributed.rank()}/{distributed.world_size()} device={device} "
    f"seed={args.env.seed}; per-rank/global envs={args.num_envs}/"
    f"{args.num_envs * distributed.world_size()}, WM batch={args.runner.wm.batch_size}/"
    f"{args.runner.wm.batch_size * distributed.world_size()}, policy batch="
    f"{args.runner.policy.batch_size}/{args.runner.policy.batch_size * distributed.world_size()}"
  )
  print(f"[INFO] Loading environment: {chosen_task}")

  if args.motion_file:
    # Local motions handed straight to MotionLoader (globs/concatenates).
    if "," in args.motion_file:
      args.env.commands["motion"].motion_file = [
        os.path.expanduser(p.strip()) for p in args.motion_file.split(",")
      ]
    else:
      args.env.commands["motion"].motion_file = os.path.expanduser(args.motion_file)
    args.registry_name = None  # avoid linking a stale registry artifact for lineage
    print(f"[INFO] Using local motion file: {args.motion_file}")
  else:
    motion_file = (
      download_registry_motions(args.registry_name) if distributed.rank() == 0 else None
    )
    motion_file = distributed.broadcast_object(motion_file)
    if motion_file is not None:
      args.env.commands["motion"].motion_file = motion_file

  if not args.env.commands["motion"].motion_file:
    raise ValueError(
      "No reference motion selected. For EVT, convert its motions with "
      "supermimic.scripts.convert_evt_motions and pass --motion-file."
    )

  # Auto-detect representation from task_id if not explicitly set
  if "MaxCoord" in chosen_task and args.runner.representation == "gencoord":
    args.runner.representation = "maxcoord"
    print("[INFO] Auto-detected maxcoord representation from task_id")

  args.env.commands["motion"].sampling_mode = (
    "external" if args.runner.sampling_mode == "return" else args.runner.sampling_mode
  )

  env = create_env(args.env, args.num_envs, device)

  # Init Runner
  try:
    runner = SuperMimicRunner(
      env=env,
      cfg=args.runner,
      task_id=chosen_task,
      registry_name=args.registry_name,
      device=device,
      model_seed=model_seed,
      batch_layout=layout,
    )
  except BaseException:
    env.close()
    raise
  runner.startup_seconds = time.perf_counter() - setup_start
  runner.run()


if __name__ == "__main__":
  main()
