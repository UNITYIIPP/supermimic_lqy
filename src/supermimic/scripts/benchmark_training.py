"""Run sequential full-training scaling experiments with identical hyperparameters.

Example: uv run -m supermimic.scripts.benchmark_training --motion-file PATH
Results include the exact command, raw logs, profiles, and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from supermimic.scripts.summarize_training import summarize


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--motion-file", required=True)
  parser.add_argument("--task", default="SuperMimic-GenCoord-EVT")
  parser.add_argument("--output", type=Path, default=Path("benchmarks/distributed"))
  parser.add_argument("--gpus", default="0,1,2,3")
  parser.add_argument(
    "--cases",
    nargs="+",
    default=["1-split", "2-eager", "2-split", "4-split", "4-split-weak"],
  )
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--wm-batch", type=int, default=2048)
  parser.add_argument("--policy-batch", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=200)
  parser.add_argument("--warmup", type=int, default=50)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
  args = parser.parse_args()
  if args.warmup >= args.iterations:
    parser.error("--warmup must be smaller than --iterations")
  args.output.mkdir(parents=True, exist_ok=True)
  devices = args.gpus.split(",")
  summaries = []
  for case in args.cases:
    parts = case.split("-")
    world, mode = int(parts[0]), parts[1]
    weak = len(parts) == 3 and parts[2] == "weak"
    if world > len(devices) or mode not in {"eager", "split"}:
      parser.error(f"Invalid case or insufficient GPUs: {case}")
    divisor = 1 if weak else world
    if any(n % divisor for n in (args.num_envs, args.wm_batch, args.policy_batch)):
      parser.error(f"Global batches must be divisible by {divisor}")
    out = args.output / case
    out.mkdir(exist_ok=True)
    cmd = [sys.executable]
    if world > 1:
      cmd += [
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
      ]
    cmd += [
      "-m",
      "supermimic.scripts.train",
      args.task,
      "--motion-file",
      args.motion_file,
      "--seed",
      str(args.seed),
      "--num-envs",
      str(args.num_envs // divisor),
      "--runner.wm.batch-size",
      str(args.wm_batch // divisor),
      "--runner.policy.batch-size",
      str(args.policy_batch // divisor),
      "--runner.max-iterations",
      str(args.iterations),
      "--runner.profile-warmup-iterations",
      str(args.warmup),
      "--runner.execution",
      mode,
      "--runner.profile",
      "True",
      "--runner.export-onnx",
      "False",
      "--runner.wandb-enabled",
      "False",
      "--runner.log-root",
      str(out / "logs"),
      "--runner.run-name",
      case,
      *args.extra,
    ]
    env = dict(
      os.environ,
      CUDA_VISIBLE_DEVICES=",".join(devices[:world]),
      PYTHONUNBUFFERED="1",
      OMP_NUM_THREADS="1",
    )
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
    (out / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    (out / "environment.json").write_text(
      json.dumps(
        {
          key: env[key]
          for key in (
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "TORCHINDUCTOR_COMPILE_THREADS",
          )
        },
        indent=2,
      )
      + "\n"
    )
    old_profiles = set(out.rglob("performance.jsonl"))
    print(f"Running {case}; log: {out / 'stdout.log'}", flush=True)
    started = time.perf_counter()
    with (out / "stdout.log").open("w") as log:
      subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    profiles = set(out.rglob("performance.jsonl")) - old_profiles
    if len(profiles) != 1:
      raise RuntimeError(f"Expected one profile, found {profiles}")
    result = summarize(profiles.pop())
    result.update(case=case, process_wall_s=time.perf_counter() - started)
    summaries.append(result)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
  main()
