"""Summarize full-training profiles: uv run -m supermimic.scripts.summarize_training PATH..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
  rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  if not rows:
    raise ValueError(f"Empty profile: {path}")
  steady = [r for r in rows if r["phase"] == "steady"]
  warmup = [r for r in rows if r["phase"] == "warmup"]
  seconds = sum(r["iteration_s"] for r in steady)
  result = {
    "profile": str(path),
    "world_size": rows[0]["world_size"],
    "execution": rows[0]["execution"],
    "startup_s": rows[0].get("startup_s"),
    "warmup_training_s": sum(r["iteration_s"] for r in warmup),
    "steady_iterations": len(steady),
    "steady_updates": sum(r["updates"] for r in steady),
    "steady_training_s": seconds,
    "steady_env_steps_per_s": sum(r["global_env_steps"] for r in steady) / seconds
    if seconds
    else None,
    "steady_updates_per_s": sum(r["updates"] for r in steady) / seconds
    if seconds
    else None,
    "stage_seconds_per_iteration": {
      k: sum(r[f"{k}_s"] for r in steady) / len(steady) if steady else None
      for k in ("collect", "wm", "actor", "critic")
    },
    "note": "Stage/iteration timers exclude logging, saving and evaluation; elapsed_s includes earlier I/O. Compare identical configurations and warmed compiler caches.",
  }
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("profiles", nargs="+", type=Path)
  args = parser.parse_args()
  print(json.dumps([summarize(p) for p in args.profiles], indent=2))


if __name__ == "__main__":
  main()
