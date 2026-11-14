"""Read complete checkpoints for playback/export without constructing an env."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def checkpoint_run_dir(path: Path) -> Path:
  """Find the owning run for numbered snapshots, or use a standalone file's dir."""
  return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def load_policy_checkpoint(path: Path, task_id: str) -> tuple[dict, dict, dict]:
  """Return config, world-model bundle and actor from the same complete snapshot.

  New checkpoints embed the JSON-compatible run configuration. Older
  training_state.pt files can use the owning run's config.json instead.
  Optimizer and sampler tensors are loaded on CPU and released on return.
  """
  state = torch.load(path, map_location="cpu", weights_only=True)
  required = {"runner_cfg", "world_model", "actor"}
  if not isinstance(state, dict) or not required.issubset(state):
    raise ValueError(
      "--checkpoint requires a complete training_state.pt or checkpoints/model_*.pt; "
      "use --checkpoint-dir for legacy actor.pt/world_model.pt files"
    )
  if state.get("version") != 1:
    raise ValueError(f"Unsupported checkpoint version: {state.get('version')}")
  if state.get("task_id") != task_id:
    raise ValueError(
      f"Checkpoint task {state.get('task_id')!r} does not match {task_id!r}"
    )
  run_config = state.get("run_config")
  if run_config is None:
    config_path = checkpoint_run_dir(path) / "config.json"
    if not config_path.is_file():
      raise FileNotFoundError(
        f"Older checkpoint has no embedded run configuration; missing {config_path}"
      )
    run_config = json.loads(config_path.read_text())
  # Architecture must always come from the selected snapshot, never latest files.
  run_config = {**run_config, "runner_cfg": state["runner_cfg"]}
  return run_config, state["world_model"], state["actor"]
