"""Play a SuperMimic multi-encoder actor against the sim."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.robots.control import restore_control
from supermimic.runners.policy_export import export_policy_checkpoint
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.video_recorder import PlaybackVideoRecorder, video_frame_count
from supermimic.scripts.viewers import (
  SuperMimicHeadlessViewer,
  SuperMimicNativeViewer,
  SuperMimicViserViewer,
  create_wm_vis_model,
)
from supermimic.utils.wandb_artifacts import (
  download_artifact_cached,
  download_model_dir,
)

PLAY_FLAGS = tuple(f for f in mjlab.TYRO_FLAGS if f is not tyro.conf.FlagConversionOff)


@dataclass
class PlayConfig:
  wandb_path: str | None = None
  checkpoint_dir: Path | None = None
  """Local training directory containing config.json, actor.pt and world_model.pt."""
  export_onnx: bool = False
  """Export a policy without embedded motion before playback."""
  export_only: bool = False
  """Export and exit without loading motions or creating a simulation/viewer."""
  export_dir: Path | None = None
  """Default: <checkpoint_dir>/exports (W&B uses its downloaded model directory)."""
  export_task: Literal["motion", "velocity", "all"] = "motion"
  motion_file: str | None = None
  initial_task: Literal["motion", "velocity"] = "motion"
  """Task selected when the viewer opens. Switch live via the GUI dropdown."""
  device: str | None = None
  num_steps_rollout: int = 1000000
  viewer: Literal["auto", "native", "viser", "none"] = "viser"
  video: bool = False
  """Save a video starting with the first policy step."""
  video_duration: float = 30.0
  """Video duration in simulation seconds, rounded up to a control step."""
  video_dir: Path = Path("videos/play")
  video_width: int = 1280
  video_height: int = 720
  video_overlay: Literal["robot", "reference", "all"] = "reference"
  """Video shows robot only, robot + reference, or also the WM prediction."""
  motion_start: Literal["start", "random"] | None = None
  """Clip reset position. Video defaults to start; regular play keeps its behavior."""

  def __post_init__(self) -> None:
    if (self.wandb_path is None) == (self.checkpoint_dir is None):
      raise ValueError("Specify exactly one of --wandb-path or --checkpoint-dir")
    if self.viewer == "none" and not self.video and not self.export_only:
      raise ValueError("--viewer none requires --video or --export-only")
    if self.export_only and self.video:
      raise ValueError("--export-only cannot be combined with --video")
    if self.video:
      video_frame_count(self.video_duration, 1.0)
      if (
        self.video_width <= 0
        or self.video_height <= 0
        or self.video_width % 2
        or self.video_height % 2
      ):
        raise ValueError("Video width and height must be positive even integers")


def play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[INFO] Loading task configuration: {task_id}")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = True

  run = None
  if cfg.checkpoint_dir is not None:
    model_dir = cfg.checkpoint_dir.expanduser().resolve()
    for name in ("config.json", "actor.pt", "world_model.pt"):
      if not (model_dir / name).is_file():
        raise FileNotFoundError(f"Missing local checkpoint file: {model_dir / name}")
    print(f"[INFO] Loading local checkpoint: {model_dir}")
    run_config = json.loads((model_dir / "config.json").read_text())
    run_id = model_dir.name
    run_path = str(model_dir)
  else:
    import wandb

    print(f"[INFO] Connecting to WandB run: {cfg.wandb_path}")
    run = wandb.Api().run(cfg.wandb_path)
    run_config = run.config
    model_dir = download_model_dir(run)
    run_id = run.id
    run_path = cfg.wandb_path
  runner_cfg = load_runner_cfg(run_config)
  check_representation_matches(task_id, runner_cfg.representation)
  bundle = torch.load(
    model_dir / "world_model.pt",
    map_location="cpu" if cfg.export_only else device,
    weights_only=True,
  )
  if cfg.export_onnx or cfg.export_only:
    motion_cfg = env_cfg.commands["motion"]
    saved_motion = run_config.get("env_cfg", {}).get("commands", {}).get("motion", {})
    paths = export_policy_checkpoint(
      torch.load(model_dir / "actor.pt", map_location="cpu", weights_only=True),
      runner_cfg,
      bundle.get("robot_control"),
      task_id=task_id,
      body_names=list(saved_motion.get("body_names", motion_cfg.body_names)),
      anchor_body_name=saved_motion.get(
        "anchor_body_name", motion_cfg.anchor_body_name
      ),
      path=cfg.export_dir or model_dir / "exports",
      run_path=run_path,
      task=cfg.export_task,
    )
    for path in paths:
      print(f"[INFO] Exported: {path}")
    if cfg.export_only:
      return
  restore_control(env_cfg, bundle.get("robot_control"))

  motion_artifacts = (
    [a for a in run.used_artifacts() if a.type == "motions"] if run is not None else []
  )
  command_term = env_cfg.commands["motion"]
  assert isinstance(command_term, MotionCommandCfg)
  motion_names: list[str] = []
  if cfg.motion_file is not None:
    expanded = os.path.expanduser(cfg.motion_file)
    if any(c in expanded for c in "*?[]"):
      files = sorted(glob.glob(expanded))
    else:
      files = [expanded]
    if not files:
      raise ValueError(f"No motion files matched: {cfg.motion_file}")
    command_term.motion_file = files
    motion_names = [Path(f).stem for f in files]
    print(f"[INFO] Using {len(files)} local motion(s)")
  elif motion_artifacts:
    motion_files = []
    for artifact in motion_artifacts:
      print(f"[INFO] Found linked motion artifact: {artifact.name}")
      motion_files.append(str(download_artifact_cached(artifact) / "motion.npz"))
    command_term.motion_file = (
      motion_files if len(motion_files) > 1 else motion_files[0]
    )
    motion_names = [a.name.split(":")[0] for a in motion_artifacts]
    print(f"[INFO] Loaded {len(motion_files)} motion(s)")
  elif run is None:
    saved_motion = run_config.get("env_cfg", {}).get("commands", {}).get("motion", {})
    if saved_motion.get("motion_file"):
      command_term.motion_file = saved_motion["motion_file"]

  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  recorder = None
  try:
    wm = build_world_model(runner_cfg, env, device)
    representation = wm.representation
    dynamics = wm.dynamics
    wm.load_bundle(bundle)
    dynamics.eval()

    actor = build_actor(runner_cfg, wm, device)
    actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=device))
    actor.eval()

    wm_vis_model = create_wm_vis_model(env.unwrapped.sim.mj_model)
    if cfg.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved = "native" if has_display else "viser"
    else:
      resolved = cfg.viewer
    print(f"[INFO] Starting viewer: {resolved}")

    viewer_kwargs = dict(
      env=env,
      actor=actor,
      dynamics=dynamics,
      representation=representation,
      wm_vis_model=wm_vis_model,
      num_steps_rollout=cfg.num_steps_rollout,
      sim_context=wm.sim_context,
      task_specs=actor.task_specs,
      initial_task=cfg.initial_task,
    )
    if resolved == "native":
      viewer = SuperMimicNativeViewer(**viewer_kwargs)
    elif resolved == "none":
      viewer = SuperMimicHeadlessViewer(**viewer_kwargs)
    else:
      viewer = SuperMimicViserViewer(**viewer_kwargs, motion_names=motion_names)

    if cfg.video:
      timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
      path = cfg.video_dir / f"{task_id}_{run_id}_{timestamp}.mp4"
      recorder = PlaybackVideoRecorder(
        env.unwrapped,
        path,
        duration=cfg.video_duration,
        width=cfg.video_width,
        height=cfg.video_height,
      )
      viewer.video_recorder = recorder
      viewer.video_overlay = cfg.video_overlay
    viewer.motion_start = cfg.motion_start or ("start" if cfg.video else None)
    if viewer.motion_start is not None:
      viewer.reset_environment()
    viewer.run()
  except KeyboardInterrupt:
    print("\n[INFO] Playback interrupted. Saving recorded frames.")
  finally:
    try:
      if recorder is not None:
        recorder.close()
    finally:
      env.close()


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(PlayConfig, args=remaining_args, config=PLAY_FLAGS)
  play(chosen_task, args)


if __name__ == "__main__":
  main()
