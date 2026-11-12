"""Stream simulation frames to MP4 with an independent tracking camera."""

from __future__ import annotations

import copy
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import mediapy as media
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import OffscreenRenderer, ViewerConfig
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer


def video_frame_count(duration: float, step_dt: float) -> int:
  if not math.isfinite(duration) or duration <= 0:
    raise ValueError("Video duration must be finite and positive")
  if not math.isfinite(step_dt) or step_dt <= 0:
    raise ValueError("Environment step_dt must be finite and positive")
  return max(1, math.ceil(duration / step_dt - 1e-9))


class PlaybackVideoRecorder:
  """Render once per control step and encode immediately, without a frame buffer."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    path: Path,
    duration: float = 30.0,
    width: int = 1280,
    height: int = 720,
  ) -> None:
    self.max_frames = video_frame_count(duration, env.step_dt)
    if width <= 0 or height <= 0 or width % 2 or height % 2:
      raise ValueError("Video width and height must be positive even integers")
    self.env = env
    self.path = path
    self.fps = 1.0 / env.step_dt
    self.frame_count = 0
    self._closed = False
    self._resources = ExitStack()
    self._renderer: OffscreenRenderer | None = None
    self._writer: media.VideoWriter | None = None
    self._camera = copy.deepcopy(env.cfg.viewer)
    self._camera.width = width
    self._camera.height = height
    self._camera.env_idx = 0
    self._camera.max_extra_envs = 0
    if self._camera.body_name:
      self._camera.origin_type = ViewerConfig.OriginType.ASSET_BODY
    else:
      self._camera.origin_type = ViewerConfig.OriginType.ASSET_ROOT
    if not media.video_is_available():
      media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

  @property
  def complete(self) -> bool:
    return self.frame_count >= self.max_frames

  def capture(
    self, draw_overlays: Callable[[MujocoNativeDebugVisualizer], None]
  ) -> None:
    if self._closed or self.complete:
      return
    if self._renderer is None:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      sim = self.env.sim
      self._renderer = OffscreenRenderer(
        sim.mj_model,
        self._camera,
        self.env.scene,
        sim_model=sim.model,
        expanded_fields=sim.expanded_fields,
      )
      self._resources.callback(self._renderer.close)
      self._renderer.initialize()
      self._writer = self._resources.enter_context(
        media.VideoWriter(
          self.path,
          shape=(self._camera.height, self._camera.width),
          fps=self.fps,
          codec="h264",
          encoded_format="yuv420p",
          crf=23,
          ffmpeg_args=["-movflags", "+faststart", "-threads", "2"],
        )
      )
      print(f"[INFO] Recording {self.fps:g} FPS video to {self.path}")
    self._renderer.update(self.env.sim.data, debug_vis_callback=draw_overlays)
    assert self._writer is not None
    self._writer.add_image(self._renderer.render())
    self.frame_count += 1
    if self.complete:
      self.close()

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    self._resources.close()
    if self.frame_count:
      print(
        f"[INFO] Saved video: {self.path} "
        f"({self.frame_count} frames, {self.frame_count / self.fps:g} seconds)"
      )
