"""Single-node synchronous data parallel utilities (one process per GPU)."""

from __future__ import annotations

import os
import random
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


def world_size() -> int:
  return dist.get_world_size() if dist.is_initialized() else 1


def rank() -> int:
  return dist.get_rank() if dist.is_initialized() else 0


def resolve_batch_layout(
  num_envs: int,
  wm_batch: int,
  policy_batch: int,
  scope: str,
  workers: int,
) -> dict:
  """Resolve explicit batch semantics without silently rounding global sizes."""
  requested = dict(num_envs=num_envs, wm_batch=wm_batch, policy_batch=policy_batch)
  if workers < 1 or any(value < 1 for value in requested.values()):
    raise ValueError("GPU count, environment count and batch sizes must be positive")
  if scope not in {"per_rank", "global"}:
    raise ValueError(f"Unknown batch size scope: {scope}")
  divisor = workers if scope == "global" else 1
  if any(value % divisor for value in requested.values()):
    raise ValueError(f"Global sizes {requested} must be divisible by {workers} GPUs")
  local = {key: value // divisor for key, value in requested.items()}
  return dict(
    scope=scope,
    requested=requested,
    local=local,
    global_sizes={key: value * workers for key, value in local.items()},
  )


def initialize(device: str | None) -> str:
  if int(os.environ.get("WORLD_SIZE", "1")) == 1:
    resolved = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.device(resolved).type == "cuda":
      torch.cuda.set_device(resolved)
      os.environ["MUJOCO_EGL_DEVICE_ID"] = str(torch.cuda.current_device())
    return resolved
  local = int(os.environ["LOCAL_RANK"])
  expected = f"cuda:{local}"
  if device is not None and device != expected:
    raise ValueError(f"torchrun assigns {expected}; omit --device")
  torch.cuda.set_device(local)
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local)
  dist.init_process_group("nccl", timeout=timedelta(minutes=15))
  return expected


def shutdown() -> None:
  # No barrier here: another worker may have failed.
  if dist.is_initialized():
    dist.destroy_process_group()


def seed_all(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed % (2**32))
  torch.manual_seed(seed)


@contextmanager
def model_initialization(seed: int | None):
  """Networks are constructed on CPU before .to(device); isolate their RNG.

  Environment resets consume a batch-size-dependent number of draws. Keeping
  this stream separate gives the same initial weights at different GPU counts
  without resetting each rank's simulation/exploration stream.
  """
  if seed is None:
    yield
    return
  with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(seed)
    yield


def broadcast_object(value):
  if world_size() == 1:
    return value
  objects = [value]
  dist.broadcast_object_list(objects, src=0)
  return objects[0]


def check_same(value, label: str) -> None:
  """Fail before training if ranks disagree about shapes or collective order."""
  if world_size() == 1:
    return
  values = [None] * world_size()
  dist.all_gather_object(values, value)
  if any(v != values[0] for v in values):
    raise ValueError(f"Ranks disagree about {label}: {values}")


@torch.no_grad()
def broadcast_modules(modules: Iterable[nn.Module]) -> None:
  if world_size() == 1:
    return
  for module in modules:
    for tensor in [*module.parameters(), *module.buffers()]:
      dist.broadcast(tensor, src=0)


def sum_(tensor: torch.Tensor) -> torch.Tensor:
  if world_size() > 1:
    dist.all_reduce(tensor)
  return tensor


def all_ready(ready: bool, device: str) -> bool:
  if world_size() == 1:
    return ready
  flag = torch.tensor(int(ready), device=device)
  dist.all_reduce(flag, op=dist.ReduceOp.MIN)
  return bool(flag.item())


class GradientReducer:
  """Average equal-size local batches before clipping, using a reusable bucket.

  The presence mask is reduced too: locally unused parameters get zero local
  contributions; globally unused parameters retain grad=None (no optimizer
  momentum/weight-decay update). This also checks non-finite gradients globally.
  Communication is deliberately outside CUDA graph capture.
  """

  def __init__(self, params: Iterable[torch.Tensor]):
    self.params = list(params)
    self.bucket = self.params[0].new_empty(0)
    self.presence = torch.zeros(len(self.params), device=self.params[0].device)
    self._active: list[torch.Tensor] = []
    self._views: list[torch.Tensor] = []
    self._local_presence: tuple[bool, ...] | None = None
    self._global_presence: list[float] = []
    self._synthesized: dict[int, torch.Tensor] = {}

  @torch.no_grad()
  def __call__(self) -> None:
    if world_size() == 1:
      return
    if torch.cuda.is_current_stream_capturing():
      raise RuntimeError("Gradient synchronization must run outside CUDA graphs")
    presence = tuple(p.grad is not None for p in self.params)
    if self._local_presence is None:
      self.presence.copy_(torch.tensor(presence, device=self.bucket.device))
      sum_(self.presence)
      self._global_presence = self.presence.tolist()
      self._local_presence = presence
      self._active = [
        p
        for p, active in zip(self.params, self._global_presence, strict=True)
        if active
      ]
      self.bucket = self.params[0].new_empty(sum(p.numel() for p in self._active))
      self._views = list(self.bucket.split([p.numel() for p in self._active]))
    elif presence != self._local_presence:
      raise RuntimeError(
        "Gradient participation changed; split graphs require fixed active tasks"
      )
    for p, view in zip(self._active, self._views, strict=True):
      if p.grad is None:
        view.zero_()
      else:
        view.copy_(p.grad.reshape(-1))
    if self.bucket.numel():
      sum_(self.bucket)
    self.bucket.div_(world_size())
    torch._assert_async(
      torch.isfinite(self.bucket).all(), "Non-finite globally reduced gradient"
    )
    for p, view in zip(self._active, self._views, strict=True):
      if p.grad is None:
        if id(p) not in self._synthesized:
          self._synthesized[id(p)] = torch.zeros_like(p)
        p.grad = self._synthesized[id(p)]
      p.grad.copy_(view.view_as(p))


@torch.no_grad()
def assert_synced(tensors: Iterable[torch.Tensor], atol: float = 1e-6) -> None:
  """Diagnostic check, called by every rank at the same boundary."""
  if world_size() == 1:
    return
  error = torch.zeros((), device=torch.cuda.current_device())
  for tensor in tensors:
    reference = tensor.detach().clone()
    dist.broadcast(reference, src=0)
    if tensor.numel():
      error.copy_(
        torch.maximum(error, (tensor.double() - reference.double()).abs().max().float())
      )
  dist.all_reduce(error, op=dist.ReduceOp.MAX)
  if not torch.isfinite(error) or error.item() > atol:
    raise RuntimeError(f"Distributed state diverged: max error={error.item()}")


def mean_metrics(
  pairs: dict[str, tuple[float, float]], device: str
) -> dict[str, float]:
  """Merge sums/counts, including metrics absent on some ranks."""
  keys = sorted(pairs)
  if world_size() > 1:
    all_keys: list[Any] = [None] * world_size()
    dist.all_gather_object(all_keys, keys)
    keys = sorted({k for rank_keys in all_keys for k in rank_keys})
  if not keys:
    return {}
  values = torch.tensor(
    [pairs.get(k, (0, 0)) for k in keys], device=device, dtype=torch.float64
  )
  sum_(values)
  return {
    k: total / count
    for k, (total, count) in zip(keys, values.tolist(), strict=True)
    if count
  }


def max_values(values: dict[str, float], device: str) -> dict[str, float]:
  keys = sorted(values)
  tensor = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float64)
  if world_size() > 1:
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
  return dict(zip(keys, tensor.tolist(), strict=True))
