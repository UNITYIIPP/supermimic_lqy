"""Shared learner utilities."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.utils.distributed import GradientReducer, world_size

Execution = Literal["auto", "eager", "split", "full"]


def resolve_execution(mode: Execution, profile: bool = False) -> Execution:
  if mode == "auto":
    return "split" if world_size() > 1 or profile else "full"
  if mode == "full" and (world_size() > 1 or profile):
    raise ValueError("full graphs require one GPU and profile=False; use split")
  return mode


class Timings:
  def __init__(self, enabled: bool, device: str):
    self.enabled, self.device = enabled, torch.device(device)
    self.values: dict[str, float] = {}

  @contextmanager
  def measure(self, name: str):
    if not self.enabled:
      yield
      return
    if self.device.type == "cuda":
      torch.cuda.synchronize(self.device)
    start = time.perf_counter()
    yield
    if self.device.type == "cuda":
      torch.cuda.synchronize(self.device)
    self.values[name] = self.values.get(name, 0.0) + time.perf_counter() - start

  def drain(self) -> dict[str, float]:
    values, self.values = self.values, {}
    return values


def build_optimizers(
  modules: Sequence[nn.Module],
  lr: float,
  betas: tuple[float, float] = (0.9, 0.999),
  device: str = "cpu",
) -> list[torch.optim.Optimizer]:
  """Muon on the Linear weights between the first and the last Linear of each
  ``nn.Sequential`` in ``modules``, fused capturable Adam on the rest, both at
  ``lr``. Muon's ``lr`` is a float, fixed once an update is captured."""
  hidden = [
    linear.weight
    for module in modules
    for seq in module.modules()
    if isinstance(seq, nn.Sequential)
    for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
  ]
  hidden_ids = {id(p) for p in hidden}
  rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
  adam = torch.optim.Adam(
    rest, lr=torch.tensor(lr, device=device), betas=betas, fused=True, capturable=True
  )
  if not hidden:
    return [adam]
  return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]


def _optimize_step(
  optimizers: Sequence[torch.optim.Optimizer],
  loss: torch.Tensor,
  params: Iterable[torch.Tensor] | None = None,
  max_norm: float | None = None,
  clip_value: float = 0.0,
) -> torch.Tensor | None:
  """Run one optimizer step: zero_grad -> backward -> optional clip -> step."""
  for optimizer in optimizers:
    optimizer.zero_grad(set_to_none=True)
  loss.backward()
  grad_norm = None
  if params is not None and max_norm is not None:
    # Don't set error_if_nonfinite=True: it forces a per-step CPU<->GPU sync.
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    if clip_value > 0:
      torch.nn.utils.clip_grad_value_(params, clip_value)
  for optimizer in optimizers:
    optimizer.step()
  return grad_norm


def graph_state(
  modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]
) -> list[torch.Tensor]:
  """What a captured update reads that Python can rebind: parameters, buffers,
  optimizer state and learning-rate tensors."""
  state: list[torch.Tensor] = []
  for module in modules:
    state += [*module.parameters(), *module.buffers()]
  for optimizer in optimizers:
    state += [t for s in optimizer.state.values() for t in s.values()]
    state += [
      group["lr"] for group in optimizer.param_groups if torch.is_tensor(group["lr"])
    ]
  return state


def _identity(state: list[torch.Tensor]) -> list[int]:
  return [x.data_ptr() for x in state]


@functools.cache
def _capture_stream_and_pool(device: torch.device):
  return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
  """Runs ``fn`` eagerly for its first ``warmup`` calls, then records it once as
  a CUDA graph and replays it on the device of ``state()``. ``fn`` must be
  capture-safe: no host syncs, fixed shapes, and every tensor it reads besides
  its inputs updated in place. Python values it reads are fixed at capture, and
  each replay overwrites the outputs it returns. By default, CapturedUpdates on
  a device share one pool and must replay in their recording order; private_pool
  gives each update an independent pool. A replay raises if a tensor in
  ``state()`` moved; a learning rate
  that changes is a device tensor, changed in place with ``group["lr"].fill_(lr)``."""

  def __init__(
    self,
    fn: Callable[[TensorDict], dict[str, torch.Tensor]],
    state: Callable[[], list[torch.Tensor]],
    warmup: int = 3,
    private_pool: bool = False,
  ):
    self._fn = fn
    self._state = state
    self._warmup = warmup
    self._inputs: TensorDict | None = None
    self._graph: torch.cuda.CUDAGraph | None = None
    self._identity: list[int] = []
    self._outputs: dict[str, torch.Tensor] = {}
    # Split graphs can replay in arbitrary minibatch order. Their outputs may
    # remain live while a different learner executes, so they cannot share pools.
    self._private_pool = private_pool
    self._pool = None

  def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
    state = self._state()
    if not state[0].is_cuda:
      return self._fn(inputs)
    with torch.cuda.device(state[0].device):
      stream, pool = _capture_stream_and_pool(state[0].device)
      if self._private_pool:
        if self._pool is None:
          self._pool = torch.cuda.graph_pool_handle()
        pool = self._pool
      if self._inputs is None:
        self._inputs = inputs.clone()
      else:
        self._inputs.update_(inputs)
      if self._warmup > 0:
        self._warmup -= 1
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
          outputs = self._fn(self._inputs)
        torch.cuda.current_stream().wait_stream(stream)
        return outputs
      if self._graph is None:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
          self._outputs = self._fn(self._inputs)
        self._graph, self._identity = graph, _identity(self._state())
      elif _identity(state) != self._identity:
        raise RuntimeError("a captured tensor moved")
      self._graph.replay()
      if not self._private_pool:
        torch.cuda.current_stream().synchronize()
      return self._outputs


class GradientStep:
  """Forward/backward graph -> external reduction -> clip/optimizer graph.

  A separate instance per optimizer preserves graph gradient storage. Optimizer
  steps use a separate graph (fused Adam / Muon); no NCCL operations are captured.
  """

  def __init__(
    self,
    loss_fn: Callable[[TensorDict], tuple[torch.Tensor, dict[str, torch.Tensor]]],
    modules: Sequence[nn.Module],
    optimizers: Sequence[torch.optim.Optimizer],
    execution: str,
    clip_value: float = 0.0,
    profile: bool = False,
  ):
    self.loss_fn, self.modules = loss_fn, modules
    self.optimizers = optimizers
    self.params = [
      p for opt in optimizers for g in opt.param_groups for p in g["params"]
    ]
    self.reducer = GradientReducer(self.params)
    self.clip_value = clip_value
    self.execution = execution
    self.timings = Timings(profile, str(self.params[0].device))
    self.captured = CapturedUpdate(
      self._backward, lambda: graph_state(self.modules, []), private_pool=True
    )
    self._empty = TensorDict({}, batch_size=[])
    self._captured_optimizer = CapturedUpdate(
      self._finish,
      lambda: [
        *self.params,
        *[p.grad for p in self.params if p.grad is not None],
        *graph_state([], self.optimizers),
      ],
      private_pool=True,
    )

  def _backward(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
    for opt in self.optimizers:
      opt.zero_grad(set_to_none=True)
    loss, outputs = self.loss_fn(inputs)
    loss.backward()
    return {k: v.detach() for k, v in outputs.items()}

  def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
    outputs = (
      self.captured(inputs) if self.execution == "split" else self._backward(inputs)
    )
    # A captured backward has no Python zero_grad on replay. Clear gradients
    # synthesized by the reducer for parameters unused on this rank.
    if self.execution == "split" and self.reducer._local_presence is not None:
      for p, local in zip(self.params, self.reducer._local_presence, strict=True):
        if not local:
          p.grad = None
    with self.timings.measure("grad_sync"):
      self.reducer()
    result = (
      self._captured_optimizer(self._empty)
      if self.execution == "split"
      else self._finish(self._empty)
    )
    return {**outputs, **result}

  def _finish(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
    del inputs
    norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
    if self.clip_value > 0:
      torch.nn.utils.clip_grad_value_(self.params, self.clip_value)
    for opt in self.optimizers:
      opt.step()
    return {"grad_norm": norm.detach()}
