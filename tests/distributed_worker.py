"""Run with uv run torchrun --standalone --nproc-per-node=2 tests/distributed_worker.py."""

from copy import deepcopy
from unittest.mock import patch

import torch
from tensordict import TensorDict

from supermimic.learners.learner_utils import GradientStep, build_optimizers
from supermimic.models import _CudaGraphNormalization
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.utils import distributed as d


def check_steps(device, execution):
  torch.manual_seed(123)
  model = torch.nn.Sequential(
    torch.nn.Linear(5, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 2),
  ).to(device)
  # A genuinely unused parameter must not receive an optimizer update.
  model.register_parameter("unused", torch.nn.Parameter(torch.ones(3, device=device)))
  reference = deepcopy(model)
  optimizers = build_optimizers([model], 0.0002, device=device)
  reference_opts = build_optimizers([reference], 0.0002, device=device)

  def loss_from(td):
    loss = (model(td["x"]) - td["y"]).square().mean()
    return loss, {"loss": loss.detach()}

  step = GradientStep(loss_from, [model], optimizers, execution, clip_value=0.1)
  for _ in range(8):
    x = torch.randn(16 * d.world_size(), 5, device=device)
    y = torch.randn(16 * d.world_size(), 2, device=device)
    # Reference sees exactly the union of rank batches.
    for opt in reference_opts:
      opt.zero_grad(set_to_none=True)
    loss = (reference(x) - y).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    torch.nn.utils.clip_grad_value_(reference.parameters(), 0.1)
    if execution == "eager":
      # Check the raw global-batch gradient independently of Muon's BF16
      # Newton-Schulz orthogonalization, which amplifies rounding differences.
      for opt in optimizers:
        opt.zero_grad(set_to_none=True)
      lo, hi = 16 * d.rank(), 16 * (d.rank() + 1)
      ((model(x[lo:hi]) - y[lo:hi]).square().mean()).backward()
      step.reducer()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
      if execution == "eager":
        for actual, expected in zip(
          model.parameters(), reference.parameters(), strict=True
        ):
          if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-5, rtol=2e-4)
    for opt in reference_opts:
      opt.step()
    lo, hi = 16 * d.rank(), 16 * (d.rank() + 1)
    step(TensorDict({"x": x[lo:hi], "y": y[lo:hi]}, batch_size=[16]))
    for actual, expected in zip(
      model.parameters(), reference.parameters(), strict=True
    ):
      torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    assert model.unused.grad is None
    d.assert_synced(model.parameters())
  if d.rank() == 0:
    print(
      f"PASS: {execution} Muon/Adam updates match global-batch reference through graph capture/replay",
      flush=True,
    )


def check_statistics(device):
  norm = _CudaGraphNormalization(3).to(device)
  pointers = [b.data_ptr() for b in norm.buffers()]
  torch.manual_seed(82)
  xs = [torch.randn(5 + r, 3, device=device) + r for r in range(d.world_size())]
  norm.update(xs[d.rank()])
  union = torch.cat(xs)
  torch.testing.assert_close(norm.mean, union.mean(0))
  torch.testing.assert_close(norm.std, union.std(0, unbiased=False))
  assert pointers == [b.data_ptr() for b in norm.buffers()]
  d.assert_synced(norm.buffers())
  cfg = ReturnSamplerCfg(horizon=2)
  sampler = ReturnSampler(cfg, 4, 40, 2, 1.0, device)
  # Capture global observations for an independent single-process reference.
  frames = torch.arange(2 * d.world_size(), device=device) * 4
  rewards = torch.arange(2 * d.world_size(), device=device).float() / 10
  for t in range(2):
    idx = slice(2 * d.rank(), 2 * (d.rank() + 1))
    zeros = torch.zeros(2, device=device, dtype=torch.bool)
    with torch.inference_mode():
      weights = sampler.record(frames[idx] + t, rewards[idx], zeros, zeros, zeros)
  d.assert_synced([sampler.value, sampler.count, sampler.effort, weights])
  # Every env visits one bin in this fixture; count is globally accumulated.
  assert sampler.count.sum().item() == 2 * d.world_size()
  reference = ReturnSampler(cfg, 4, 40, 2 * d.world_size(), 1.0, device)
  zeros = torch.zeros(2 * d.world_size(), device=device, dtype=torch.bool)
  with patch("supermimic.runners.reset_sampling.sum_", lambda x: x):
    for t in range(2):
      expected_weights = reference.record(frames + t, rewards, zeros, zeros, zeros)
  for name in ("value", "count", "effort"):
    torch.testing.assert_close(getattr(sampler, name), getattr(reference, name))
  torch.testing.assert_close(weights, expected_weights)
  if d.rank() == 0:
    print(
      "PASS: global normalizer moments/storage and return sampler statistics",
      flush=True,
    )


def check_local_missing_gradient(device):
  params = [torch.nn.Parameter(torch.ones(2, device=device)) for _ in range(2)]
  params[0].grad = torch.full_like(params[0], d.rank() + 1.0)
  if d.rank() == 0:
    params[1].grad = torch.full_like(params[1], 4.0)
  d.GradientReducer(params)()
  torch.testing.assert_close(
    params[0].grad, torch.full_like(params[0], (d.world_size() + 1) / 2)
  )
  torch.testing.assert_close(
    params[1].grad, torch.full_like(params[1], 4 / d.world_size())
  )


def check_partial_graph(device):
  class Branch(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.a = torch.nn.Parameter(torch.tensor([1.0], device=device))
      self.b = torch.nn.Parameter(torch.tensor([2.0], device=device))

  model, reference = Branch(), Branch()
  opt = torch.optim.SGD(model.parameters(), lr=0.05)
  ref_opt = torch.optim.SGD(reference.parameters(), lr=0.05)

  def loss_from(td):
    pred = model.a + model.b if d.rank() == 0 else model.a
    loss = (pred * td["x"]).square().mean() / 2
    return loss, {"loss": loss.detach()}

  step = GradientStep(loss_from, [model], [opt], "split")
  for _ in range(7):
    ref_opt.zero_grad(set_to_none=True)
    loss = (
      (reference.a + reference.b).square() + (d.world_size() - 1) * reference.a.square()
    ).mean() / (2 * d.world_size())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    ref_opt.step()
    step(TensorDict({"x": torch.ones(4, 1, device=device)}, batch_size=[4]))
    for actual, expected in zip(
      model.parameters(), reference.parameters(), strict=True
    ):
      torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    d.assert_synced(model.parameters())
  assert not d.all_ready(d.rank() != 0, device)
  d.seed_all(42 + 1000 * d.rank())
  torch.rand(5 + d.rank())
  before = torch.random.get_rng_state().clone()
  with d.model_initialization(91):
    initial = torch.nn.Linear(4, 4).to(device)
  assert torch.equal(before, torch.random.get_rng_state())
  d.assert_synced(initial.parameters())
  d.seed_all(42 + 1000 * d.rank())
  sample = torch.rand(4, device=device)
  if d.world_size() > 1:
    gathered = [torch.empty_like(sample) for _ in range(d.world_size())]
    torch.distributed.all_gather(gathered, sample)
    assert all(not torch.equal(gathered[0], x) for x in gathered[1:])
  if d.rank() == 0:
    print(
      "PASS: partial gradients across graph replays, collective readiness and independent RNG",
      flush=True,
    )


if __name__ == "__main__":
  device = d.initialize(None)
  try:
    check_steps(device, "eager")
    check_steps(device, "split")
    check_statistics(device)
    check_partial_graph(device)
    if d.world_size() > 1:
      check_local_missing_gradient(device)
    if d.rank() == 0:
      print(f"ALL CHECKS PASSED ({d.world_size()} ranks)", flush=True)
  finally:
    d.shutdown()
