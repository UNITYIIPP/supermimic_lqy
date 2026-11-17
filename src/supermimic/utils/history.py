"""Temporal windows shared by collection, imagination and deployment playback."""

from __future__ import annotations

import torch


def append_history(
  history: torch.Tensor | None,
  frame: torch.Tensor,
  length: int,
  reset: torch.Tensor | None = None,
) -> torch.Tensor:
  """Append once per control step, oldest first; reset rows repeat this frame.

  This function does not mutate its inputs, preserving rollout autograd history.
  Reading a returned window never advances it.
  """
  if length < 1:
    raise ValueError("History length must be positive")
  current = frame.unsqueeze(1)
  if history is None:
    return current.expand(-1, length, -1).clone()
  result = torch.cat([history[:, 1:], current], dim=1)
  if reset is not None:
    result = torch.where(reset.bool().reshape(-1, 1, 1), current, result)
  return result
