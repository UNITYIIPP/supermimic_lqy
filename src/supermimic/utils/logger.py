from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from supermimic.utils import distributed as distributed


class Logger:
  """Shared training metrics for W&B, local TensorBoard, and the console."""

  def __init__(
    self,
    run_name: str | None,
    num_envs: int,
    max_episode_length: int,
    device: str,
    wandb_enabled: bool = False,
  ) -> None:
    self.run_name = run_name
    self.num_envs = num_envs
    self.max_episode_length = max_episode_length
    self.device = device
    self.wandb_enabled = wandb_enabled
    self._writer: SummaryWriter | None = None

    self.tot_timesteps = 0
    self.tot_time = 0.0

    # Create buffers
    self.ep_extras = []
    self.rewbuffer = deque(maxlen=100)
    self.lenbuffer = deque(maxlen=100)
    self.cur_reward_sum = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )
    self.cur_episode_length = torch.zeros(
      self.num_envs, dtype=torch.float, device=self.device
    )
    self._episode_counts = torch.zeros(4, dtype=torch.long, device=self.device)

  def init_tensorboard(self, log_dir: Path) -> None:
    """Write events alongside this run's checkpoints, flushing every 30 seconds."""
    self.close()
    self._writer = SummaryWriter(log_dir=str(log_dir), flush_secs=30)

  def close(self) -> None:
    """Flush pending metrics and release the writer; safe to call repeatedly."""
    writer, self._writer = self._writer, None
    if writer is not None:
      try:
        writer.flush()
      finally:
        writer.close()

  def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
    *,
    failures: torch.Tensor | None = None,
    motion_end: torch.Tensor | None = None,
    time_out: torch.Tensor | None = None,
  ) -> None:
    """Add metrics from the environment step to the buffers."""
    if "episode" in extras:
      self.ep_extras.append(extras["episode"])
    elif "log" in extras:
      self.ep_extras.append(extras["log"])

    self.cur_reward_sum += rewards
    self.cur_episode_length += 1

    flush_mask = dones.bool() | (self.cur_episode_length >= self.max_episode_length)
    self._episode_counts[0] += flush_mask.sum()
    self._episode_counts[1] += (dones if failures is None else failures).bool().sum()
    if motion_end is not None:
      self._episode_counts[2] += motion_end.bool().sum()
    if time_out is not None:
      self._episode_counts[3] += time_out.bool().sum()
    new_ids = flush_mask.nonzero(as_tuple=False)
    if len(new_ids) > 0:
      self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
      self.lenbuffer.extend(
        self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
      )
      self.cur_reward_sum[new_ids] = 0
      self.cur_episode_length[new_ids] = 0

  def _episode_extras(self) -> dict[str, float]:
    """Mean of each episode extra over the episodes since the last log."""
    pairs = {}
    all_keys = dict.fromkeys(k for extra in self.ep_extras for k in extra)
    for key in all_keys:
      infotensor = torch.tensor([], device=self.device)
      for ep_info in self.ep_extras:
        if key not in ep_info:
          continue
        val = ep_info[key]
        if not isinstance(val, torch.Tensor):
          val = torch.tensor([val], device=self.device)
        else:
          val = val.to(self.device).reshape(-1)

        if val.ndim == 0:
          val = val.unsqueeze(0)
        infotensor = torch.cat((infotensor, val))

      if len(infotensor) > 0:
        pairs[key] = (infotensor.sum().item(), infotensor.numel())
    return distributed.mean_metrics(pairs, self.device)

  def _build_console_log(
    self,
    it: int,
    total_it: int,
    fps: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    losses: dict[str, float],
    extras: dict[str, float],
    iteration_time: float,
    width: int,
    pad: int,
    counters: dict[str, int] | None = None,
  ) -> str:
    """Build the formatted console log string."""
    log_string = f"""{"#" * width}\n"""
    log_string += (
      f"""\033[1m{f" Learning iteration {it}/{total_it} ".center(width)}\033[0m \n\n"""
    )

    log_string += f"""{"Run name:":>{pad}} {self.run_name}\n"""
    for name, value in (counters or {}).items():
      log_string += f"{name + ':':>{pad}} {value}\n"

    log_string += (
      f"""{"Total steps:":>{pad}} {self.tot_timesteps} \n"""
      f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
      f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
      f"""{"WM Learning time:":>{pad}} {wm_learn_time:.3f}s \n"""
      f"""{"Policy Learning time:":>{pad}} {policy_learn_time:.3f}s \n"""
    )

    for key, value in losses.items():
      log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

    for key, value in extras.items():
      label = f"{key}:" if "/" in key else f"Episode {key}:"
      log_string += f"""{label:>{pad}} {value:.4f}\n"""

    done_it = max(it, 1)
    remaining_it = max(total_it - it, 0)
    eta = self.tot_time / done_it * remaining_it
    log_string += (
      f"""{"-" * width}\n"""
      f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
      f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
      f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
    )

    return log_string

  def log(
    self,
    it: int,
    total_it: int,
    collect_time: float,
    wm_learn_time: float,
    policy_learn_time: float,
    loss_dict: Mapping[str, float | torch.Tensor],
    num_collected_steps: int,
    width: int = 80,
    pad: int = 40,
    env_iterations: int | None = None,
    learning_updates: int | None = None,
  ) -> None:
    times = distributed.max_values(
      {"collect": collect_time, "wm": wm_learn_time, "policy": policy_learn_time},
      self.device,
    )
    collect_time, wm_learn_time, policy_learn_time = (
      times["collect"],
      times["wm"],
      times["policy"],
    )
    steps = torch.tensor(num_collected_steps, device=self.device, dtype=torch.long)
    distributed.sum_(steps)
    num_collected_steps = int(steps.item())
    iteration_time = collect_time + wm_learn_time + policy_learn_time
    self.tot_timesteps += num_collected_steps
    self.tot_time += iteration_time
    fps = int(num_collected_steps / (iteration_time + 1e-6))

    extras = self._episode_extras()
    # Device stats reach the host in one transfer.
    on_device = {k: v for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}
    values = torch.stack(list(on_device.values())).tolist() if on_device else []
    losses = {**loss_dict, **dict(zip(on_device, values, strict=True))}
    losses = distributed.mean_metrics(
      {k: (float(v), 1) for k, v in losses.items()}, self.device
    )
    episode = distributed.mean_metrics(
      {
        "Train/mean_reward": (sum(self.rewbuffer), len(self.rewbuffer)),
        "Train/mean_episode_length": (sum(self.lenbuffer), len(self.lenbuffer)),
      },
      self.device,
    )
    counts = self._episode_counts.clone()
    distributed.sum_(counts)
    completed, failed, motion_ends, time_outs = counts.tolist()
    episode.update(
      {
        "Train/completed_episodes": completed,
        "Train/failed_episodes": failed,
        "Train/motion_end_count": motion_ends,
        "Train/time_out_count": time_outs,
        "Train/failure_rate": failed / completed if completed else 0.0,
      }
    )
    self._episode_counts.zero_()
    self.ep_extras.clear()
    if distributed.rank() != 0:
      return
    row = (
      losses
      | episode
      | {
        "Perf/total_fps": fps,
        "Perf/collection_time": collect_time,
        "Perf/wm_learning_time": wm_learn_time,
        "Perf/policy_learning_time": policy_learn_time,
      }
    )
    counters = {"GPU count": distributed.world_size()}
    if env_iterations is not None:
      counters["Environment iterations"] = env_iterations
      row["Counters/env_iterations"] = env_iterations
    if learning_updates is not None:
      counters["Learning updates"] = learning_updates
      row["Counters/learning_updates"] = learning_updates
    row["Counters/global_env_steps"] = self.tot_timesteps
    row |= {k if "/" in k else f"Episode/{k}": v for k, v in extras.items()}
    if self._writer is not None:
      for name, value in row.items():
        # Preserve Python-float precision; TB 2.20 needs --load_fast=false.
        self._writer.add_scalar(
          name, value, global_step=it, new_style=True, double_precision=True
        )
    if self.wandb_enabled:
      wandb.log(row, step=it)
    print(
      self._build_console_log(
        it,
        total_it,
        fps,
        collect_time,
        wm_learn_time,
        policy_learn_time,
        losses,
        extras | episode,
        iteration_time,
        width,
        pad,
        counters,
      )
    )
    self.ep_extras.clear()
