"""GenCoord EVT observation contract used by new motion policies."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_inv, quat_mul

from supermimic.utils.helpers import quat_to_rot6d

HISTORY = 5
FUTURE = 5
LAYOUT = "evt_causal_v2"
GRAVITY_LAYOUT = "evt_causal_v2_gravity"


def advance_history(history, frame, reset=None):
  """Append a control frame. Newly reset rows repeat their first observation."""
  first = frame.unsqueeze(1).expand(-1, HISTORY, -1)
  if history is None:
    return first
  shifted = torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)
  if reset is None:
    return shifted
  return torch.where(reset.reshape(-1, 1, 1).bool(), first, shifted)


def motion_frame(motion, times, root_quat):
  """Reference at each supplied timestamp, in that timestamp's robot frame."""
  relative = quat_mul(quat_inv(root_quat), motion.root_quat_w[times])
  return torch.cat(
    (motion.joint_pos[times], motion.joint_vel[times], quat_to_rot6d(relative)), -1
  )


def privileged_reference(motion, times, aux):
  """Current relative translation and five consecutive future frames in root(t)."""
  end = (
    motion.motion_ends[
      torch.searchsorted(motion.motion_ends, times.contiguous(), right=True)
    ]
    - 1
  )
  indices = torch.minimum(
    times[:, None] + torch.arange(1, FUTURE + 1, device=times.device), end[:, None]
  )
  pos, quat = aux[:, :3], aux[:, 3:7]
  current_offset = quat_apply_inverse(quat, motion.root_pos_w[times] - pos)
  frames = quat[:, None, :].expand(-1, FUTURE, -1)
  offset = quat_apply_inverse(frames, motion.root_pos_w[indices] - pos[:, None, :])
  rot = quat_to_rot6d(quat_mul(quat_inv(frames), motion.root_quat_w[indices]))
  future = torch.cat(
    (motion.joint_pos[indices], motion.joint_vel[indices], offset, rot), -1
  )
  return current_offset, future.flatten(1)


def goal_contract():
  return {
    "order": "frame_major_oldest_to_newest: joint_pos, joint_vel, relative_anchor_rot6d",
    "frame_offsets": list(range(1 - HISTORY, 1)),
    "orientation": "reference relative to robot pelvis at the same timestamp; first two matrix columns, row-major",
    "joint_positions": "absolute radians in robot_control.joint_names order",
    "reset_padding": "repeat first available frame within the episode and clip",
  }
