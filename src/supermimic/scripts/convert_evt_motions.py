"""Convert EVT compact motions to one research-MjLab NPZ per clip."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import tyro

from supermimic.robots.evt import EVT_CFG


def convert_file(source: Path, output_dir: Path, overwrite: bool = False) -> list[Path]:
  """Convert single clips or compact_v1 archives without merging boundaries."""
  with np.load(source, allow_pickle=False) as data:
    missing = {"root_pos", "rest", "fps", "joint_names"} - set(data.files)
    if missing:
      raise ValueError(f"{source}: missing fields {sorted(missing)}")
    if "robot" in data and data["robot"].item() not in {"evt", "tiangong_evt2"}:
      raise ValueError(f"{source}: incompatible robot tag")
    if "storage" in data and data["storage"].item() != "compact_v1":
      raise ValueError(f"{source}: expected compact_v1 storage")
    for key, expected in (
      ("root_body", "pelvis"),
      ("quaternion_order", "wxyz"),
      ("velocity_frame", "world_link_origin"),
    ):
      if key in data and data[key].item() != expected:
        raise ValueError(f"{source}: expected {key}={expected}")
    if not np.isclose(float(data["fps"].item()), 50.0, rtol=0, atol=1e-8):
      raise ValueError(f"{source}: expected 50 Hz motion")
    pos, rest = data["root_pos"].astype(np.float32), data["rest"].astype(np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3 or rest.shape != (len(pos), 68):
      raise ValueError(f"{source}: expected root_pos (T,3) and rest (T,68)")
    if not np.isfinite(pos).all() or not np.isfinite(rest).all():
      raise ValueError(f"{source}: non-finite motion data")
    joints = data["joint_names"].tolist()
    canonical = [j.name for j in EVT_CFG.joints]
    if len(joints) != len(canonical) or set(joints) != set(canonical):
      raise ValueError(f"{source}: joints do not match EVT")
    perm = [joints.index(name) for name in canonical]
    norms = np.linalg.norm(rest[:, :4], axis=1, keepdims=True)
    if np.any(norms < 1e-6):
      raise ValueError(f"{source}: zero root quaternion")
    rest[:, :4] /= norms
    if {"starts", "lengths", "names"} & set(data.files):
      if not {"starts", "lengths", "names"} <= set(data.files):
        raise ValueError(f"{source}: incomplete clip-boundary metadata")
      starts, lengths, names = data["starts"], data["lengths"], data["names"].tolist()
    else:
      starts, lengths, names = np.array([0]), np.array([len(pos)]), [source.stem]
    if (
      starts.ndim != 1
      or lengths.shape != starts.shape
      or len(names) != len(starts)
      or len(starts) == 0
      or not np.issubdtype(starts.dtype, np.integer)
      or not np.issubdtype(lengths.dtype, np.integer)
      or np.any(lengths < 2)
      or not np.array_equal(starts, np.r_[0, np.cumsum(lengths)[:-1]])
      or int(lengths.sum()) != len(pos)
    ):
      raise ValueError(f"{source}: invalid clip boundaries")
    if any(
      not str(n) or str(n) in {".", ".."} or Path(str(n)).name != str(n) for n in names
    ):
      raise ValueError(f"{source}: clip names must be plain filenames")
    outputs = [output_dir / f"{str(name).removesuffix('.npz')}.npz" for name in names]
    if source.resolve() in {p.resolve() for p in outputs}:
      raise ValueError("Output must not overwrite the source motion")
    if len(set(outputs)) != len(outputs):
      raise ValueError(f"{source}: duplicate clip names")
    for output in outputs:
      if output.exists() and not overwrite:
        raise FileExistsError(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for start, length, output in zip(starts, lengths, outputs, strict=True):
      sl = slice(int(start), int(start + length))
      tmp = output.with_suffix(".tmp.npz")
      np.savez(
        tmp,
        root_pos_w=pos[sl],
        root_quat_w=rest[sl, :4],
        joint_pos=rest[sl, 4:33][:, perm],
        root_lin_vel_w=rest[sl, 33:36],
        root_ang_vel_w=rest[sl, 36:39],
        joint_vel=rest[sl, 39:68][:, perm],
        joint_names=np.array(canonical),
        fps=np.array(50.0),
        root_body=np.array("pelvis"),
        quaternion_order=np.array("wxyz"),
        velocity_frame=np.array("world_link_origin"),
      )
      tmp.replace(output)
  return outputs


def main(input_path: str, output_dir: Path, overwrite: bool = False) -> None:
  """INPUT_PATH accepts one compact NPZ, a directory, or a quoted glob."""
  source = Path(input_path).expanduser()
  files = (
    sorted(source.glob("*.npz"))
    if source.is_dir()
    else [Path(p) for p in sorted(glob.glob(str(source)))]
  )
  if not files:
    raise ValueError(f"No motion files matched: {input_path}")
  count = 0
  for file in files:
    count += len(convert_file(file, output_dir.expanduser(), overwrite))
  print(f"Converted {count} clips into {output_dir}")


if __name__ == "__main__":
  tyro.cli(main)
