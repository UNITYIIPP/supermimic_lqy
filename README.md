# SuperMimic

A [SuperTrack](https://montreal.ubisoft.com/en/supertrack-motion-tracking-for-physically-simulated-characters-using-supervised-learning/)
implementation for training world models and policies to mimic motion data, built on [mjlab](../mjlab_research).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone https://github.com/UNITYIIPP/supermimic_lqy.git supermimic
git clone git@github.com:kevinzakka/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation
- `SuperMimic-GenCoord-EVT` — EVT generalized coordinates representation
- `SuperMimic-MaxCoord-EVT` — EVT maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## EVT training

EVT uses the sibling asset `../allmimic/assets/robots/evt/urdf/evt2_simple.xml`
and its mesh files. The source assets are read without modification. The robot's
embedded ground and actuators are replaced in memory by the tracking scene's
ground and 29 native MuJoCo position PD actuators.

All EVT control settings are in [`src/supermimic/robots/evt.py`](src/supermimic/robots/evt.py):
`default_joints()` defines Kp, Kd, effort limits, nominal velocity limits, initial
joint positions and action scales; `EVT_CFG` defines the asset paths, root height
and delay range. The current defaults use action scale **0.25 rad**, root height
**0.95 m**, soft joint limit factor **0.9**, and **zero actuator delay**.
The target is `default_joint_pos + action_scale * action`. Nominal velocity
limits are recorded for deployment and are not hard velocity constraints in
MuJoCo. Passive damping, armature, friction, inertias and joint ranges come from
the XML. No additional system identification randomization is enabled.
Tracking resets still initialize from the sampled reference frame; the configured
joint pose supplies the action offset and relative-position observation baseline.

The separate [`src/supermimic/tasks/evt_env_cfgs.py`](src/supermimic/tasks/evt_env_cfgs.py)
uses the basic tracking environment, with 14 tracked bodies, pelvis as anchor,
5 ms physics steps and 20 ms control steps. Foot friction follows G1's shared
randomization in `[0.3, 1.2]`; base COM and encoder bias randomization are retained.
The original G1 environment factories remain in `tasks/env_cfgs.py`.

First synchronize dependencies, then convert the existing **EVT** compact motions
(`root_pos` + `rest`, 50 Hz, wxyz quaternions, world-frame root velocities):

```bash
uv sync --group dev
uv run -m supermimic.scripts.convert_evt_motions \
  --input-path ../allmimic/data/Lafan_urdf --output-dir data/evt
```

The converter accepts a single clip, a packed archive with clip boundaries, a
directory, or a quoted glob. It reorders joints by name, normalizes quaternions
and writes one NPZ per clip, preserving clip boundaries for the motion loader.
Existing outputs require an explicit `--overwrite`.

Start with one motion and a small environment batch:

```bash
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-EVT --motion-file data/evt/walk1_subject1.npz \
  --num-envs 64 --runner.actor-obs proprio --runner.run-name evt_walk \
  --runner.wm.batch-size 64 --runner.policy.batch-size 64
```

Use `SuperMimic-MaxCoord-EVT` for the maximal coordinate representation. To use
all converted clips, pass `--motion-file 'data/evt/*.npz'`. EVT tasks require
an explicit motion selection and do not default to the G1 motion registry.
Train new EVT models; G1 checkpoints are incompatible despite matching joint
counts. Network architecture and learning algorithms use the existing framework.

Each EVT run records `robot_control.json` and embeds the effective control
settings in its world model checkpoint and ONNX metadata. Playback restores
the saved settings before constructing the environment, so later edits to
`EVT_CFG` apply to new runs. ONNX `robot_control` metadata contains the full
precision parameter snapshot alongside the standard deployment fields.
Automatic ONNX export retains the framework's existing conditions: actor
observations must use `proprio`, and the reference dataset must contain at most
15,000 frames. The single walking clip above meets this limit; the full dataset
uses checkpoint saving without automatic ONNX export.

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-EVT \
  --wandb-path <RUN_PATH> --motion-file data/evt/walk1_subject1.npz
```

## Exporting ONNX without motion data

Export the saved EVT policy without loading any NPZ files or starting a simulator:

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-EVT \
  --wandb-path <RUN_PATH> --export-only --export-dir exports/evt
```

This writes `policy_motion.onnx` and `policy_motion.json`. Without `--export-dir`,
the output directory is `exports/<TASK_ID>/<RUN_ID>`. The checkpoint and runner
configuration are fetched from W&B, using the existing artifact cache. No motion
artifact is downloaded. This mode has no 15,000-frame limit, works on CPU, and
supports both `proprio` and `rep_state` policies with their original input layouts.
Use `--export-onnx` instead of `--export-only` to export and then continue normal
playback (playback still requires reference motions).

The ONNX has two **float32** inputs and one output, with batch size fixed at 1:

- `history`: `[1, history_length, obs_dim]`, oldest frame first, newest last.
  Supply unnormalized observations; normalization is included in the graph.
- `goal`: `[1, goal_dim]`, the externally supplied task target.
- `actions`: `[1, num_joints]`, raw policy actions. Convert them with
  `joint_target = action_offset + action_scale * actions`.

For EVT with history length 1, `proprio` history is `[1, 1, 93]`; GenCoord
`rep_state` history is `[1, 1, 68]`. Motion goals are `[1, 640]`: ten frames of
`[joint_pos(29), joint_vel(29)]`, followed by ten relative anchor orientations
(rot6d). Frames are spaced five control steps apart. The JSON sidecar and the
ONNX `policy_contract` metadata document the precise layout, joint/body order,
input shapes and action mapping. Unlike the training-time motion export, this
graph has no `time_step` input or embedded reference-motion outputs. Deployment
must construct the goal each step; this is not an autonomous motion generator.
A `rep_state` policy still needs the corresponding root/body state estimates.

Both files contain the **checkpoint's** PD gains, action scales, offsets, initial
joint positions, torque/nominal velocity limits, control period, delay settings,
and complete `robot_control` snapshot at full precision. Later changes to
`robots/evt.py` do not silently replace these saved settings. Old checkpoints
without `robot_control` (including existing G1 checkpoints) fail explicitly;
the exporter does not guess their training control parameters.

`--export-task velocity` exports `policy_velocity.onnx` with a 3D
`[vx_b, vy_b, wz_b]` goal; `--export-task all` exports both encoders. The default
is `motion`. A warning is emitted if the selected encoder had zero training
slots, since the presence of encoder weights does not establish that it was trained.

## Recording playback

Add `--video` to record an MP4 while using Viser or the native viewer:

```bash
uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-EVT \
  --wandb-path <RUN_PATH> --motion-file data/evt/walk1_subject1.npz \
  --viewer viser --video --video-duration 30 --video-dir videos/play
```

For a server without a desktop, use `--viewer none`. It exits automatically
when recording finishes and does not require a browser connection:

```bash
MUJOCO_GL=egl uv run src/supermimic/scripts/play.py SuperMimic-GenCoord-EVT \
  --wandb-path <RUN_PATH> --motion-file data/evt/walk1_subject1.npz \
  --viewer none --video --video-duration 30 \
  --video-width 1280 --video-height 720 --video-overlay reference
```

`<RUN_PATH>` is `entity/project/run_id`. Use the same task/representation as
training, including `SuperMimic-MaxCoord-EVT` for a MaxCoord model.

Videos are streamed to
`videos/play/<task>_<run_id>_<timestamp>.mp4` (or `--video-dir`) as H.264/YUV420P.
FFmpeg is selected from the system or the bundled `imageio-ffmpeg` executable;
no separate system FFmpeg installation is required. The default resolution is
1280 x 720; both dimensions must be positive even integers.

Each successful control step produces one frame, so the current EVT environment
records at 50 FPS. `--video-duration` uses simulation seconds, rounded up to a
control step. Pausing does not add frames; a paused single-step adds one. Viewer
speed changes affect recording wall time, while the exported video plays at
normal simulation speed. Interactive playback continues after recording ends.
Closing the viewer or pressing Ctrl+C saves the frames recorded so far.

Choose `--video-overlay robot` for the policy robot alone, `reference` (default)
to add a green reference-motion ghost, or `all` to also add the blue world-model
prediction. The video uses an independent camera tracking the robot; browser
camera changes, panels, and UI visibility toggles are not recorded.

Recording starts from the selected clip's first frame by default. Use
`--motion-start random` to start within that clip, or explicitly select
`--motion-start start`. With multiple motions, Viser's Motion Clip selector
changes the clip on reset; headless/native recording uses the first loaded clip.
Episode termination resets the robot within the selected clip and recording
continues until the requested duration. Very short clips may therefore repeat.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
