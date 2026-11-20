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

## Training curves

Training records scalar metrics to local TensorBoard by default. W&B recording
is disabled (`BaseRunnerCfg.wandb_enabled = False`); add `--runner.wandb-enabled True`
to enable W&B metrics and model uploads alongside local recording.
Metric names, values, logging frequency, and steps match between both backends.
This includes losses, policy statistics, real-environment rewards and episode
statistics, and performance metrics. W&B's automatic system monitoring and model
artifacts are separate from these scalar metrics.

TensorBoard events and a `config.json` snapshot of the training configuration are
saved alongside checkpoints in `logs/<TASK_ID>/<timestamp>/`. The PPO entry point
uses `logs/<TASK_ID>/ppo/<timestamp>/`. Events flush every 30 seconds and on normal
exit, Ctrl+C, or a Python exception. Existing runs are not backfilled.
Local checkpoints and ONNX export still work with W&B recording disabled.
Explicit remote inputs (`--registry-name` or `--runner.wandb-run-path`) still
access W&B to download data/models; use `--motion-file` for local motion data.

```bash
uv run tensorboard --logdir logs --port 6006 --load_fast=false
```

Open `http://localhost:6006`; forward port 6006 when training on a remote machine.
Use `--load_fast=false`: events preserve W&B's double-precision scalar values,
which TensorBoard 2.20's experimental fast loader does not display.
`--runner.log-interval` controls recording frequency for both backends. Both
runners use the completed training iteration as the TensorBoard and W&B step.

## EVT training

The EVT model and its STL meshes are packaged in `src/supermimic/assets/evt/`.
Legacy checkpoints that point to the original default `allmimic` EVT paths
use the packaged model. New control snapshots record paths relative to the
`supermimic` package. The robot's
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
In the default `future_window` mode, automatic ONNX export retains the existing conditions: actor
observations must use `proprio`, and the reference dataset must contain at most
15,000 frames. The single walking clip above meets this limit; the full dataset
uses checkpoint saving without automatic ONNX export.

### Causal actor observations

Enable the optional causal reference history when training a new policy:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  uv run torchrun --standalone --nproc-per-node=3 \
  -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file 'data/evt/*.npz' --runner.actor-obs proprio \
  --runner.history-length 5 --runner.tasks.motion.mode causal_history \
  --runner.tasks.motion.history-length 5 --runner.run-name evt_causal
```

For the 29-joint EVT robot the actor receives two windows, both ordered from
oldest to newest and including the current control step:

| Input | Shape per environment | Contents per frame |
| --- | --- | --- |
| Robot `history` | `[5, 93]` | Projected gravity (3), body angular velocity (3), joint position relative to default (29), joint velocity (29), previous action (29) |
| Reference `goal_motion` | `[5, 64]` | Absolute reference joint position (29), reference file `joint_vel` (29), relative anchor orientation in rot6d (6) |

The windows cover `t-4, t-3, t-2, t-1, t`, or 80 ms of past observations at
50 Hz. Each relative orientation uses the robot and reference anchor at that
historical step; cache it rather than recomputing it using the current robot
orientation. On reset, repeat the first observed frame across both windows,
retaining its reference file velocity. There is no future-frame wait or backward
difference velocity calculation. The supplied file velocity may itself have been
computed using offline processing; live sources must supply a causal velocity
estimate if end-to-end causality is required.

Normalization is shared across time with 93 robot and 64 reference statistics;
updates use the newest frame. The motion encoder consumes 320 normalized values
and emits the existing 32D latent. The decoder consumes 465 robot-history values
plus that latent (497 total) and emits 29 actions.

By default the Critic keeps its original 917D input: MaxCoord state (214),
privileged motion goal (670), and simulation context (33). Its ten reference samples are current
through nine future frames, spaced five control steps apart (up to 900 ms).
The dynamics network still receives state, action and simulation context without
reference goals. Rewards, terminations and multistep optimization are unchanged.
This is an asymmetric training setup: future targets guide training but are not
actor inputs at deployment. The velocity task remains separately configured;
its default centered smoothing is unchanged.

To test a Critic that also observes the actor's histories, add
`--runner.critic-obs privileged_history` to a **new causal training run**. For
example, from the project directory:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  uv run python -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file /mnt/dataset/lqi3lr/dyh/supermimic/data/evt/walk1_subject1.npz \
  --runner.actor-obs proprio --runner.history-length 10 \
  --runner.tasks.motion.mode causal_history \
  --runner.tasks.motion.history-length 10 \
  --runner.critic-obs privileged_history --runner.run-name evt_critic_history10
```

The expanded Critic concatenates, in order: the original privileged input (917),
current reference root linear velocity (3), flattened robot history (`93 * H`),
and flattened causal reference history (`64 * G`). The velocity comes from the
motion file at the current frame and is rotated into the **reference root's**
frame, not the robot's frame. Both windows are exactly the actor's cached windows,
including noise, reset padding and previous actions; no extra history buffer is
allocated. The total is `920 + 93*H + 64*G`: 1705 for H=G=5, or 2490 for H=G=10.
Future reference samples remain available to the Critic. This history expansion
is a project-specific experiment, not a copy of DEX_MOSAIC's Critic history.

This mode changes only Critic inputs and its first-layer width. Actor/ONNX inputs,
Critic hidden widths, dynamics, reward, termination and rollout horizons retain
their existing settings. Histories stay differentiable for the actor's value
bootstrap. The mode is saved in configuration and checked when resuming; old
configurations default to `legacy`. An old Critic/training-state checkpoint cannot
be resumed into the expanded input layout. Use `--runner.critic-obs legacy` for
the original Critic; resume an expanded-mode run with its matching configuration.

The default remains `future_window`, so old configurations and checkpoints keep
their original layout. Start a new causal run: an old actor/training-state
checkpoint cannot be directly resumed into the new architecture. Both history
lengths are saved in the runner configuration. Replay now stores a 64D reference
frame alongside each transition; with robot history enabled, storing the added
93D robot frame and 64D reference frame costs about 2.4 GiB per GPU for the default
4096 environments and 1000 buffer steps (float32).

EVT causal checkpoints automatically export `policy_motion.onnx` plus its JSON
contract at checkpoint saves, including for datasets exceeding 15,000 frames.
The export contains no motion dataset. Inputs are float32 `history: [1,5,93]`
and `goal_history: [1,5,64]`, output `actions: [1,29]`; normalization is embedded.
Use the lengths recorded in the JSON contract when configuring different windows.

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
- `goal`: `[1, goal_dim]`, the externally supplied task target in legacy mode;
  causal motion uses `goal_history: [1, motion_history_length, 64]` as above.
- `actions`: `[1, num_joints]`, raw policy actions. Convert them with
  `joint_target = action_offset + action_scale * actions`.

For EVT with history length 1, `proprio` history is `[1, 1, 93]`; GenCoord
`rep_state` history is `[1, 1, 68]`. Legacy motion goals are `[1, 640]`: ten frames of
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

For a local training run, pass its directory or `training_state.pt` using
`--checkpoint`. This loads one atomic snapshot of the actor, world model and
control settings, and works while training continues. W&B is not required;
supply the reference clip explicitly with `--motion-file`:

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 \
uv run python -m supermimic.scripts.play SuperMimic-MaxCoord-EVT \
  --checkpoint logs/SuperMimic-MaxCoord-EVT/2026-11-16_23-43-02-665996 \
  --motion-file /mnt/dataset/lqi3lr/dyh/supermimic/data/evt/walk1_subject1.npz \
  --viewer none --video --video-duration 30 --video-overlay reference \
  --video-dir videos/evt_causal
```

Use exactly one of `--checkpoint` and `--wandb-path`. Local checkpoints also
support `--export-only --export-dir <DIR>` without reference motions.

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

## Multi-GPU training

Single-node `torchrun` training, profiling, checkpoint resume and local evaluation
are documented in [docs/distributed_training.md](docs/distributed_training.md).
Environment counts and learner batch sizes are **per rank**; divide all three by
GPU count to preserve the single-GPU global workload.
