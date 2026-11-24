#allmimic数据转换
uv run --no-project --python .venv-cu128/bin/python \
  -m supermimic.scripts.convert_evt_motions \
  --input-path /home/mig/dou/allmimic/data/train_data \
  --output-dir data/evt

#train
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python .venv-cu128/bin/python \
  src/supermimic/scripts/train.py \
  SuperMimic-MaxCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --num-envs 4096 \
  --runner.actor-obs proprio \
  --runner.run-name evt_multi_motion

CUDA_VISIBLE_DEVICES=3 WANDB_MODE=online uv run --no-dev \
  src/supermimic/scripts/train.py \
  SuperMimic-MaxCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt_seed_lafan/*.npz' \
  --num-envs 4096 \
  --runner.actor-obs proprio \
  --runner.run-name evt_multi_lafan_seed_motion_ama

CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online uv run --no-dev   \
src/supermimic/scripts/train.py   SuperMimic-GenCoord-EVT   \
--motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz'  \
--num-envs 4096   --runner.actor-obs proprio  \
 --runner.run-name evt_multi_lafan_motion_ama_new_obs_delay   --env.sim.mujoco.iterations 20 \
  --env.sim.mujoco.ls-iterations 40 


CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online uv run --no-dev \
  src/supermimic/scripts/train.py SuperMimic-GenCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz' \
  --num-envs 4096 \
  --runner.actor-obs proprio \
  --runner.run-name evt_multi_lafan_motion_ama_new_obs_delay \
  --env.sim.mujoco.iterations 20 \
  --env.sim.mujoco.ls-iterations 40


#play
CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl uv run --no-dev \
  src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/enmyex61' \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt_seed_lafan/walk1_subject1.npz' \
  --viewer none \
  --video \
  --video-duration 230 \
  --video-dir 'logs/SuperMimic-GenCoord-EVT/2026-11-22_19-33-40/videos/play' \
  --export-onnx \
  --export-dir 'logs/SuperMimic-GenCoord-EVT/2026-11-22_19-33-40/exports'

#export
uv run --no-dev \
  src/supermimic/scripts/play.py \
  SuperMimic-MaxCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/e3ciyzpm' \
  --export-only \
  --export-dir exports/evt_e3ciyzpm
