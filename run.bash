#allmimic数据转换
python convert_evt_motions.py
uv run -m supermimic.scripts.convert_evt_motions   --input-path /mnt/dataset/lqi3lr/dyh/allmimic/data/processed/evt_seed_lafan_train_flat_v1_grounded_filtered.npz   --output-dir data/evt_seed_lafan

#train
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --num-envs 4096 \
  --runner.actor-obs proprio \
  --runner.run-name evt_multi_motion


CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 \
TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run torchrun --standalone --nproc-per-node=4 \
  -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt_seed_lafan/*.npz' \
  --runner.actor-obs proprio \
  --batch-size-scope per_rank \
  --num-envs 4096 \
  --runner.wm.lr 2e-4 \
  --runner.policy.lr 2e-4


CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-MaxCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --num-envs 4096


#distributed: keep the same GLOBAL environment / WM / policy batch sizes.
# For four GPUs, use CUDA_VISIBLE_DEVICES=0,1,2,3 and --nproc-per-node=4.
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 uv run torchrun \
  --standalone --nproc-per-node=2 src/supermimic/scripts/train.py \
  SuperMimic-MaxCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --batch-size-scope global --num-envs 4096 \
  --runner.wm.batch-size 2048 --runner.policy.batch-size 4096 \
  --seed 42

# To resume, append --runner.resume logs/<TASK>/<TIMESTAMP>/training_state.pt
# --runner.max-iterations is the final total, not additional iterations.
# Historical snapshots: checkpoints/model_<zero-based collection step>.pt
# Default save interval: 1000 collection iterations (100 console logs by default).
# Resume can also use --runner.resume <RUN_DIR>/checkpoints/model_0001000.pt

#play
uv run src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/xesxb0u1' \
  --motion-file data/evt/walk1_subject1.npz \
  --viewer viser \
  --video \
  --video-duration 30 \
  --video-dir videos/play


uv run src/supermimic/scripts/play.py \
  SuperMimic-MaxCoord-EVT \
  --checkpoint-dir logs/SuperMimic-MaxCoord-EVT/2026-11-12_17-19-04 \
  --motion-file data/evt/walk1_subject1.npz \
  --viewer viser \
  --video \
  --video-duration 30 \
  --video-dir videos/play

  uv run src/supermimic/scripts/play.py \
  SuperMimic-MaxCoord-EVT \
  --checkpoint-dir logs/SuperMimic-MaxCoord-EVT/2026-11-12_17-19-04 \
  --export-onnx \
  --motion-file data/evt/walk1_subject1.npz \
  --viewer viser \
  --video \
  --video-duration 200 \
  --video-dir videos/play

#export
uv run src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/6ja94e4t' \
  --export-only \
  --export-dir exports/evt_6ja94e4t
  
uv run src/supermimic/scripts/play.py \
  SuperMimic-MaxCoord-EVT \
  --checkpoint-dir logs/SuperMimic-MaxCoord-EVT/2026-11-12_17-19-04 \
  --export-only
# Historical model: replace this timestamp with the NEW training run to inspect.
# Snapshots are created by new training runs; older runs are not backfilled.
checkpoint_run_dir="logs/SuperMimic-MaxCoord-EVT/REPLACE_WITH_TRAINING_TIMESTAMP"

# Record the chosen stage into <run>/videos/model_0001000/ and export into
# <run>/exports/model_0001000/. No --video-dir/--export-dir is needed.
MUJOCO_GL=egl uv run src/supermimic/scripts/play.py \
  SuperMimic-MaxCoord-EVT \
  --checkpoint "logs/SuperMimic-MaxCoord-EVT/2026-11-14_00-46-17-916206/checkpoints/model_0199000.pt" \
  --motion-file data/evt/walk1_subject1.npz \
  --viewer none --video --video-duration 30 --export-onnx

# Export the same historical model without loading any motion data or simulation.
uv run src/supermimic/scripts/play.py \
  SuperMimic-MaxCoord-EVT \
  --checkpoint "$checkpoint_run_dir/checkpoints/model_0001000.pt" \
  --export-only

#tensorbaord
  tensorboard --logdir logs/SuperMimic-MaxCoord-EVT/2026-11-12_17-19-04/ --load_fast=false
