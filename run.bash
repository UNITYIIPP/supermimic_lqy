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


CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-MaxCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --num-envs 4096


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
#tensorbaord
  tensorboard --logdir logs/SuperMimic-MaxCoord-EVT/2026-11-12_17-19-04/ --load_fast=false