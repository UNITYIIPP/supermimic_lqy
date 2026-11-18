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

#play
uv run --no-project --python .venv-cu128/bin/python \
  src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/xesxb0u1' \
  --motion-file data/evt/walk1_subject1.npz \
  --viewer viser \
  --video \
  --video-duration 30 \
  --video-dir videos/play

#export
uv run --no-project --python .venv-cu128/bin/python \
  src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/6ja94e4t' \
  --export-only \
  --export-dir exports/evt_6ja94e4t
