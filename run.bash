#allmimic数据转换
python convert_evt_motions.py

#train
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-GenCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --num-envs 4096 \
  --runner.actor-obs proprio \
  --runner.run-name evt_multi_motion

#play
uv run src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/xesxb0u1' \
  --motion-file data/evt/walk1_subject1.npz \
  --viewer viser \
  --video \
  --video-duration 30 \
  --video-dir videos/play