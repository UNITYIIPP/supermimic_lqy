#allmimic数据转换
python convert_evt_motions.py

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
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 uv run torchrun --standalone --nproc-per-node=3   -m supermimic.scripts.train SuperMimic-MaxCoord-EVT   --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz'   --runner.actor-obs proprio   --runner.run-name evt_3gpu_MAX

#causal actor: 5 robot frames + 5 reference frames, no future actor input
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  uv run torchrun --standalone --nproc-per-node=3 \
  -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz' \
  --runner.actor-obs proprio \
  --runner.history-length 5 \
  --runner.tasks.motion.mode causal_history \
  --runner.tasks.motion.history-length 5 \
  --runner.run-name evt_3gpu_MAX_causal

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run python -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz' \
  --num-envs 4096 \
  --runner.actor-obs proprio \
  --runner.history-length 5 \
  --runner.tasks.motion.mode causal_history \
  --runner.tasks.motion.history-length 5 \
  --runner.run-name evt_single_MAX_causal_gpu2

#play
uv run src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/xesxb0u1' \
  --motion-file data/evt/walk1_subject1.npz \
  --viewer viser \
  --video \
  --video-duration 30 \
  --video-dir videos/play

#export
uv run src/supermimic/scripts/play.py \
  SuperMimic-GenCoord-EVT \
  --wandb-path '244033717-huazhong-university-of-science-and-technology/supermimic/6ja94e4t' \
  --export-only \
  --export-dir exports/evt_6ja94e4t
