#allmimic数据转换
python convert_evt_motions.py

#train
CUDA_VISIBLE_DEVICES=0 uv run src/supermimic/scripts/train.py \
  SuperMimic-MaxCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --num-envs 4096
  
#causal actor: 5 robot frames + 5 reference frames, no future actor input
CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
  uv run torchrun --standalone --nproc-per-node=3 \
  -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt_seed_lafan/*.npz' \
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


CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl OMP_NUM_THREADS=1 \
uv run python -m supermimic.scripts.play SuperMimic-MaxCoord-EVT \
  --checkpoint logs/SuperMimic-MaxCoord-EVT/2026-11-18_15-48-41-655270 \
  --motion-file /mnt/dataset/lqi3lr/dyh/supermimic/data/evt/walk1_subject1.npz \
  --viewer none \
  --video \
  --video-duration 50 \
  --video-width 1280 \
  --video-height 720 \
  --video-overlay reference \
  --video-dir videos/evt_causal_seed_lafan1

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
uv run python -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file /mnt/dataset/lqi3lr/dyh/supermimic/data/evt_seed_lafan/*.npz \
  --num-envs 4096 \
  --runner.actor-obs proprio \
  --runner.history-length 5 \
  --runner.tasks.motion.mode causal_history \
  --runner.tasks.motion.history-length 5 \
  --runner.critic-obs privileged_history \
  --runner.run-name walk1_critic_history5

# Dual-GPU EVT with cosine learning-rate decay, 2048 environments per GPU
CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run torchrun --standalone --nproc-per-node=2 --log-dir logs/torchrun --tee 3 \
  -m supermimic.scripts.train SuperMimic-MaxCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt_seed_lafan/*.npz' \
  --num-envs 2048 \
  --runner.wm.batch-size 1024 \
  --runner.policy.batch-size 2048 \
  --runner.actor-obs proprio \
  --runner.history-length 5 \
  --runner.tasks.motion.mode causal_history \
  --runner.tasks.motion.history-length 5 \
  --runner.critic-obs privileged_history \
  --runner.wm.lr 2.828427e-4 \
  --runner.policy.lr 2.828427e-4 \
  --runner.lr-decay-iterations 250000 \
  --runner.lr-min-ratio 0.03 \
  --runner.run-name evt_lafan_2gpu_2048_critic_history5_cosine


  
  uv run python -m supermimic.scripts.play SuperMimic-MaxCoord-EVT \
  --checkpoint logs/SuperMimic-MaxCoord-EVT/2026-11-18_15-48-41-655270 \
  --export-only \
  --export-dir exports/evt_causal_seed_lafan1

