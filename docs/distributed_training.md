# 单机多 GPU 训练

`supermimic.scripts.train` 支持一张 GPU 一个进程的同步数据并行。各卡独立运行真实环境、Replay Buffer 和想象 rollout，世界模型、Actor、Critic 在每次优化前平均梯度。默认 `return` 采样器汇总各卡新增的 bin 回报/计数，归一化统计使用锁定版本 RSL-RL 已有的全局统计实现。

实现参考了 DEX_MOSAIC 的 `source/rsl_rl/rsl_rl/algorithms/ppo.py::reduce_parameters`（初始广播、优化前平均梯度、rank 0 日志）和 allmimic 的 `allmimic/train.py::sync_grads`（WM/Actor/Critic 分别同步、独立采样 RNG）。针对本项目额外处理了 CUDA Graph 拆分、未使用 encoder 梯度、共享 return 采样统计和全卡 replay 就绪条件。allmimic 可选的学习率按卡数平方根缩放没有启用；固定全局规模的对照保持原学习率。

本次验收记录见 [distributed_validation.md](distributed_validation.md)。

## 启动

在 `supermimic_lqy` 中执行。EVT 模型需要同级 `../allmimic/assets/robots/evt`；本机已把 `../allmimic` 链接到 `/mnt/dataset/lqi3lr/dyh/allmimic`。动作数据通过 `--motion-file` 指定，支持引号包围的 glob。

参数里的环境数和 batch size **均为每个进程的数量**；启动信息会打印 local/global 数量。不要向 torchrun 指定固定 `--device`。

保持原始全局规模（4096 环境、WM batch 2048、policy batch 4096）的双卡命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run torchrun --standalone --nproc-per-node=2 \
  -m supermimic.scripts.train SuperMimic-GenCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz' \
  --num-envs 2048 \
  --runner.wm.batch-size 1024 \
  --runner.policy.batch-size 2048 \
  --runner.run-name evt_2gpu
```

四卡命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4 \
uv run torchrun --standalone --nproc-per-node=4 \
  -m supermimic.scripts.train SuperMimic-GenCoord-EVT \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz' \
  --num-envs 1024 \
  --runner.wm.batch-size 512 \
  --runner.policy.batch-size 1024 \
  --runner.run-name evt_4gpu
```

学习率不自动缩放，horizon、num_batches、replay_ratio 不随卡数改变。当前默认更新频率为 `num_envs * replay_ratio / (wm.num_batches * wm.batch_size * wm.horizon)`；环境数和 WM batch 同比拆分可以保持此频率。每卡加载完整动作库，采样 RNG 使用 `seed + 1000 * rank`，模型初始化使用独立 CPU 随机流（基础 seed、seed+1、seed+2 分别用于 WM、Actor、Critic），避免环境数量改变初始权重；参数随后由 rank 0 广播。

本机 GPU 0/1 与 GPU 2/3 分别属于两个 NUMA 节点，无 NVLink。先用同节点双卡测量，再决定是否使用四卡。独立回放和浮点归约使多卡结果不会逐步等同单卡，需要用多个 seed 验证最终效果。

## 执行模式与同步

- `--runner.execution auto`：单卡使用原有完整更新 Graph；多卡使用 `split`。启用 profile 时单卡也选择 split，以便分开测量 Actor/Critic。
- `eager`：不捕获外层更新；现有局部 `torch.compile` 仍启用，用于正确性参考。
- `split`：每次优化步骤捕获前向/反向；图外执行梯度平均，再用独立 Graph 执行裁剪与 fused Adam/Muon 更新。各步骤使用独立 Graph 内存池，避免跨学习器重放覆盖仍存活的数据。
- `full`：仅允许单卡且 profile=False，保留完整更新捕获路径。

默认每轮学习有 10 次 WM、1 次 Actor 和 20 次 Critic 优化，因此同步开销可能明显。当前 reducer 使用预分配连续缓冲区；梯度参与集合固定，未使用的 encoder 保持 `grad=None`，不会被优化器悄悄更新。所有卡在相同优化边界同步；若一张卡无有效回放窗口，所有卡共同跳过这次学习。

可通过 `--runner.check-sync-interval 100` 定期检查模型、normalizer、优化器和共享采样器状态是否一致。这是诊断开关，会增加通信，不建议用于正式吞吐对照。

首版支持 `train.py` 的 Actor-Critic/BPTT 路径及 return/uniform/start 采样。`adaptive` 在多卡下明确报错，以免各卡维护不同的隐藏采样状态；`train_ppo.py` 也会拒绝多卡启动。

## 保存、恢复与指标

只有 rank 0 写 TensorBoard/W&B、配置、checkpoint 和 ONNX。各卡参与指标归约：环境步数求和，loss/回报按计数汇总，阶段耗时取各卡最大值。日志 step 使用真实外层迭代号。

每次保存仍输出 `world_model.pt`、`actor.pt`、可选 `critic.pt`，并新增原子替换的 `training_state.pt`，包含学习模型、优化器、共享采样器状态、迭代/更新/全局采样计数、训练配置及累计训练墙钟时间。

恢复时使用原训练配置并增加：

```bash
--runner.resume logs/<task>/<timestamp>/training_state.pt
```

`max_iterations` 是恢复后要达到的绝对迭代号。允许调整每卡 batch，但要求其他学习器超参数匹配。恢复会重新创建环境、重新填充各卡 Replay Buffer，保留已有模型/优化器/共享采样器状态；不保存仿真现场、Replay Buffer 或 RNG 现场，因此不是逐步完全复现。旧的 `wandb_run_path` 仍只用于原有 WM 权重加载，不等同完整恢复。

## 完整训练性能对照

直接训练时可增加：

```bash
--runner.profile True --runner.profile-warmup-iterations 50 --runner.export-onnx False
```

每轮记录 `performance.jsonl`，包含真实采样、WM、Actor、Critic、整轮耗时和全局样本数。profile 会同步计时边界，存在测量开销。归一化和采样器同步包含在 collect；梯度通信包含在对应学习器阶段；整轮还包含 Replay Buffer 抽样和同步就绪检查。

首次遇到重置或新编译路径可能在预热阈值之后仍发生编译。`warmup/steady` 是按指定迭代划分的标签，分析时应检查尖峰、延长预热，并报告 compiler cache 状态。阶段计时不含日志、保存和评估；`elapsed_s` 包含先前迭代的 I/O，benchmark 另记录完整进程墙钟时间。

串行对比 1/2/4 卡及四卡扩大全局 batch：

```bash
uv run -m supermimic.scripts.benchmark_training \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz' \
  --output logs/scaling \
  --cases 1-split 2-eager 2-split 4-split 4-split-weak \
  --iterations 200 --warmup 50
```

默认使用原始模型规模、horizon、学习率和 batch。`4-split-weak` 保持每卡原 batch，因此全局 batch/环境数变成单卡的 4 倍；它的吞吐提升不能直接解释成达到同等回报的加速。`--extra` 后可追加训练参数，脚本保存精确命令、原始日志和 `summary.json`。不要并行运行其他训练干扰计时。

汇总已有 profile：

```bash
uv run -m supermimic.scripts.summarize_training logs/<task>/<timestamp>/performance.jsonl
```

## 效果评估

固定动作集、seed、环境数和步数，对各次保存的 checkpoint 运行：

```bash
uv run -m supermimic.scripts.evaluate_checkpoint \
  --checkpoint logs/<task>/<timestamp>/training_state.pt \
  --motion-file '/mnt/dataset/lqi3lr/dyh/supermimic/data/evt/*.npz' \
  --num-envs 128 --steps 1000 --seed 1234
```

这是无探索噪声、motion encoder 的固定预算评估，输出平均单步 reward、失败次数及每千环境步失败数，并附带 checkpoint 的训练时间和样本量。它不等同逐动作完成率；很短的评估可能尚未发生任何终止。默认追加到 `evaluations.jsonl`。若需保留多个历史点，应在下一次保存前复制 checkpoint，或及时评估原子更新后的 checkpoint。

最终应同时比较同样全局样本量下的效果，以及达到预设评估回报的墙钟时间，并重复多个 seed。短程 smoke/吞吐测试不能证明收敛质量相同。

## 正确性验证

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc-per-node=2 tests/distributed_worker.py
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc-per-node=4 tests/distributed_worker.py
```

固定数据测试检查全局 batch 与拆分 batch 的梯度/更新、跨 Graph 捕获/重放的 Muon/Adam 更新、未使用参数、全局 normalizer 的统计与地址稳定性，以及 ReturnSampler 与单进程合并数据的结果。Muon 内部低精度正交化会放大微小的归约舍入差异，因此采用数值容差，而非逐 bit 比较。
