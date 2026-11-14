# 单机多 GPU 训练

主入口 `src/supermimic/scripts/train.py` 支持 torchrun：每卡一个进程，
各自创建 MuJoCo 环境、动作数据副本和 replay buffer。Dynamics、Actor、Critic
在每次对应的优化器更新前平均梯度，再进行原有的梯度裁剪和 Muon/Adam 更新。
网络结构、学习率、reward、PD、action scale 均由原有配置决定。

## 代码位置

| 文件（相对 `src/supermimic/`） | 修改职责 |
| --- | --- |
| `utils/distributed.py` | NCCL 进程组、设备/种子、batch 解析、梯度 bucket 和统计归约 |
| `scripts/train.py` | torchrun 入口、每卡配置分配、启动校验与清理 |
| `learners/learner_utils.py` | 拆分反向和优化器 Graph，在图外同步梯度、记录耗时 |
| `learners/wm_learner.py`、`policy_learner.py` | 接入三套网络的同步更新，保留 full 路径 |
| `runners/supermimic_runner.py` | 广播初始化、同步就绪判断、日志编号、状态恢复及原子保存 |
| `runners/reset_sampling.py` | 合并新增采样统计、恢复全局采样器状态 |
| `utils/logger.py` | 全卡汇总、rank 0 输出、episode 结算与失败计数 |

`scripts/train_ppo.py` 增加多卡入口拒绝提示，避免误用；本次未改变网络定义。

## 启动和 batch 含义

原单卡命令仍可使用，默认 `--batch-size-scope per_rank`，默认单卡完整 CUDA Graph。
默认配置的 WM batch 是 **2048 条长度 8 的序列**，Policy batch 是
**4096 个起点、各想象 32 步**；环境数 4096 不会自动修改这两个 batch。
WM 每次 update 有 10 个 batch，Actor 每次 update 有 1 个 batch；Critic
仍将想象轨迹拆成 4 个 minibatch，训练 5 轮。

固定全局环境数和 batch 的双卡示例（从项目目录运行）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 uv run torchrun \
  --standalone --nproc-per-node=2 src/supermimic/scripts/train.py \
  SuperMimic-MaxCoord-EVT \
  --motion-file 'data/evt/*.npz' \
  --batch-size-scope global --num-envs 4096 \
  --runner.wm.batch-size 2048 --runner.policy.batch-size 4096 \
  --seed 42
```

| 配置 | 每卡环境数 | 每卡 WM batch | 每卡 Policy batch | 全局环境数 |
| --- | ---: | ---: | ---: | ---: |
| 单卡，global | 4096 | 2048 | 4096 | 4096 |
| 双卡，global | 2048 | 1024 | 2048 | 4096 |
| 四卡，global | 1024 | 512 | 1024 | 4096 |
| 双卡，per_rank | 4096 | 2048 | 4096 | 8192 |

`per_rank` 表示命令中的三个数字分别应用到每卡，全局 batch 随卡数增加。
`global` 会严格平分三个数字，不能整除时直接报错。四卡将设备列表换成
`0,1,2,3`、进程数换成 4 即可。三卡可用 3072/1536/3072 这组全局数字。
不要给 torchrun 额外指定固定的 `--device cuda:0`；设备由 LOCAL_RANK 分配。
每卡必须保留各活跃任务的采样槽；Actor-Critic 每卡 `policy.batch_size * policy.horizon`
需要能被 4 整除。

归一化使用已安装 rsl_rl 的全局样本数、均值和方差归约；ReturnSampler
合并各卡本轮新增的 return 总和与样本数，每卡使用同一采样概率。
模型从 rank 0 广播，各卡环境与探索使用不同随机种子。显式 `--seed`
使网络初始化不受环境数影响；未指定种子的单卡保留原有初始化路径，多卡
使用环境种子（环境未设置时为 42）。

`auto` 在多卡选择 `split`：前向/反向 Graph → 图外 NCCL 梯度归约 →
裁剪/优化器 Graph。全局没有参与损失的 encoder 保持 `grad=None`，
不进入通信 bucket、不产生优化器更新。支持 `--runner.execution eager`
排查问题；多卡不能用 `full`。本次支持主入口的 Actor-Critic 和 BPTT，
sampling mode 为 return/uniform/start。PPO 入口和 adaptive 采样不支持多卡。

## 日志、模型和恢复

只有 rank 0 写 TensorBoard、W&B 和模型文件。控制台保留原来的
`Learning iteration 27/100000`，TensorBoard/W&B 横轴也保留日志序号：
默认每 10 个环境迭代记一次日志，从第 0 次开始记录。
例如序号 27 对应 271 次环境迭代；卡数不会额外乘到日志序号上。
`Total steps` 为所有卡累计的真实环境步数。新增 GPU count、Environment
iterations、Learning updates，以及 `Counters/*` 指标便于核对。

episode 在失败、动作片段结束、超时时都结算并清零，避免跨片段累计长度。
因此修正后的 mean_reward/mean_episode_length 与旧日志可能不同。
`Train/failed_episodes` 每个失败环境计一次，即使触发多个失败条件；
`Train/completed_episodes`、`motion_end_count`、`time_out_count` 为**本日志区间**
计数，`failure_rate` 为区间失败数/结束数。结束原因可能重叠，不能相加得到总数。

每次保存保留带编号的完整历史 checkpoint，同时更新最新文件：

```text
logs/<TASK_ID>/<timestamp>/
  config.json            # 每卡实际配置 + requested/local/global batch
  robot_control.json     # 原有控制参数
  actor.pt               # play / ONNX 导出
  world_model.pt         # play 所需的模型/表示/控制参数
  critic.pt              # Actor-Critic 时保存
  training_state.pt      # 恢复：模型、优化器、归一化、采样统计、迭代/日志计数
  checkpoints/
    model_0000000.pt      # 完整快照，包含播放/导出需要的配置和控制参数
    model_0001000.pt
    ...
```

模型各自先写临时文件，再原子替换；保存失败会通知所有进程。
保留原来的保存周期，并在训练正常结束时补存最后状态。未到下个保存点被强制
结束时，使用上一个已保存状态。各最新文件分别替换；训练中的 play/导出建议
使用 `--checkpoint <run>/checkpoints/model_0001000.pt`，从同一完整快照读取，
避免混用不同保存阶段。历史文件不会随最新文件更新而改变；支持硬链接的文件
系统让 training_state.pt 与最近历史文件共享存储，否则复制为独立文件。
旧的 `--checkpoint-dir` 用法和无需动作数据的 `--export-only` 继续可用。
历史模型的 ONNX 默认写入 `<run>/exports/<model>/`，视频默认写入
`<run>/videos/<model>/`。编号为底层 collection step，不是终端日志编号，
也不随 GPU 数量倍增；默认保存间隔 1000 对应约 100 条终端日志。

恢复时在原训练命令后添加：

```bash
--runner.resume logs/<TASK_ID>/<timestamp>/training_state.pt \
--runner.max-iterations 1000000
```

也可以将 `--runner.resume` 指向 `checkpoints/model_0001000.pt` 选择历史阶段。

`max_iterations` 是最终环境迭代上限；恢复使用新的日志目录，日志序号继续。
必须保持动作文件及顺序、任务、网络结构、loss 和 learner 超参数一致；允许
修改 batch 和 GPU 数量，会打印新旧 batch 配置。`log_interval` 必须保持一致。
只恢复训练状态，不恢复 replay、环境状态、未结束 episode 或逐卡 RNG；
会重新预热，不能保证恢复后的轨迹逐位一致。旧的 actor.pt/world_model.pt
仍能 play/导出，但没有优化器等状态，不能作为 `--runner.resume` 输入。

## 运行诊断

添加 `--runner.profile True`，rank 0 会写 `performance.jsonl`，区分采样、
WM、Actor、Critic 和三个梯度同步耗时；阶段用时取最慢卡。
同步时间包含梯度打包、NCCL、解包，是 WM/Actor/Critic 阶段的子集。
profile 会同步 CUDA 阶段边界，单卡也会切换 split，不用于推断默认单卡的实际速度。
`--runner.check-sync-interval 100` 可定期检查三套网络、归一化、优化器与
ReturnSampler 是否一致；此诊断开销较大，默认关闭。

正式训练速度应在编译预热后测量；PCIe 通信和小 batch 可能抵消多卡的计算收益。
