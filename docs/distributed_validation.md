# 多卡实现验收记录

日期：2026-11-12。工作目录：`supermimic_distributed/supermimic_lqy`。

实现与启动参数见 [distributed_training.md](distributed_training.md)。当前支持单机同步 WM + Actor/Critic/BPTT 数据并行；本次真实环境验收使用 `SuperMimic-GenCoord-EVT` 和 `walk1_subject1.npz`，不代表已经覆盖所有机器人、动作库、任务组合或 PPO。

## 环境与测量约束

- Python 3.13.18、PyTorch 2.15.0+cu132、CUDA 13.2、4 张 RTX 5090；GPU 0/1 和 2/3 分属两个 NUMA 节点，无 NVLink。
- 验收中发现 GPU 0 正在运行另一个原项目的 MaxCoord 训练，未终止或修改该任务。涉及 GPU 0 的四卡短程结果只作为功能验收，不能据此计算可信的四卡加速比。
- 另行使用 GPU 2 单卡和 GPU 2/3 双卡进行顺序性能对照。GPU 0 的任务仍共享主机资源，因此这不是完全独占机器的测量。
- 编译缓存沿用本机缓存，未清空。预热包含首次编译、Graph 捕获和 replay 填充，不能视为统一的冷启动编译基准。
- 原始日志、命令、profile 和 checkpoint 位于被 Git 忽略的 `logs/`，避免将训练数据和模型权重纳入代码变更。

## 已通过的正确性检查

| 检查 | 结果与范围 | 原始记录 |
| --- | --- | --- |
| 双卡固定数据 | eager、split 的 Muon/Adam 更新与单进程合并 batch 在数值容差内一致 | `logs/validation/supermimic-final-unit-check.log` |
| 四卡固定数据 | 同上；覆盖预热、捕获及多次 replay | `logs/validation/supermimic-final-four-check.log` |
| 全局统计 | normalizer 均值/标准差与合并样本一致，buffer 地址稳定；ReturnSampler 与合并数据参考一致 | 同上 |
| 特殊梯度与采样 | 全局未使用参数保留 `grad=None`；某卡缺失梯度补零；全卡共同判断 replay 就绪；各卡 RNG 独立，模型初始化不受环境数量影响 | 同上 |
| 真实双卡 eager/split | 各完成 28 轮，定期检查参数、normalizer、优化器和共享采样器一致 | `logs/validation/supermimic-dual-{eager,split}-smoke.log` |
| 真实双卡恢复 | 从 split checkpoint 恢复至绝对迭代 40，重新填充 replay 后继续更新 | `logs/validation/supermimic-dual-resume.log` |
| 最终版本默认规模恢复 | 使用双卡默认模型规模 checkpoint，从第 240 轮继续至 280 轮；每 10 轮检查模型、normalizer、优化器及共享采样器一致，重新捕获拆分 Graph | `logs/validation/supermimic-final-resume.log` |
| 单卡原有 full Graph | 12 轮回归通过 | `logs/validation/supermimic-full-regression.log` |
| 四卡固定/扩大全局规模 | 各完成 64 轮，每 16 轮检查共享状态；包括模型和训练状态保存 | `logs/distributed_benchmark/{4-split,4-split-weak}/stdout.log` |
| 最终 checkpoint 评估 | 固定 seed、无探索噪声，16 环境、64 步 smoke 通过，正确读出累计训练时间和采样/更新计数 | `logs/validation/supermimic-final-evaluation.log` |

固定数据验证针对共用的梯度同步/Graph/优化器执行路径；真实环境运行的随机轨迹不同，未声称单卡与多卡每一步权重或回报完全相同。Muon 的低精度正交化可能放大归约舍入差异，因此采用数值容差。

最终恢复后的 checkpoint 位于 `logs/final_resume/SuperMimic-GenCoord-EVT/2026-11-12_18-11-14-384401/training_state.pt`：学习更新数由 93 增至 106，全局采样数由 983,040 增至 1,146,880，累计训练墙钟由 81.19 s 增至 134.83 s；动作文件 manifest 保持一致。评估结果保存在同目录 `evaluations.jsonl`。验收启动的 GPU 进程均已结束。

四卡功能测试使用小网络 WM `(64,64)`、Actor `(64,32)`、Critic `(32,16)`，WM/policy horizon 均为 4、WM num_batches 为 1、replay_ratio 为 2。固定规模为全局 64 环境、WM batch 32、policy batch 64；扩大规模为全局 256 环境、WM batch 128、policy batch 256。原始 `logs/distributed_benchmark/summary.json` 包含计时，但运行中存在 GPU 竞争，且早期 case 在优化器 Graph 优化前执行，因此不将这些数字作为性能对照。

## 默认模型规模的单卡/双卡性能

固定全局 4096 环境、WM batch 2048、policy batch 4096。使用默认 WM `(2048,2048,2048)`、Actor `(2048,1024,512,256)`、Critic `(256,128)`；WM horizon 8、policy horizon 32、WM num_batches 10、policy num_batches 1、replay_ratio 16、学习率 2e-4。为短程测试将 replay 容量设为 128，其余学习超参数不变。

最终对照每组 240 轮，前 80 轮作为预热，后 160 轮包含 64 次完整 WM+Actor+Critic 学习更新和 655,360 个真实环境步。两组最终均为 93 次学习更新、983,040 个环境步。没有同时运行本次验收的其他 GPU 测试。

| 指标 | 单卡 GPU 2 | 双卡 GPU 2/3 |
| --- | ---: | ---: |
| 启动时间 | 9.99 s | 10.24 s |
| 预热训练时间（80 轮） | 45.63 s | 52.91 s |
| 稳定训练时间（160 轮） | 25.25 s | 17.82 s |
| 稳定完整训练吞吐 | 25,958 环境步/s | 36,781 环境步/s |
| 稳定学习更新率 | 2.535 次/s | 3.592 次/s |
| 真实采样，平均每外层迭代 | 23.03 ms | 21.89 ms |
| WM，平均每外层迭代 | 54.51 ms | 40.69 ms |
| Actor，平均每外层迭代 | 64.87 ms | 36.98 ms |
| Critic，平均每外层迭代 | 12.15 ms | 8.43 ms |
| 整轮，平均每外层迭代 | 157.80 ms | 111.36 ms |
| 完整进程墙钟时间 | 92.07 s | 94.91 s |

本次稳定段双卡约 **1.42×** 吞吐，单轮训练时间降低约 **29.4%**；收益主要来自 WM/Actor。学习迭代的耗时范围分别为 357.7–368.1 ms、244.1–251.7 ms，稳定窗口内未出现明显编译尖峰。真实采样的收益较小。这里所有阶段均按外层迭代平均，部分外层迭代只采样、不更新。

两组均使用 `split` 和同步 profile 计时，因此这不是双卡与原有单卡 `full` 模式的直接速度对照。阶段/整轮计时不含日志、保存和评估；进程墙钟时间包含这些启动和退出开销。虽然稳定段提速，这次仅 240 轮的双卡进程总耗时仍略长，不能据此宣称短任务端到端提速。

首轮 60 轮测试的预热训练时间曾为单卡 213.68 s、双卡 324.03 s；上表为使用已有编译缓存的延长测试。不要把不同缓存状态下的启动/预热差异归因于卡数。

可复现命令：

```bash
uv run -m supermimic.scripts.benchmark_training \
  --motion-file /mnt/dataset/lqi3lr/dyh/supermimic/data/evt/walk1_subject1.npz \
  --output logs/scaling_idle_gpus_extended \
  --gpus 2,3 --cases 1-split 2-split \
  --iterations 240 --warmup 80 \
  --extra --runner.buffer-size 128 --runner.save-interval 1000
```

结果：`logs/scaling_idle_gpus_extended/summary.json`；各 case 目录包含 `command.json`、`environment.json`、`stdout.log`、`config.json` 和 `performance.jsonl`。首轮结果另存于 `logs/scaling_idle_gpus/summary.json`。

## 验收边界与后续效果比较

本轮短程测试验证更新路径与训练流程可用，**没有验证多 seed 的长期收敛质量，也没有得到达到同等评估回报的加速比**。初始阶段奖励不能代替训练效果结论。

完整效果对照应固定动作集、模型初始化 seed 和全局训练规模，至少重复 3 个 seed；定期保存历史 checkpoint，用同一评估 seed、环境数和步数评估。对每个 seed 记录首次达到预先约定回报阈值的 checkpoint 及其 `training_wall_s`，同时比较全局采样量。扩大全局规模需作为独立实验，不默认提高学习率，也不把样本吞吐提高直接换算为收敛加速。

恢复包含模型、优化器、共享采样器和计数；不包含 replay、环境和 RNG 现场。完整训练计时包括启动及此前迭代的 I/O，但不包括保存当前 checkpoint 之后的导出/评估；进程总墙钟时间由 benchmark 单独记录。评估工具报告固定预算的单步平均 reward 和失败率，不是逐动作完成率。

Ruff、修改文件格式检查、Python 编译和 `git diff --check` 已通过。Pyright 尚未全绿：原项目所检查的 6 个文件已有 75 项诊断，当前检查的 7 个文件有 82 项，包含 Torch 类型存根的私有导入诊断；不将其标记为通过。
