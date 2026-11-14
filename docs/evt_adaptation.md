# EVT 配置说明

已接入 `SuperMimic-GenCoord-EVT` 和 `SuperMimic-MaxCoord-EVT`，原 G1
`src/supermimic/tasks/env_cfgs.py` 保留原样。

## 配置入口

| 文件 | 用途 |
| --- | --- |
| `src/supermimic/robots/evt.py` | EVT 资产路径、逐关节 PD、力矩与标称速度上限、初始位姿、action scale、延迟参数 |
| `src/supermimic/tasks/evt_env_cfgs.py` | 独立 EVT tracking 环境、观测、跟踪刚体、摩擦与 COM 配置 |
| `src/supermimic/tasks/__init__.py` | 根据任务 ID 选择 G1 / EVT 和坐标表示 |
| `src/supermimic/scripts/convert_evt_motions.py` | 将 EVT compact 动作转换为本框架格式，保留片段边界 |
| `src/supermimic/robots/control.py` | 读取实际控制参数，保存与恢复训练参数快照 |
| `src/supermimic/runners/exporter.py` | 将实际参数写入 ONNX 元数据，包括完整精度 JSON |

控制参数采用已确认的逐关节数值。动作目标为默认关节位置加上动作乘以
`0.25`，根节点初始高度为 `0.95 m`，软关节范围系数为 `0.9`。
延迟范围当前为 `0～0` 个物理步，后续可在同一配置中启用。
标称速度上限会保存和导出，本次没有新增硬速度限制或系统辨识随机化。

资产来自 `../allmimic/assets/robots/evt/urdf/evt2_simple.xml` 和其 mesh 文件。
只在内存中移除原地面、原执行器与执行器禁用分组，再建立原生位置 PD。
关节总力矩限制和执行器力矩限制均按新参数设置，避免旧 XML 限制继续截断力矩。
质量、惯量、关节范围和被动阻尼、armature、frictionloss 使用原机械模型。

环境采用基础 tracking，跟踪 14 个刚体，以 pelvis 为 anchor。
物理步长 `5 ms`，控制步长 `20 ms`；足部摩擦沿用 G1 的 `[0.3, 1.2]`
共享随机化，并保留原有 COM 和编码器偏置随机化。Actor、critic 的传感器引用
均已适配 EVT。训练时的重置位姿继续来自参考动作。

网络结构、潜变量维度、优化器和学习算法沿用框架配置。各表示的状态维度为：

| 表示 | full_state | representation | prediction | actor proprio | goal |
| --- | ---: | ---: | ---: | ---: | ---: |
| GenCoord | 71 | 68 | 35 | 93 | 670 |
| MaxCoord | 182 | 214 | 84 | 93 | 670 |

EVT 训练需要显式选择 EVT 动作，不再默认下载 G1 动作。
G1 检查点与 EVT 不兼容；加载时会校验机器人及控制参数。
新训练保存 `robot_control.json` 和检查点内的参数快照，播放先恢复快照再创建环境。
后续修改中央配置用于新的训练，已有模型继续使用训练时保存的参数。

自动 ONNX 导出沿用原框架条件：`actor_obs=proprio` 且动作总帧数不超过 15,000。
全部动作联合训练时，超过该帧数只自动保存检查点；可在训练后使用
`play.py --export-only` 单独导出。

数据转换、正式训练及播放命令见项目 README 的 EVT training 部分。
