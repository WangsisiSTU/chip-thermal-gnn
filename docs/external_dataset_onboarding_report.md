# 外部复杂热数据接入与质量校验（2026-09-16）

## 结论

IC-ThermBench 的代码、S3/S4/S5 数据契约和官方指标实现已固定到提交
`75566fa8a213a5fd61bb89d33a25af258d2a4ae8`；未下载约 44 GB checkpoint。
S5 `manifest.json` 只随约 4.6 GB 数据归档发布，本机访问 Google Drive 超时，因此没有伪造或用推断记录替代它。
本项目已经提供严格 manifest 校验器，拿到文件后可直接验证 5 case × 1000 条及 500/500 adaptation/holdout 顺序。

Multi-topology 的官方 20,350 行 manifest、summary、数据卡和七个 topology 的代表 HDF5
已下载到被 Git 忽略的 `external/Multi-topology-sample/`。全部七个真实样本通过字段、形状、类型、有限值、二值材料掩膜、实例 ID 和换热系数一致性检查。

最重要的研究边界是：官方 Multi-topology split 对 topology 和工况做了良好分层，但**不是版图隔离 split**。
每个 topology 的全部 layout/mask ID 都同时出现在 train、validation 和 test。它可以测同一几何族/版图下的工况泛化，不能单独支撑“未知非规则边界”创新结论。

## IC-ThermBench S3/S4/S5 contract

| Scope | 样本数 | 输入通道 | 官方用途 |
| --- | ---: | --- | --- |
| S3 / `level3` | 15,000 | power, x, y, local thermal conductivity | 材料变化 |
| S4 / `level4` | 15,000 | S3 + ambient, convection h, convection resistance | 材料 + 边界变化 |
| S5 / `level5` | 5,000 | 与 S4 相同 | 未见结构 OOD 与少样本适配 |

原始 HDF5-backed MATLAB 轴为 `input[B,P,Z,Y,X]`、`output[B,Z,Y,X]`，均为
`float32`，温度单位 K；adapter 输出 `input[B,X,Y,Z,P]`、`target[B,X,Y,Z]`。
S3/S4 官方默认不打乱，15,000 样本切为 10,800/1,200/3,000。S5 按 manifest
中的 `case` 分组，每 case 前 500 条为 adaptation pool、后 500 条为共同 holdout；代码不硬编码
case 编号，从而避免上游文档修订造成标签漂移。

指标没有在本项目另写“近似实现”，而是动态加载官方 `utils/metrics.py`：

- RMSE：每样本、每通道计算后平均；
- MAE；pooled R²，并保留 per-sample R² 诊断；
- MaxAE；预测峰值与真实峰值之差；
- Top-50 MAE：真实温度最高 50 个点上的 MAE；
- 官方实现附带 MAPE/PAPE。

## Multi-topology HDF5 映射

| 原字段 | 处理 | 模型语义 |
| --- | --- | --- |
| `power_map` | 原值 float32 | 空间热源/功率密度 |
| `instance_map` | 原始 uint16 单独保留；另取 `>0` | 芯片实例标签与无序 chip mask |
| `ceramic_mask` / `baseplate_mask` / `copper_mask` | 二值校验 | 复杂材料区域 |
| `h_normalized` | 广播为 256×256 平面 | 对流边界工况，等于 `h/10000` |
| `temperature` | 原值 float32 | 256×256 摄氏温度目标 |
| `row_ids` / `max_temps` | 长度与实例集合校验 | 芯片级热点辅助监督/指标 |

稠密 adapter 默认生成六通道：`power_map, chip_mask, ceramic_mask, baseplate_mask,
copper_mask, h_normalized`。没有把实例编号 1/2/… 当连续物理量；原始 `instance_map` 仍可用于
构图、实例 pooling 或 permutation-invariant 编码。

## Manifest 与分层校验

| Topology | Train | Validation | Test | layout ID 数 | train/test 重叠 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mask_3P6P_V1 | 2,586 | 554 | 555 | 77 | 77 |
| Mask_3P6P_V2 | 1,948 | 417 | 419 | 58 | 58 |
| Mask_HB_V1 | 672 | 144 | 144 | 20 | 20 |
| Mask_HB_V2 | 2,083 | 446 | 447 | 62 | 62 |
| Mask_HB_V3-0 | 3,427 | 734 | 735 | 102 | 102 |
| Mask_HB_V3-1 | 2,250 | 482 | 483 | 67 | 67 |
| Mask_HB_V3-2 | 1,276 | 273 | 275 | 38 | 38 |

质量检查结果：20,350 行；0 重复路径；0 路径/标签错位；0 `h_normalized` 错位；
manifest 与 summary 总数完全一致。每个 topology 的三个 split 都覆盖
`Pigbt={150,250,350,450}`、`Pfwd={100,150,200}`、`h={2000,4000,6000,8000}`。

建议论文同时报告两套协议：

1. 保留官方 split，作为可比较的同版图工况基准；
2. 新增按 `topology + mask_id` group 的 layout-disjoint split，必要时进一步留出完整 topology，
   作为非规则边界/结构 OOD 主结果。任何归一化统计都只在训练组拟合。

## 可复现产物

- `src/ic_thermbench_adapter.py`：S3/S4/S5 lazy HDF5、轴映射、固定 split、S5 校验、官方指标入口；
- `src/multitopology_adapter.py`：字段映射、样本校验、代表样本选择、manifest 分层审计；
- `src/validate_external_datasets.py`：生成机读质量报告；
- `configs/ic_thermbench_s5_manifest_contract.yaml`：S5 manifest 验证契约，不含伪造记录；
- `configs/multitopology_sample_subset.yaml`：七个样本的路径、大小和 SHA-256；
- `outputs/external_data_quality/quality_report.json`：本次真实文件检查结果；
- 两个新增测试文件：7 项测试全部通过。

## 清理记录

已删除四个一次性 smoke 数据目录（53.2 MiB）、三个 smoke 输出目录及 pytest 临时目录；
这些文件不可从回收站恢复，但可由仓库脚本重新生成。
经用户明确确认，另删除 15 个旧 benchmark/corrected/FEM-operator 数据目录，回收
1,905,997,606 bytes（约 1.775 GiB）。历史报告和复现代码仍保留，相关数据需要时可按报告命令重新生成。
源码检查没有发现孤立的 `copy/bak/tmp/old` 文件；历史汇总和绘图脚本仍被文档引用，因此继续保留。
