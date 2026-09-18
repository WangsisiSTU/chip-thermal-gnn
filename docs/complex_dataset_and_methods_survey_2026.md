# 复杂边界与复杂材料热场：数据集和方法调研（2026-09）

## 结论先行

当前数据量并非完全不够，但**物理支持域明显不够**。仓库已有的 3D 数据已经覆盖功率、TIM 导热率、顶部对流和多网格分辨率；最新 operator-learning 协议有 384/64/128 个独立物理工况，并在多个网格上复用同一批工况。主要问题是所有 case 仍来自同一个平直长方体四层封装，材料是分层、线性、各向同性常数，边界条件的类型和施加区域也基本固定。

因此，再生成几千个同一长方体上的随机功率样本，只会改善参数插值，不足以支撑“非规则边界、复杂材料”的创新主张。下一阶段应把任务从

> 固定几何上的参数回归

改为

> 几何、拓扑、材料张量、界面和边界条件共同变化的稳态/瞬态热算子学习。

建议采用两条互补路线：

1. **外部公开基准**用于和社区方法公平比较，首选 IC-ThermBench S2--S5，再用 Multi-topology 和真实红外热图做跨域验证。
2. **本项目自建 3D 非规则四面体数据**用于论文创新，标签由可审计的 FEM/3D-ICE/ATSim 生成，显式加入非规则域、各向异性、温变物性、接触热阻和空间变化边界。

## 1. 现有训练集的真实覆盖范围

| 轴 | 当前覆盖 | 缺口 |
| --- | --- | --- |
| 独立物理 case | operator-learning 协议为 384 train / 64 val / 128 test | 数量可做方法原型，但不足以覆盖几何族和复合材料族 |
| 网格 | 粗、中、细、各向不同分辨率、局部加密 | 仍是同一长方体和同一层界面；不是跨 CAD/跨拓扑 |
| 热源 | 单个矩形热点 + base heat | 缺多芯粒、多热点数量变化、旋转/不规则热源、瞬态功率轨迹 |
| 材料 | 4 层、标量常数导热率 | 缺完整导热张量、温度依赖、空间随机场、TIM 空洞、TSV/微凸点、界面热阻 |
| 边界 | 底部定温、顶部均匀 Robin、侧面绝热 | 缺空间变化 Robin、多面冷却、局部散热器、混合边界、接触/辐射 |
| 标签 | 3D 稳态温升 | 缺瞬态、热流、界面跳变、能量平衡、真实测温校准 |

已有局部加密留出测试只证明了对一种新网格连接关系的敏感性，不能称为新封装几何泛化。当前 GraphSAGE 还不使用边特征，复杂材料和界面信息即使进入数据，也需要新的算子表示才能真正被模型消费。

## 2. 推荐获取的公开数据与工具

完整机器可读目录见 [`configs/external_dataset_catalog.yaml`](../configs/external_dataset_catalog.yaml)。

### 2.1 第一优先级：IC-ThermBench（2026）

- 论文与协议：[IC-ThermBench, arXiv:2608.23977](https://arxiv.org/abs/2608.23977)
- 代码、数据说明和下载入口：[Day333/ThermalBench](https://github.com/Day333/ThermalBench)
- 规模：S2/S3/S4 各 15,000，S5 为 5,000，总计 50,000 个 64×64 稳态样本；S2--S5 数据约 4.6 GB。
- 关键变量：
  - S2：芯粒布局/配置；
  - S3：S2 + 局部导热率场；
  - S4：S3 + 环境温度、换热系数和对流热阻；
  - S5：5 个完全留出的封装系统，测试结构 OOD 和少样本适配。
- 许可：S2--S5 原始张量、固定划分和结果记录为 CC BY 4.0；代码为 MIT；S1 保留上游许可。
- 对本项目的价值：这是最适合回答“布局、材料、边界逐级变复杂后，模型还能否泛化”的外部主基准。其论文还显示 S4 到 S5 的结构 OOD 会使最佳 MAE 从 0.938 K 跳到 15.00 K，而每个新系统仅 10 个标签就能把最佳 MAE 降到 2.60 K，说明**跨封装迁移**比固定封装内插值更值得研究。
- 限制：S2--S5 当前为 `Z=1` 的 2D 顶层温度场，不等同于本项目 3D 四面体体场。

接入建议：保留官方 split 和单位，增加一个独立的规则栅格 adapter；不要把它直接拼进 3D PyG 数据，也不要混报 MAE。它适合做“外部基准分支”，本项目 3D 非规则数据适合做“创新分支”。

### 2.2 第二优先级：Multi-topology Power Module Dataset

- 数据与数据卡：[Multi-topology-Dataset](https://github.com/henkuailederen/Multi-topology-Dataset)
- 规模：20,350 个 HDF5 样本，约 4.67 GB，7 类拓扑/材料掩膜族。
- 字段：256×256 温度、功率密度、芯片实例、铜/陶瓷/底板掩膜、对流换热条件和每芯片峰值。
- 许可：CC BY 4.0。
- 价值：适合验证 topology-aware、多模态输入和多芯片峰值预测；比当前单热点长方体显著复杂。
- 限制：COMSOL 生成的二维顶表面绝对温度，不是 3D 体场；完整仓库较大。

仓库已验证过一个样本格式。下一步应只下载官方 manifest 和按拓扑分层的小子集做 adapter 冒烟，确认训练吞吐后再拉全量。

### 2.3 第三优先级：Therm-FM 数据与 checkpoint（2026）

- 官方仓库：[haiyangxin/Therm-FM](https://github.com/haiyangxin/Therm-FM)
- 论文：[Therm-FM, arXiv:2605.22663](https://arxiv.org/abs/2605.22663)
- 数据：HotSpot 单/四/八核多分辨率稳态与瞬态任务，以及 8 核/32 核工业任务；官方发布 `.mat` 数据、归一化常数和 21M/158M/629M 三档 checkpoint。
- 方法：从 Poseidon PDE foundation model 迁移，并使用 thermal-equivalent 多保真训练，强调跨芯片泛化。
- 价值：适合做低样本 fine-tuning、稳态到瞬态迁移、预训练模型对照。
- 限制：模型和环境较重；必须把“预训练收益”与“架构收益”分开做消融。

### 2.4 真实测量域：UCR Commercial Thermal Map Dataset

- 官方数据仓库：[commercial_thermal_map_dataset](https://github.com/sheldonucr/commercial_thermal_map_dataset)
- 官方实验室说明：[UCR VSCLAB](https://vsclab.ece.ucr.edu/news/2025/01/06/vsclab-introduced-open-sourced-thermal-map-dataset-commercial-cpugputpu-multi-core)
- 覆盖：Intel/AMD CPU、RTX 4060、Google Coral TPU、Snapdragon 等商业器件；输入包含性能计数器/工作负载或时间序列，标签为红外温度图。
- 价值：不是用来替代 FEM 训练，而是用来做 simulation-to-real 校准、峰值位置和热图形态的外部验证。
- 限制：GitHub 只含代表性样本，完整数据需联系作者；传感器、封装开盖和空间配准误差必须单独记录。

### 2.5 非规则域方法基准：RIGNO datasets

- 模型和下载说明：[camlab-ethz/rigno](https://github.com/camlab-ethz/rigno)
- 论文：[RIGNO, arXiv:2501.19205](https://arxiv.org/abs/2501.19205)
- 数据包括 L 形域 Heat、圆域 Wave、带孔 Elasticity 等非结构点云任务。
- 价值：可先验证模型是否真的具备分辨率/离散化/非规则边界泛化，再迁移到芯片热场。
- 限制：多数不是芯片封装任务，不能作为芯片热学精度证据。

### 2.6 应当作为数据生成器，而非现成训练集的资源

| 工具 | 复杂性支持 | 推荐用途 |
| --- | --- | --- |
| [3D-ICE 4.0](https://github.com/esl-epfl/3d-ice) | 异质/各向异性 floorplan、GDS、非均匀网格、稳态/瞬态、不同散热器；GPL-3.0 | 批量生成贴近 2.5D/3D chiplet 的低/中保真标签 |
| [ATSim 2.0](https://github.com/PKU-IDEA/ATSim_pub) | 2D--3.5D、非线性和正交各向异性材料、温度反馈功率、TSV、稳态/瞬态、自适应多尺度网格 | 复杂材料和工业封装的高价值生成器；目前公开的是 Linux 可执行文件和案例，应先审查再使用 |
| [MFIT](https://github.com/AlishKanani/MFIT) | 异质 chiplet、非均匀节点、各向异性材料、RC/DSS、稳态/瞬态 | 多保真 RC/FEM 对齐、长功率轨迹；GPL-3.0，Windows 运行受限 |
| [3D-IC benchmark testcases](https://arxiv.org/abs/2608.25155) | 多种 3D-IC 物理设计案例 | 提供几何/布局种子，再送入热求解器；本身不等于热场标签集 |

第三方 GPL 工具应隔离运行，只保存清晰授权的输入、输出和必要元数据；不要复制其源码进本项目核心模块。

## 3. 最新方法中真正与创新点相关的做法

### 3.1 用“渐进物理支持域”定义数据，而不是只报样本数

[IC-ThermBench](https://arxiv.org/abs/2608.23977) 把布局、材料、边界、结构 OOD 分成 S2--S5。这比随机地把所有参数混成一个训练集更能回答模型究竟学会了什么。本项目应采用同样的逐级 scope：

- C0：固定几何、变化工况；
- C1：变化布局与热点；
- C2：C1 + 复杂材料和界面；
- C3：C2 + 空间变化/多类型边界；
- C4：新几何族、新拓扑、新网格的结构 OOD；
- C5：材料 + 边界 + 几何同时 OOD 的 compound shift。

所有 split 必须按 `geometry_id/material_family_id/physical_case_id` 分组，不能把同一个物理 case 的不同网格版本分别放到训练和测试。

### 3.2 非规则几何：区域图、几何编码和域分解

- [RIGNO（NeurIPS 2025）](https://arxiv.org/abs/2501.19205)：物理节点先局部聚合到稀疏 regional graph，在多尺度区域边上做消息传递，再投影回查询点；重点是任意点云和分辨率鲁棒性。它比在全部四面体节点上堆深 attention 更适合当前问题。
- [GINO](https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/gino.py)：不规则输入/输出点通过 GNO 映射到规则潜在网格，在潜在网格上用 FNO 建模全局相互作用，再查询回原网格。可把 SDF、材料占据率和边界距离作为几何通道。
- [EqGINO（ICML 2026）](https://arxiv.org/abs/2606.03260)：为 3D 不规则几何加入离散旋转/反射等变性，适合测试封装旋转、芯粒方向变化是否仍保持物理一致。
- [Schwarz Neural Inference（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/710445227fa8c1b6a9ceada902dd4741-Abstract-Conference.html)：把任意域分为局部子域，用可复用局部神经算子求解，再通过 Schwarz 迭代拼接；这对层状封装、chiplet、散热器等天然子域特别契合。
- [PI-GANO](https://arxiv.org/abs/2408.01600)：几何编码器 + 物理约束，目标是同时泛化 PDE 参数和域几何，可作为少标签/无标签消融。

建议主模型首先尝试“RIGNO 式区域图 + FEM 弱形式”，而不是直接换成更大的 Transformer。域分解可作为第二条更鲜明但实现量更大的路线。

### 3.3 复杂边界：边界应是独立对象，不是两个 node flag

当前 `is_top/is_bottom + h_top` 无法表达局部散热器、不同面的 Robin/Neumann/Dirichlet 混合边界。相关方法包括：

- [BENO（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/218ca0d92e6ed8f9db00621e103dc70c-Abstract-Conference.html)：分别编码内部源项和边界值，并用 Transformer 编码全局边界几何；与稳态热传导的椭圆型问题高度匹配。
- [Hard Constraint Projection for chiplet thermal analysis（2026）](https://github.com/LIMIGwenshao/HCP-enhanced-DeepONet)：把边界条件硬投影进输出，而不是只靠 penalty；适合作为精确 Dirichlet/部分 Robin 的对照，但其公开任务仍是规则 2.5D 域。

本项目应新增**边界面图**：每个三角面保存面积、法向、BC 类型、BC 值/系数、所属冷却区域，并通过 node--face 二部边与体网格通信。这样边界表示才与 FEM 的表面积分同构。

### 3.4 复杂材料：从 material ID 升级为算子系数

复杂材料不应仅扩大四个标量 `k` 的范围，而应至少覆盖：

1. 对称正定导热张量 `K(x,T)`，含方向和旋转；
2. 材料内部空间变化/随机场；
3. 接触热阻 `R_contact`，允许界面温度跳变；
4. TIM 空洞、铜填充、TSV/微凸点造成的多尺度异质性；
5. `k(T)`、`c_p(T)` 和必要时辐射 `epsilon` 的非线性。

图表示应把单元而非节点作为主要物理载体：每个四面体保存 `K_xx,K_yy,K_zz,K_xy,K_xz,K_yz`、体积、材料族和温变参数；每个材料界面面保存两侧材料、法向和接触导热系数。消息传递中使用几何一致的 `B_e^T K_e B_e V_e` 或其低维编码，而不是仅用端点材料导热率的谐均值。

### 3.5 用弱形式/能量残差替代纯节点误差

- [Finite-element physics-informed operator learning](https://arxiv.org/abs/2405.12465) 用 FEM 插值与离散弱形式处理任意域，避免在复杂界面上直接用高阶自动微分。
- [WINO（2026 preprint）](https://arxiv.org/abs/2605.24651) 把 level-set/非拟合 FEM 与弱形式神经算子结合，说明“几何表示 + 弱形式 + hard BC”是当前活跃方向，但其任务为超弹性，不能直接照搬结论。

对本项目最可落地的是在已保存的四面体几何上计算：

```text
r = K_FEM(K, geometry, contact, Robin) @ T_hat - b(q, BC)
L_phys = ||r_free||^2 / (||b_free||^2 + eps)
L_energy = |T_hat^T K T_hat - T_hat^T b|
```

并分别报告体域残差、边界热流误差、界面通量守恒和全局能量平衡。复杂材料下，界面温度跳变是否正确比单纯全场 MAE 更关键。

### 3.6 多保真与主动采样，而不是全量高保真穷举

- [SAU-FNO（2025）](https://arxiv.org/abs/2510.15968) 使用低分辨率预训练 + 少量高分辨率 fine-tuning；论文实验中使用 4,000 个低保真与 1,000 个高保真 case。
- [Therm-FM（2026）](https://github.com/haiyangxin/Therm-FM) 使用 PDE foundation model 适配和 thermal-equivalent 多保真数据。
- [Sorting Krylov Recycling（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bf331c87e29f473b610336f00fe1cb51-Abstract-Conference.html) 针对相似线性系统排序并复用 Krylov 子空间，可减少批量 PDE 标签生成的重复计算。
- [Physics-agnostic geometry pretraining（ICLR 2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html) 先用大量无标签几何做 occupancy 重构，再用少量物理标签训练算子。

建议把预算放在“更多几何、材料族和边界族”，而不是每个族内密集采样。低保真生成器可用 3D-ICE/MFIT，中高保真用本地四面体 FEM 或 ATSim，少量真实热图只用于校准。

## 4. 建议的本项目创新主线

### 主张

> 面向非规则 3D chiplet 封装和异质/各向异性复合材料，构建边界面--体单元双图，以 FEM 弱形式和界面热流守恒约束区域图神经算子，实现跨几何、跨网格、跨材料与跨边界条件的热场泛化。

这个主张比“在固定规则封装上把 GraphSAGE 换成 Transformer”更有辨识度，也能自然形成四个可验证贡献：

1. 新数据协议：非规则几何 + 复杂材料 + 混合边界的 3D 数据；
2. 新表示：node/tet/interface-face/boundary-face 多实体图；
3. 新模型：区域图全局传播 + 局部离散算子消息；
4. 新约束与评估：弱残差、界面通量、边界通量、能量平衡和结构 OOD。

不建议把“用了 Transolver/GINO/RIGNO”本身当创新；创新应落在芯片热问题特有的**界面、边界面和离散热算子一致性**上。

## 5. 推荐的数据生成协议

可执行前的设计规范见 [`configs/complex_thermal_dataset_protocol_v1.yaml`](../configs/complex_thermal_dataset_protocol_v1.yaml)。建议首轮规模：

| 层级 | 目的 | 建议规模 |
| --- | --- | ---: |
| 低保真 | 覆盖几何/材料/边界组合 | 8,000--12,000 case |
| 中保真 | 本项目四面体 FEM，训练主干 | 1,500--2,500 case |
| 高保真 | 加密网格/ATSim/独立求解器交叉校验 | 300--500 case |
| 真实测量 | 校准与 domain-gap 报告 | 50--200 对齐 case |

首轮不必一步到位。建议先做一个可发表的最小闭环：

- 40 个训练几何 + 10 个验证几何 + 15 个完全留出测试几何；
- 每个几何 16--32 个物理工况；
- 3 类非规则外形、3 类多芯粒拓扑、4 类材料复杂度、4 类边界模式；
- 每个 case 至少两个独立 mesh seed；
- 结构 OOD、材料 OOD、边界 OOD、compound OOD 分开报告。

## 6. 评价门槛

除节点 MAE/relative L2/peak error 外，至少增加：

- 体积加权 MAE，避免细网格区域支配指标；
- 芯粒峰值位置误差和热点 IoU；
- 材料界面两侧温度跳变误差；
- 法向热流连续性误差；
- Robin/Neumann 边界热流误差；
- 全局输入功率与散热功率不平衡；
- 每个 geometry family 的均值、P95 和最坏 case；
- 跨网格、跨几何、跨材料、跨边界和 compound OOD；
- 至少 3 个训练种子，并按 geometry family 而非节点做 bootstrap。

最终对比应至少包含：GraphSAGE、当前 MGN+attention、纯 MGN、RIGNO 式区域图、区域图 + 弱形式、区域图 + 弱形式 + 界面/边界面表示。只有逐项消融才能证明创新来自物理表示，而非参数量。

## 7. 立即执行顺序

1. 下载 IC-ThermBench **代码和 manifest**，先不拉 44 GB checkpoint；确认 S3/S4/S5 adapter 和指标。
2. 从 Multi-topology 每个 topology 抽取少量样本，完成 HDF5 字段映射和分层 split 校验。
3. 扩展 FEM 数据结构：tet material tensor、interface faces、boundary faces、BC type/value、contact conductance。
4. 先实现 10--20 个非规则几何的 smoke dataset，做 MMS/能量守恒/网格收敛。
5. 再生成完整多保真数据；外部公开数据和自建 3D 数据分别训练、分别报告。
6. 用 UCR 真实热图做最后的定性/少样本校准，避免把合成 FEM 精度写成真实芯片精度。

