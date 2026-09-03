# chip-thermal-gnn：基于图神经网络的芯片封装二维稳态热场代理模型

本项目构建了一个简化的二维芯片封装截面稳态导热有限元（FEM）数据集，并训练
MeshGraphNet 风格的图神经网络（GNN）作为温度场代理模型，展示 **有限元网格图学习、
工程仿真加速与 AI4S** 能力。所有数据由 `scikit-fem` 有限元求解器实际计算生成，
所有指标均由实际运行产生，不依赖任何商业软件、内部代码或数据。

## 1. 项目背景

芯片功耗密度不断提升，热设计成为封装设计的关键环节。传统有限元/有限体积热仿真
精度高但计算成本随网格规模与工况数量线性增长；在**多工况扫描、优化迭代、实时热监控**
等场景中，人们希望用代理模型（surrogate）以极低成本近似替代求解器。图神经网络
直接在有限元网格（天然的图结构）上学习，不依赖规则栅格，是网格类仿真代理的
主流方案之一（DeepMind MeshGraphNets, ICLR 2021）。

本项目实现完整闭环：**几何/工况随机采样 -> FEM 求解生成数据 -> 网格转图 ->
GNN 训练 -> 精度/速度评估 -> 可视化**。

## 2. 物理模型与假设

### 2.1 几何结构

二维矩形截面（宽 20 mm，高 12 mm），自下而上四层：

| 层 | 材料 | 厚度 | 导热系数采样范围 W/(m·K) |
| --- | --- | --- | --- |
| 基板 Substrate | 有机/陶瓷类 | 6 mm | 15 – 60 |
| 散热层 | 铜 | 3 mm | 360 – 400 |
| 界面材料 TIM | 膏体/垫片类 | 1 mm | 1 – 8（OOD: 0.2 – 0.9） |
| Die | 硅 | 2 mm | 130 – 155 |

### 2.2 控制方程与边界条件

稳态热传导方程：

```text
-∇·(k ∇T) = q
```

- **顶部**（Die 上表面）：对流边界 `-k ∂T/∂n = h·(T - T_amb)`，h ∈ [500, 15000] W/(m²·K)（OOD: 100 – 400）；
- **底部**（基板下表面）：恒温 Dirichlet 边界 `T = T_amb`（理想散热地）；
- **左右两侧**：绝热（零通量自然边界条件）。

热源：Die 层内施加体热源，包含背景功率密度 `q_base ∈ [1e5, 5e5] W/m³` 与一个
位置/宽度随机的矩形**热点区域** `q_hot ∈ [5e6, 3e7] W/m³`（OOD: 3.5e7 – 6e7）。

### 2.3 明确的物理简化假设

1. **二维截面近似**：假设垂直纸面方向无限延伸（单位深度），实际封装为三维结构；
2. **层厚经过放大**：真实 TIM 约 0.05–0.1 mm、Die 约 0.5 mm，本项目为保证规则网格
   单元质量与可视化清晰度将各层厚度等比放大，不影响方法论演示；
3. **材料线性且各向同性**：k 与温度无关（线性问题）；
4. **理想界面**：层间无接触热阻（TIM 层本身即代表界面热阻的显式建模）；
5. **边界简化**：底部理想恒温、侧面绝热，仅顶部为对流边界；
6. **关于环境温度 T_amb 的说明**：由于问题线性且所有边界条件均以 T_amb 为参考
   （底部 Dirichlet = T_amb，顶部对流参考 T_amb），温升场 ΔT = T − T_amb 在数学上
   **与 T_amb 的取值无关**（T_amb 只平移绝对温度）。数据集中仍对 T_amb 采样并作为
   特征保留，一方面保持通用框架完整性，另一方面为未来引入温度相关材料属性
   （非线性问题，此时 T_amb 将真实影响 ΔT）预留接口。

## 3. 数据构建

- 共 **360 个样本**：训练 300 / 验证 30 / 测试 30；
- 所有样本共用同一张规则三角形网格（61×37 个节点 = 2257 节点，4320 个三角单元），
  由 `skfem.MeshTri.init_tensor` 生成；
- 每个样本随机采样：各层导热系数、背景/热点功率密度、热点位置与宽度、环境温度、
  顶部对流换热系数；
- **测试集包含 10 个分布外（OOD）样本**，用于验证对未见工况组合的泛化能力：
  - `ood_high_power`：热点功率密度超出训练范围（3.5e7 – 6e7 W/m³ vs 训练最大 3e7）；
  - `ood_poor_tim`：TIM 导热系数低于训练范围（0.2 – 0.9 vs 训练最小 1.0）；
  - `ood_poor_cooling`：对流换热系数低于训练范围（100 – 400 vs 训练最小 500）；
- 原始样本保存在 `data/raw/`（每样本一个 `.npz` + 全局 `mesh.npz` + `metadata.json`，
  metadata 记录采样范围、划分、随机种子与每个样本的完整工况参数）。

### 图数据格式（`data/processed/*.pt`，`torch_geometric.data.Data`）

| 字段 | 内容 |
| --- | --- |
| `x` | 21 维节点特征：归一化坐标、材料 one-hot、局部热源强度、环境温度、边界换热系数、边界类型标志，以及**广播到全部节点的全局工况特征**（各层 k、q_base/q_hot、热点位置/宽度、TIM 与对流热阻） |
| `edge_index` | 由三角单元生成的无向边（双向，去重），13152 条有向边 |
| `edge_attr` | 4 维边特征：归一化相对坐标 (dx, dy)、边长、两端点材料导热系数的调和平均 |
| `y` | 节点温升 ΔT = T − T_amb [K]（预测温升而非绝对温度） |
| `pos` | 节点物理坐标 [m]，用于可视化 |

> **为何广播全局工况特征**：稳态导热是椭圆型方程，任一点温度受全域参数影响。
> 消息传递轮数有限（8 轮）而网格直径约 60 跳，若不广播，远离热源/边界的节点
> 无法感知工况参数，模型精度会受结构性限制。这是稳态（非时间步进）图代理模型
> 的常见做法。

## 4. 模型

### 4.1 主模型：MeshGraphNet 风格 Encode–Process–Decode

- **Node/Edge Encoder**：MLP 将 21 维节点特征、4 维边特征映射到 128 维隐空间；
- **Processor**：8 轮残差消息传递。每轮先由源节点、目标节点与边隐变量经 MLP 更新
  边隐变量（残差），再按目标节点求和聚合并经 MLP 更新节点隐变量（残差）；
  全部 MLP 带 LayerNorm 与 dropout；
- **Decoder**：两层 MLP 输出单节点标准化温升。

### 4.2 基线模型

4 层 GraphSAGE（可配置为 GCNConv），不使用边特征，直接回归节点温升。

### 4.3 训练

- 损失 = 标准化温升的场损失 + 0.1 × 峰值温度误差项（取每图真实温升最大节点处的
  预测误差平方，提升峰值预测能力）；
- 场损失默认采用**逐样本相对形式**（每个样本的误差平方和除以该样本自身信号能量），
  避免损失被高功率大温升样本主导（样本间峰值温升跨越 0.35 – 11 K，动态范围大）；
  可在 `configs/train_config.yaml` 中切换为普通标准化 MSE（`loss_type: zscore_mse`）；
- Adam + ReduceLROnPlateau，梯度裁剪，早停（验证集损失 40 轮不改善），
  保存验证集最优权重，固定随机种子（2024）。

## 5. 运行方式

```bash
pip install -r requirements.txt

# 1. 生成 FEM 数据集（约 360 次求解，单次 ~8 ms，总计 < 1 分钟）
python src/generate_dataset.py --config configs/data_config.yaml --out_dir data/raw

# 2. 网格 -> 图数据
python src/graph_dataset.py --raw_dir data/raw --out_dir data/processed

# 3. 训练（主模型与基线相互独立）
python src/train.py --model meshgraphnet
python src/train.py --model baseline

# 4. 评估（输出指标表到 outputs/metrics/）
python src/evaluate.py

# 5. 可视化（输出全部图片到 outputs/figures/）
python src/visualize.py

# 单独运行某类图：
python src/visualize.py --figures field peak

# 单元测试
python -m pytest tests -q
```

每个脚本均支持命令行独立运行，可用 `-h` 查看全部参数。

## 6. 实验结果

以下指标均由本机实际运行 `evaluate.py` 得到（测试集 30 个样本，含 10 个 OOD），
不得伪造。硬件：NVIDIA GeForce RTX 3060；随机种子 2024。

### 6.1 整体测试集指标对比

| 模型 | 参数量 | MAE [K] | 相对 L2 | 峰值绝对误差 [K] | R² | 单样本推理 [ms] | 批量摊销 [ms/样本] | 平均 FEM [ms] | 批量加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MeshGraphNet | 979,329 | 0.1095 | 0.2294 | 0.3562 | 0.9228 | 10.23 | 6.77 | 8.50 | 1.25× |
| GraphSAGE 基线 | 104,321 | 0.1035 | 0.2231 | 0.3431 | 0.9270 | 2.08 | 0.46 | 8.50 | 18.5× |

> 本例网格仅约 2257 节点，稀疏直接法 FEM 本身极快（~8.5 ms），因此单样本 GNN
> 相对 FEM 的加速不显著；在**批量多工况**扫描场景下，基线批量推理可达约 **18×**。
> 网格规模增大后，FEM 成本上升更快，代理模型加速比通常更明显。

### 6.2 分布内 / 分布外分拆（MeshGraphNet）

| 工况类型 | 样本数 | MAE [K] | 相对 L2 | 峰值绝对误差 [K] |
| --- | ---: | ---: | ---: | ---: |
| 分布内 (id) | 20 | 0.0389 | 0.0736 | 0.0807 |
| OOD 高功率 | 4 | 0.1867 | 0.1694 | 0.4634 |
| OOD 差 TIM | 3 | 0.2074 | 0.3844 | 0.9186 |
| OOD 差对流 | 3 | 0.3798 | 0.4422 | 1.4873 |

分布内精度很高（MAE ≈ 0.04 K）；分布外误差明显增大，尤其是差对流工况，
说明外推仍有挑战，但也验证了测试集确实包含训练集未见的工况组合。

### 6.3 训练摘要

| 模型 | 最佳 epoch | 最佳验证损失 | 训练时长 |
| --- | ---: | ---: | ---: |
| MeshGraphNet | 296 | 0.01404 | ~56 min |
| GraphSAGE 基线 | 265 | 0.00837 | ~4 min |

### 6.4 可视化结果

图片保存在 `outputs/figures/`：

1. `package_structure_mesh.png` — 封装结构与网格示意图  
2. `field_triptych_meshgraphnet_sample*.png` / `field_triptych_baseline_sample*.png` — 真实/预测/误差三联图  
3. `peak_scatter.png` — 测试集峰值温升真值–预测散点图（含 OOD 标注）  
4. `metrics_bar.png` — 两模型指标柱状图  
5. `loss_curves.png` — 训练/验证损失曲线  

指标表保存在 `outputs/metrics/model_comparison.csv` 及各模型 `*_eval_summary.json`。

## 7. 局限性

1. **固定网格与固定几何**：所有样本共用同一张网格，模型未验证跨网格/跨几何泛化
   （MeshGraphNet 架构本身支持变网格，扩展只需在数据生成端随机化网格与层厚）；
2. **二维简化**：未建模三维热扩散与横向热扩展效应；
3. **线性稳态**：无温度相关材料属性、无辐射、无接触热阻；
4. **样本量适中**（360）：更大规模数据可进一步提升精度，尤其是 OOD 外推；
5. **小网格下的加速比有限**：本例网格仅 2257 节点，稀疏直接法求解本身极快（~8 ms），
   GNN 的加速优势主要体现在批量多工况并行推理；网格规模越大（1e5+ 节点），
   FEM 成本增长越快，代理模型加速比越显著。

## 8. 从稳态扩展到瞬态

当前模型学习映射 `工况参数 -> 稳态温度场`。扩展为瞬态版本的路线：

1. **数据端**：FEM 增加热容项 `ρc ∂T/∂t`，对每个工况用隐式时间积分（如向后欧拉）
   生成时间序列温度场（`scikit-fem` 组装质量矩阵即可实现）；
2. **模型端**：改为自回归时间步进模型（MeshGraphNets 原始设定）：输入当前时刻
   温度场 T(t) + 工况特征，输出 ΔT 到下一时刻，即 `T(t+Δt) = T(t) + GNN(T(t), 条件)`；
3. **训练技巧**：训练时对输入注入小噪声以抑制自回归误差累积（rollout drift）；
   评估长时间 rollout 的稳定性与能量守恒；
4. **功耗序列输入**：支持时变热源 q(t)（如动态功耗 trace），实现瞬态热响应预测。

## 9. 相关开源实现（参考）

本项目模型为按论文思想从零实现，未复用外部代码；实现前调研过的可参考的开源实现：

- [DeepMind meshgraphnets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets)（TensorFlow 原始实现）；
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo)：提供 PyTorch/PyG 后端的
  `MeshGraphNet` 及变体（HybridMeshGraphNet、BiStrideMeshGraphNet 等），并附
  vortex shedding 等示例；其公开的预训练权重面向流体等其他物理场景，特征接口与
  本任务不兼容，因此本项目在自建 FEM 数据上从零训练。

## 10. 项目结构

```text
chip-thermal-gnn/
  README.md
  requirements.txt
  configs/
    data_config.yaml       # 几何/材料/工况采样范围、数据划分、随机种子
    train_config.yaml      # 模型结构与训练超参数
  data/
    raw/                   # FEM 原始样本 (.npz) + mesh.npz + metadata.json
    processed/             # PyG 图数据 (train/val/test.pt) + 归一化元数据
  outputs/
    checkpoints/           # 最优模型权重
    figures/               # 全部可视化图片
    metrics/               # 指标表、损失历史、评估摘要
  docs/
    report.tex             # 详细技术说明文档（LaTeX 源）
    report.pdf             # 编译后的 PDF（含全部结果图与表格）
  src/
    fem_solver.py          # scikit-fem 稳态导热求解器
    generate_dataset.py    # 工况采样 + 批量 FEM 求解
    graph_dataset.py       # 网格 -> PyG 图数据
    models.py              # MeshGraphNet + GraphSAGE/GCN 基线
    train.py               # 训练（早停、最优权重、损失历史）
    evaluate.py            # 测试集指标 + 耗时对比
    visualize.py           # 全部图片生成
  tests/                   # 单元测试（FEM 有效性 / 图数据形状 / 模型前向）
```

编译说明文档（需安装 TeX Live，使用 XeLaTeX）：

```bash
cd docs
xelatex -interaction=nonstopmode report.tex
xelatex -interaction=nonstopmode report.tex
```