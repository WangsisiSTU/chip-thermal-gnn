# chip-thermal-gnn 统一技术说明

**覆盖范围：**二维三角形 FEM--PyG--GNN 流程、三维四面体 FEM--PyG--GNN 流程、数值验证、模型评估、可视化及外部参考。  
**状态：**截至 2026-09-02 的主技术文档。文中的“实测”均能追溯到 `outputs/` 中的摘要、指标或图像；“设计/待复测”不应被理解为已经运行。

## 1. 目标、边界与总管线

本项目面向分层芯片封装的**稳态温度场代理建模**：有限元（FEM）提供节点温度真值，图神经网络（GNN）学习从工况、材料和网格拓扑到节点温升 \(\Delta T\) 的映射。二维版本用于建立小成本流程，三维版本保留了二维代码和数据契约，并将网格、热点与图数据扩展到体域。

```text
参数化封装与热工况
    -> P1 FEM（2D 三角形 / 3D 四面体）
    -> 节点温度、材料、边界标志、单元连通性
    -> PyTorch Geometric 图
    -> GraphSAGE 可复现基线 / MeshGraphNet 风格网络
    -> 场误差、峰值误差、OOD、成本和切面云图
```

本项目是研究性基线，不是签核级封装热仿真器。其模型假设为材料线性且各向同性、层间理想接触、底部恒温、顶部对流、其余外表面绝热；未包含温度相关物性、辐射、显式接触热阻、几何公差、翘曲或瞬态热容。

## 2. 物理模型与 FEM 理论

### 2.1 控制方程与边界

在 \(d\in\{2,3\}\) 维域 \(\Omega\) 上求解稳态热传导：

\[
-\nabla\!\cdot\left(k(\mathbf{x})\nabla T(\mathbf{x})\right)=q(\mathbf{x}),\qquad \mathbf{x}\in\Omega,
\]

其中 \(k\) 是分层导热系数，\(q\) 是体热源密度。边界分为：

\[
\begin{aligned}
T &= T_{amb}, && \Gamma_D\quad\text{（底部 Dirichlet）},\\
-k\nabla T\!\cdot\mathbf n &= h_{top}(T-T_{amb}), && \Gamma_R\quad\text{（顶部 Robin/对流）},\\
-k\nabla T\!\cdot\mathbf n &=0, && \Gamma_N\quad\text{（侧面/其余面绝热）}.
\end{aligned}
\]

令测试空间 \(V_0=\{v\in H^1(\Omega):v|_{\Gamma_D}=0\}\)。相应弱式为：求满足 Dirichlet 条件的 \(T\)，使对任意 \(v\in V_0\)，

\[
\int_\Omega k\nabla T\!\cdot\nabla v\,d\Omega+
\int_{\Gamma_R}h_{top}Tv\,d\Gamma=
\int_\Omega qv\,d\Omega+
\int_{\Gamma_R}h_{top}T_{amb}v\,d\Gamma.
\]

因此温升标签为 \(\Delta T=T-T_{amb}\)。在当前线性模型中，改变 \(T_{amb}\) 只平移绝对温度，理论上不改变 \(\Delta T\)；仍保留该输入，为将来的温度相关物性扩展留接口。

### 2.2 P1 单元离散

二维每个单元为三节点线性三角形，三维每个单元为四节点线性四面体。对单元 \(e\)，用线性形函数近似

\[
T_h|_e=\sum_{i=1}^{n_e}N_iT_i,\qquad n_e=3\;(2D),\;4\;(3D),
\]

并装配

\[
K_{ij}^{(e)}=\int_{\Omega_e}k\nabla N_i\!\cdot\nabla N_j\,d\Omega,
\qquad f_i^{(e)}=\int_{\Omega_e}qN_i\,d\Omega.
\]

在三维顶面三角形 \(s\) 上，Robin 项的局部矩阵为

\[
K_R^{(s)}=\frac{h_{top}A_s}{12}
\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix},
\]

右端增加 \(h_{top}T_{amb}A_s/3\)。代码使用 scikit-fem 的稀疏双线性/线性形式组装和约束消元；该库适合本项目的三角形、四面体与稀疏 FEM 工作流。[1]

### 2.3 几何、材料与热点

两种版本均由下至上使用四层：Substrate、Cu、TIM、Die。二维截面为 \(W\times H=20\,\mathrm{mm}\times12\,\mathrm{mm}\)，三维体域额外具有 \(D=16\,\mathrm{mm}\) 深度。层厚分别为 6、3、1、2 mm。

| 参数 | 分布内范围 | OOD 范围（仅测试集部分样本） |
| --- | ---: | ---: |
| \(k_{die}\) | 130--155 W/(m·K) | — |
| \(k_{TIM}\) | 1--8 W/(m·K) | 0.2--0.9 W/(m·K) |
| \(k_{Cu}\) | 360--400 W/(m·K) | — |
| \(k_{sub}\) | 15--60 W/(m·K) | — |
| \(q_{base}\) | \(1\times10^5\)--\(5\times10^5\) W/m³ | — |
| \(q_{hot}\) | \(5\times10^6\)--\(3\times10^7\) W/m³ | \(3.5\times10^7\)--\(6\times10^7\) W/m³ |
| \(h_{top}\) | 500--15,000 W/(m²·K) | 100--400 W/(m²·K) |

二维热点是在 Die 内沿 \(x\) 的矩形区；三维热点是 Die 内 \(x-z\) 平面的矩形，并沿 Die 厚度方向 \(y\) 延展：

\[
q(x,y,z)=\begin{cases}
q_{hot}, & y>y_{TIM},\;x\in[x_c-w_x/2,x_c+w_x/2],\;z\in[z_c-w_z/2,z_c+w_z/2],\\
q_{base}, & y>y_{TIM}\ \text{且不在热点内},\\
0, & \text{其他层}.
\end{cases}
\]

这一定义使 3D 版本真正学习横向 \(x-z\) 热扩散，而非将二维图像复制为“伪三维”。

## 3. 数据与图表示

### 3.1 划分、可重复性与防泄漏

二维和完整三维数据集均采用固定种子 2024，划分为训练/验证/测试 = 300/30/30；测试集含 20 个分布内样本与 10 个 OOD 样本。OOD 分为高功率、低 TIM 导热和差对流三类。所有归一化参考量取自**声明的参数范围**（包含 OOD 范围），温升的均值和标准差仅由训练拆分计算，避免从测试标签泄漏。

每个原始样本保存节点坐标、温度、材料编号、节点热源、边界标志及工况元数据；`mesh.npz` 保存共享网格。处理后每个样本是一个 `torch_geometric.data.Data` 图对象。PyG 为不同规模图的批处理和图算子提供底层实现。[2]

| 属性 | 2D | 3D |
| --- | ---: | ---: |
| FEM 单元 | P1 三角形 | P1 四面体 |
| 规则网格 | 61×37 | 13×13×11 |
| 节点数 | 2,257 | 1,859 |
| 单元数 | 4,320 三角形 | 8,640 四面体 |
| 有向图边 | 13,152 | 22,532 |
| 节点特征 | 21 维 | 24 维 |
| 边特征 | 4 维 | 5 维 |
| 原始/处理目录 | `data/raw`, `data/processed` | `data/raw_3d_full`, `data/processed_3d_full` |

三维虽节点略少，但每个四面体贡献六条无向边，因而边数比二维高约 71%。边由单元顶点对去重后双向写入 `edge_index`，无自环。

### 3.2 特征设计

节点特征由局部物理量与广播的全局工况组成：

| 类别 | 2D | 3D |
| --- | --- | --- |
| 坐标 | \(x/W,y/H\) | \(x/W,y/H,z/D\) |
| 材料 | 4 维 one-hot | 4 维 one-hot |
| 局部场 | \(q\)、\(T_{amb}\)、顶面对流系数、顶/底边界标志 | 同左 |
| 热点全局量 | \(q_{base},q_{hot},x_c,w_x\) | \(q_{base},q_{hot},x_c,z_c,w_x,w_z\) |
| 材料/边界全局量 | 各层 \(k\)、\(1/k_{TIM}\)、\(1/h_{top}\) | 同左 |
| 边 | \(\Delta x,\Delta y\)、边长、端点调和平均 \(k\) | 另加 \(\Delta z\) |

调和平均 \(k_{ij}=2k_ik_j/(k_i+k_j)\) 是串联导热界面等效导热的自然局部描述。\(1/k_{TIM}\) 与 \(1/h_{top}\) 以热阻形式编码，帮助网络表达温升对低导热界面和弱对流的单调敏感性。

全局工况向所有节点广播是有意设计：稳态导热是全局耦合的椭圆型问题，有限消息传递轮数无法保证把远端热源或边界信息传遍图。这里广播的是已知设计条件而非温度标签，因此不是标签泄漏；部署时必须同样提供这些可观测参数。

## 4. 网络与训练方案

### 4.1 GraphSAGE 基线

GraphSAGE 通过邻域聚合构造归纳式节点表征，[3] 其简化的均值聚合形式可写为

\[
\mathbf h_v^{(\ell+1)}=\sigma\left(
W^{(\ell)}\left[\mathbf h_v^{(\ell)}\;\Vert\;
\operatorname{mean}_{u\in\mathcal N(v)}\mathbf h_u^{(\ell)}\right]\right).
\]

本项目的 GraphSAGE 直接回归每个节点的标准化温升，不使用 `edge_attr`。它是重要的工程基线：参数少、训练稳定，并能检验复杂边网络是否真正带来增益。

### 4.2 MeshGraphNet 风格网络

本网格网络参考 Encode--Process--Decode 的 MeshGraphNets 思想，[4] 但以本仓库代码独立实现并重新训练，未复制第三方模型权重或代码。节点与边编码器先映射输入；第 \(\ell\) 次消息传递为

\[
\begin{aligned}
\mathbf e_{ij}^{(\ell+1)}&=\mathbf e_{ij}^{(\ell)}+
\phi_e\left([\mathbf h_i^{(\ell)},\mathbf h_j^{(\ell)},\mathbf e_{ij}^{(\ell)}]\right),\\
\mathbf m_i^{(\ell+1)}&=\sum_{j:(j,i)\in E}\mathbf e_{ji}^{(\ell+1)},\\
\mathbf h_i^{(\ell+1)}&=\mathbf h_i^{(\ell)}+
\phi_v\left([\mathbf h_i^{(\ell)},\mathbf m_i^{(\ell+1)}]\right),\\
\widehat{\Delta T_i}&=\psi(\mathbf h_i^{(L)}).
\end{aligned}
\]

MLP 使用残差、LayerNorm 和 dropout。该模型可使用边的几何增量与等效导热系数，代价是每条边维护隐状态，三维中内存和时间对消息传递次数更敏感。

| 配置 | 2D GraphSAGE | 2D MGN | 3D GraphSAGE（主基线） | 3D MGN（紧凑探针） |
| --- | ---: | ---: | ---: | ---: |
| 层/消息传递步 | 4 层 | 8 步 | 3 层 | 2 步 |
| 隐藏维度 | 128 | 128 | 64 | 64 |
| 参数量 | 104,321 | 979,329 | 19,713 | 73,153 |
| batch size | 8 | 8 | 4 | 8 |
| 训练轮上限 | 500 | 500 | 120 | 12 |

三维 MGN 故意没有直接沿用二维的 8 步、128 维配置：在 22,532 条有向边上那样做会过早放大边状态的计算成本。现有 2 步试验只用于建立成本/学习曲线锚点，不应解读为充分调优的 MGN。

### 4.3 损失、优化与早停

令 \(z\) 为训练温升统计量标准化后的目标，\(\hat z\) 为预测。当前默认场损失为逐样本相对能量：

\[
\mathcal L_{field}=\frac1B\sum_{b=1}^B
\frac{\sum_{i\in b}(\hat z_i-z_i)^2}
{\sum_{i\in b}(z_i-\bar z_b)^2+10^{-4}}.
\]

再在真实峰值节点处加入辅助损失：

\[
\mathcal L=\mathcal L_{field}+0.1\,\frac1B\sum_b
\left(\hat z_{i_b^*}-z_{i_b^*}\right)^2,
\qquad i_b^*=\arg\max_{i\in b}z_i.
\]

该设置避免大温升样本支配损失，并显式关心热设计常用的峰值温度。优化器为 Adam（初始学习率 \(10^{-3}\)）、`ReduceLROnPlateau`、梯度裁剪 1.0、固定种子和验证集早停；三维运行限定 4 个 CPU 线程，避免图聚合的线程过度订阅。

## 5. 验证与测试方案

### 5.1 分层验证原则

| 层级 | 验证对象 | 具体检查 | 状态 |
| --- | --- | --- |
| 2D FEM | `test_fem_solver.py` | 有限温度、底部恒温、热点在 Die、功率增大时峰值升高、分层编号 | 测试代码已存在；本轮未重新执行 |
| 2D 图数据 | `test_graph_dataset.py` | 三角形转双向去重边、无自环、21/4 维特征、处理文件形状 | 测试代码已存在；本轮未重新执行 |
| 2D 网络 | `test_models.py` | MGN 前后向梯度、GraphSAGE/GCN 前向形状 | 测试代码已存在；本轮未重新执行 |
| 3D FEM/图 | `test_fem_solver_3d.py`、`test_graph_dataset_3d.py` | 四面体连通、边界、x-z 热点、双向边、24/5 维特征 | **已实测通过** |
| 3D 数值收敛 | `test_mms_3d.py` | 解析制造解下网格加密误差单调下降、收敛阶阈值 | **已实测通过** |

为避免证据越界，二维单元测试在本仓库中完整存在，但没有保留一份与本报告同日的 `pytest` 终端日志；本报告不把它们虚报为“本轮已通过”。二维训练和评估的历史产物仍保留在 `outputs/metrics/`。重新确认二维逻辑时应执行 `python -m pytest tests -q`，但运行前需按项目约定获得授权。

### 5.2 三维真实测试与 MMS

实际执行的三维命令为：

```powershell
.\.venv-3d\Scripts\python.exe -m pytest tests/test_fem_solver_3d.py tests/test_graph_dataset_3d.py tests/test_mms_3d.py -q
```

结果为 **6 passed**，另有两条不影响计算的 PyTorch `torch.jit.script` 弃用警告。

物理直觉测试只能说明实现没有明显违反边界或热源规律，不能证明有限元收敛。因此增加独立制造解（MMS）：在单位立方体取

\[
T^*(x,y,z)=\sin(\pi x)\sin(\pi y)\sin(\pi z),\qquad q=3\pi^2T^*,
\]

并在全部边界施加与解析解相符的零 Dirichlet 条件。该思路与 MOOSE 的热传导验证教程一致，[5] 但本项目的求解和测试均独立运行。结果如下：

| 每轴节点数 | 节点数 | 四面体数 | 节点 RMSE | 最大绝对误差 |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 125 | 384 | 2.4351e-2 | 9.5072e-2 |
| 9 | 729 | 3,072 | 7.6178e-3 | 2.5201e-2 |
| 13 | 2,197 | 10,368 | 3.6276e-3 | 1.1324e-2 |

观测收敛阶为 **1.677、1.830**。这证明在光滑解析解上，四面体实现随网格加密稳定收敛；不代表简化封装模型已被实验测温验证。

### 5.3 学习模型评估

测试集上的指标定义为

\[
\mathrm{MAE}=\frac1N\sum_i|\widehat{\Delta T}_i-\Delta T_i|,
\quad \mathrm{RelL2}=\frac{\|\widehat{\Delta T}-\Delta T\|_2}{\|\Delta T\|_2},
\quad E_{peak}=|\max_i\widehat{\Delta T}_i-\max_i\Delta T_i|.
\]

同时报告 \(R^2\)、按 OOD 工况分组的误差、单样本与批处理推理时间、FEM 时间和进程 RSS。速度必须在相同硬件、线程、batch、计时定义下才能横比；CPU/GPU 或不同网络宽度的结果不可混算。

## 6. 已有实验结果

### 6.1 二维：历史训练与当前保存评估

二维训练摘要来自 `outputs/metrics/*_train_summary.json`：GraphSAGE 最佳第 265 epoch，共训练 305 epoch（231.57 s）；MGN 最佳第 296 epoch，共训练 336 epoch（3,394.16 s）。二维 GraphSAGE 的当前保存评估摘要为：

| 指标 | 2D GraphSAGE |
| --- | ---: |
| MAE | 0.1035 K |
| 相对 \(L_2\) | 0.2231 |
| \(R^2\) | 0.9270 |
| 峰值绝对误差 | 0.3431 K |
| ID / 高功率 / 差 TIM / 差对流 MAE | 0.0308 / 0.1166 / 0.2156 / 0.4589 K |
| 平均 FEM | 8.50 ms |
| 单样本推理 | 54.94 ms |
| 批处理推理 | 56.01 ms/样本 |
| 评估 RSS | 520.09 MiB |

该摘要的运行设备为 CPU。仓库早期 PDF 中的二维 MGN 速度来自 RTX 3060 历史运行，不能与上表 CPU 速度或三维 CPU 速度直接比较；它保留为历史参考而非本报告的跨维速度结论。

### 6.2 三维：可信 GraphSAGE 基线与 MGN 探针

完整 3D 数据集实际生成 300/30/30 个样本；原始 FEM 生成用时 39.41 s，平均单样本 FEM 为 112.25 ms。以下全部为 CPU 实测：

| 指标 | 3D GraphSAGE | 3D 紧凑 MGN |
| --- | ---: | ---: |
| 最佳 epoch / 总 epoch | 69 / 94 | 11 / 12 |
| 训练时长 | 439.29 s | 343.80 s |
| 训练 RSS 峰值 | 597.64 MiB | 887.57 MiB |
| MAE | **0.0835 K** | 0.1433 K |
| 相对 \(L_2\) | **0.4945** | 0.7079 |
| \(R^2\) | **0.6509** | 0.2846 |
| 峰值绝对误差 | **0.1206 K** | 0.1529 K |
| 加权 OOD MAE | **0.1866 K** | 0.3060 K |
| 单样本推理 | **7.43 ms** | 19.13 ms |
| 批处理推理 | **6.43 ms/样本** | 20.98 ms/样本 |
| 平均 FEM | 112.25 ms | 112.25 ms |

GraphSAGE 的 OOD MAE 分别为 ID 0.0319 K、高功率 0.0489 K、差 TIM 0.0283 K、差对流 0.5283 K。结论不是“三维已完全解决”：顶面对流外推是当前最明显短板。现阶段应冻结 GraphSAGE 为可信 3D 基线，先增强差散热覆盖和边界编码，再评估更深 MGN。

### 6.3 2D 与 3D：可比与不可比之处

| 指标 | 2D GraphSAGE | 3D GraphSAGE |
| --- | ---: | ---: |
| 节点 / 有向边 | 2,257 / 13,152 | 1,859 / 22,532 |
| 参数量 | 104,321 | 19,713 |
| MAE | 0.1035 K | **0.0835 K** |
| 峰值误差 | 0.3431 K | **0.1206 K** |
| 加权 OOD MAE | 0.2490 K | **0.1866 K** |
| 平均 FEM | 8.50 ms | 112.25 ms |
| 评估 RSS | 520.09 MiB | 600.84 MiB |

三维结果在当前数据上具有更低的场、峰值和 OOD 误差，但不能把这直接归为“维度优势”：两种网络的宽度与深度不同，且二维/三维物理场的难度不等。稳健结论是，三维带来了更丰富的热点横向扩散和更高的 FEM/图边成本；是否值得取决于目标封装是否确实需要三维效应。

## 7. 图像证据与解释

二维图像位于 `outputs/figures/`，包括结构与网格、真值/预测/误差三联图、峰值散点、指标柱图和损失曲线。三维图像按行给出 die 中心 x-z、中心 y-z、中心 x-y 切面，按列给出 FEM、预测和误差。

中位 ID 样本 #19：FEM 峰值 0.708 K、预测峰值 0.783 K；热点位置和厚度方向扩散得到重建。

![3D ID 样本的 FEM、预测与误差切面](../outputs/3d/figures/slice_clouds_baseline_sample19_id.png)

差散热 OOD 样本 #27：FEM 峰值 2.722 K、预测峰值 3.437 K；三个切面都显示大范围高估。这与该工况 0.5283 K MAE 一致，是模型局限的可视证据而非被删除的异常点。

![3D 差散热 OOD 样本的 FEM、预测与误差切面](../outputs/3d/figures/slice_clouds_baseline_sample27_ood_poor_cooling.png)

## 8. 开源方案、外部数据与引用边界

| 资源 | 在本项目中的作用 | 使用边界 |
| --- | --- | --- |
| scikit-fem [1] | 实际 FEM 组装、三角形/四面体网格 | 直接依赖；标签由其实际求解生成 |
| PyTorch / PyG [2] | 训练、张量、图 `Data` 和卷积层 | 直接依赖；网络权重均从零训练 |
| GraphSAGE [3] | 归纳图基线的理论来源 | 本地实现/调用对应算子；不使用外部权重 |
| MeshGraphNets [4] | Encode--Process--Decode 和消息传递设计参考 | 本地独立实现；公开流体/结构权重不用于芯片热任务 |
| MOOSE 热传导教程 [5] | MMS 验证思路 | 仅验证方法参考；3D MMS 独立实现 |
| MFIT [6] | 2.5D/3D 多芯粒 RC/DSS 与输入案例参考 | 已下载至 `external/MFIT`；GPL-3.0，未并入源码；Windows 不支持其 Linux `.so`，故未运行数值对比 |
| Multi-topology Dataset [7] | 跨拓扑功率模块的温度、功率和材料掩膜参考 | CC BY 4.0；已读取一个公开 HDF5 样本，不参与训练；它是二维顶表面绝对温度，不能与本项目 3D 节点温升直接算 MAE |

已运行的 Multi-topology 参考样本 `Mask_HB_V3-0/141_Pigbt450_Pfwd200_h6000` 含 `(256,256)` 温度场、功率图、芯片实例图和三种材料掩膜；温度范围 46.3787--155.626 °C，六个芯片峰值 145.440--155.626 °C，`h=6000`。其检查图仅用于数据格式和可视化对照：

![Multi-topology 公开参考样本](../outputs/3d/figures/reference_multi_topology_sample.png)

未来若与 MFIT 或其他外部求解器做数值对比，必须先对齐几何、单位、材料、热源、底部边界、顶部换热、绝对温度/温升定义、网格收敛准则和评价掩膜；否则只应报告定性或数据格式对比。

## 9. 可复现操作

环境应使用系统受信任证书存储安装依赖，**不得**通过关闭证书校验绕过 TLS：

```powershell
python -m venv .venv-3d
.\.venv-3d\Scripts\python.exe -m pip install --use-feature=truststore -r requirements.txt
```

二维流程：

```powershell
python src/generate_dataset.py --config configs/data_config.yaml --out_dir data/raw
python src/graph_dataset.py --raw_dir data/raw --out_dir data/processed
python src/train.py --model baseline --data_dir data/processed --config configs/train_config.yaml --out_dir outputs
python src/evaluate.py --data_dir data/processed --ckpt_dir outputs/checkpoints --out_dir outputs
python src/visualize.py
python -m pytest tests -q
```

三维流程：

```powershell
.\.venv-3d\Scripts\python.exe -m pytest tests/test_fem_solver_3d.py tests/test_graph_dataset_3d.py tests/test_mms_3d.py -q
.\.venv-3d\Scripts\python.exe src/generate_dataset_3d.py --config configs/data_config_3d_full.yaml --out_dir data/raw_3d_full
.\.venv-3d\Scripts\python.exe src/graph_dataset_3d.py --raw_dir data/raw_3d_full --out_dir data/processed_3d_full
.\.venv-3d\Scripts\python.exe src/train.py --model baseline --data_dir data/processed_3d_full --config configs/train_config_3d.yaml --out_dir outputs/3d
.\.venv-3d\Scripts\python.exe src/evaluate.py --data_dir data/processed_3d_full --ckpt_dir outputs/3d/checkpoints --out_dir outputs/3d --cpu_threads 4
.\.venv-3d\Scripts\python.exe src/visualize_3d.py --model baseline
```

`data_config_3d_full.yaml`、`train_config_3d.yaml`、训练摘要、评估 JSON、MMS JSON 和图像共同构成结果的最小可追溯集。改变网格、训练集拆分、硬件、线程数或模型宽度后，应生成新的输出目录，不覆盖本基线。

## 10. 局限与下一步

1. **物理保真度：**需要温度相关物性、接触热阻、真实封装层厚、各向异性与实验测温/高保真商用 FEM 对照。
2. **网格与几何泛化：**当前每一维度内的样本共享网格；应随机化层厚、热点形状和网格分辨率，并采用跨网格测试。
3. **OOD：**重点增加低 \(h_{top}\) 工况和边界表示，报告最坏工况与置信区间，而不仅是平均 MAE。
4. **模型：**先以相同拆分、参数预算和评价指标超过 3D GraphSAGE，再增加 MGN 消息传递步数；不要直接恢复 8 步、128 维。
5. **外部基准：**在 Linux/WSL 中确认 MFIT 的依赖和案例能运行后，再进行同条件 RC--FEM--GNN 比较；外部二维数据集可用于迁移学习或跨拓扑研究，但不能替代三维体场验证。

### PDF 导出

统一 PDF 由本 Markdown 文件生成，避免维护两份可能失真的内容源。安装 `reportlab` 后执行：

```powershell
.\.venv-3d\Scripts\python.exe src/export_technical_pdf.py
```

默认输出为 `docs/chip_thermal_gnn_technical_documentation.pdf`。导出器使用无 TeX、无浏览器的离线 PDF 后端；公式保留为清晰的 TeX 文本样式，确保可重复地呈现。

## 参考文献与开源引用

1. T. Gustafsson and G. D. McBain, “scikit-fem: A Python package for finite element assembly,” *Journal of Open Source Software*, 5(52), 2369, 2020. [DOI: 10.21105/joss.02369](https://doi.org/10.21105/joss.02369)
2. M. Fey and J. E. Lenssen, “Fast Graph Representation Learning with PyTorch Geometric,” 2019. [arXiv:1903.02428](https://arxiv.org/abs/1903.02428). PyTorch 见 [官方文档](https://pytorch.org/docs/stable/index.html)。
3. W. L. Hamilton, R. Ying and J. Leskovec, “Inductive Representation Learning on Large Graphs,” *NeurIPS*, 2017. [论文](https://proceedings.neurips.cc/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)
4. T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez and P. Battaglia, “Learning Mesh-Based Simulation with Graph Networks,” *ICLR*, 2021. [论文](https://openreview.net/pdf?id=roNqYL0_XP)；[开源实现](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)。
5. MOOSE Framework, “Heat Conduction Verification Tutorial.” [官方教程](https://mooseframework.inl.gov/releases/moose/2022-06-10/getting_started/examples_and_tutorials/tutorial03_verification/index.html)
6. A. Kanani et al., “MFIT: Multi-Fidelity Thermal Modeling for 2.5D and 3D Multi-Chiplet Architectures.” [源码与案例](https://github.com/AlishKanani/MFIT)
7. R. Ke, J. Tao and C. Liu, “Multi-topology Power Module Thermal Simulation Dataset.” [数据集、数据卡与 CC BY 4.0 许可](https://github.com/henkuailederen/Multi-topology-Dataset)
