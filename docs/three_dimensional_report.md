# chip-thermal-gnn：三维四面体 FEM 与图神经网络基线说明

> 状态：本说明记录已完成的真实 scikit-fem/PyG 三维基线。第三方仓库尚未下载或运行；三维云图尚待执行可视化脚本后补入，见第 8 节。

## 1. 目标与边界

项目保留原有二维三角形 FEM—PyG—GNN 流程，并新增独立三维流程：

```text
三维分层封装几何 + x-z 平面热点
    -> P1 四面体 FEM（scikit-fem）
    -> 原始节点温度场 / 四面体连通性
    -> 三维 PyG 图
    -> GraphSAGE 基线 / 紧凑 MeshGraphNet 试验
    -> 场误差、峰值、OOD、时间与内存对比
```

本版三维模型是稳态线性导热：材料各向同性、层间理想接触、底部恒温、顶部对流、其余表面绝热。它用于建立可复现基线，而不是替代完整封装签核仿真。

## 2. 物理模型与 FEM 离散

令三维域为 \(\Omega=[0,W]\times[0,H]\times[0,D]\)，其中 \(y\) 为封装堆叠方向，层次为 substrate—Cu—TIM—die。稳态导热控制方程为

\[
-\nabla\cdot(k\nabla T)=q \qquad \text{in }\ \Omega.
\]

边界条件为

\[
T=T_{amb}\ \text{on }\Gamma_{bottom},\qquad
-k\nabla T\cdot n=h_{top}(T-T_{amb})\ \text{on }\Gamma_{top},\qquad
-k\nabla T\cdot n=0\ \text{on }\Gamma_{side}.
\]

对任意测试函数 \(v\)（在底部 Dirichlet 边界为零），弱式写成

\[
\int_\Omega k\nabla T\cdot\nabla v\,d\Omega+
\int_{\Gamma_{top}}h_{top}Tv\,d\Gamma=
\int_\Omega qv\,d\Omega+
\int_{\Gamma_{top}}h_{top}T_{amb}v\,d\Gamma.
\]

每个单元为一阶四面体 \(e\)，\(T_h=\sum_{i=1}^{4}N_iT_i\)。单元矩阵和体热源载荷为

\[
K^{(e)}_{ij}=\int_{V_e} k\nabla N_i\cdot\nabla N_j\,dV,
\qquad f_i^{(e)}=\int_{V_e}qN_i\,dV.
\]

顶部三角形面 \(s\) 的 Robin 项被装配为 \(h_{top}\int_sN_iN_jdS\)；对于线性三角形，局部质量矩阵为

\[
\frac{h_{top}A_s}{12}
\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix},
\]

右端同时增加 \(h_{top}T_{amb}A_s/3\)。实现位于 `src/fem_solver_3d.py`，正常环境使用 scikit-fem 稀疏装配；离线回退仅限小网格烟雾测试。

### x-z 平面热点

热点位于 die 中，在 \(x-z\) 平面是矩形，沿 die 厚度延展：

\[
q(x,y,z)=\begin{cases}
q_{hot}, & y>y_{TIM},\ x\in[x_c-w_x/2,x_c+w_x/2],\ z\in[z_c-w_z/2,z_c+w_z/2],\\
q_{base}, & y>y_{TIM}\ \text{且不在热点内},\\
0, & \text{其他层}.
\end{cases}
\]

\(x_c,z_c,w_x,w_z\) 使用相对 \(W,D\) 的随机比例采样，因此 OOD 测试可保持几何和热点定义不变，只外推功率、TIM 导热率或对流系数。

## 3. 三维 PyG 图设计

每个四面体的六条无向边去重后保留双向，得到有向 `edge_index`。当前完整网格有 1,859 节点、8,640 个四面体和 22,532 条有向边。

- 节点特征（24 维）：\((x/W,y/H,z/D)\)、4 类材料 one-hot、局部热源、环境温度、节点对流系数、顶部/底部边界标志，以及广播的全局工况特征。
- 边特征（5 维）：\((\Delta x/W,\Delta y/H,\Delta z/D)\)、归一化边长、端点导热率调和平均值。
- 标签：节点温升 \(\Delta T=T-T_{amb}\)。
- 坐标：`pos.shape=(N,3)`；四面体连通性以 `tets.shape=(4,M)` 写入原始网格文件。

全局工况被广播到每个节点，是因为稳态椭圆问题具有全局耦合，有限消息传递轮数无法可靠地将远处热源/边界参数传播到整张图。

## 4. 可复现环境与测试原理

独立环境为 `.venv-3d`，关键版本为 Python 3.12、scikit-fem 12.0.2、PyTorch 2.13.0+cpu、PyG 2.8.0.post1、SciPy 1.18.1、NumPy 2.5.2。安装使用 pip `truststore`，即系统受信任证书存储；没有关闭 TLS 校验。`pip check` 无依赖冲突。

真实后端测试命令：

```powershell
.\.venv-3d\Scripts\python.exe -m pytest tests/test_fem_solver_3d.py tests/test_graph_dataset_3d.py -q
```

结果为 **5 passed**。测试覆盖：

1. 小规模 P1 四面体求解的节点/单元形状、有限温度、底部恒温和顶部 Robin 标志；
2. 正热源下温度上升，最高温节点位于 die；
3. 热点仅在指定 x-z 矩形和 die 中生效；
4. 四面体六条边均双向、无自环；
5. 三维节点/边特征的形状、one-hot 一致性、边长和等效导热率为正。

后续数值验证应加入制造解（MMS）网格收敛率：选定解析 \(T^*(x,y,z)\)，令 \(q=-\nabla\cdot(k\nabla T^*)\)，并由 \(T^*\) 构造 Dirichlet/Robin 边界。线性四面体在足够光滑的解下应展示预期的 \(L^2\) 与能量范数收敛趋势。这是与物理样例分开的代码正确性验证路径。

## 5. 数据集与资源检查

默认小规模检查集（24/4/6）已生成并转换：

| 项目 | 结果 |
| --- | ---: |
| 节点 / 四面体 / 有向边 | 1,859 / 8,640 / 22,532 |
| 峰值温升范围 | 0.2987–2.1923 K |
| 单图张量占用 | 0.972 MiB |
| 34 图张量估算 | 33.1 MiB |
| 可用系统内存（检查时） | 约 14.6 GiB |

完整固定种子数据集为 300/30/30；原始 FEM 生成耗时 39.41 s，平均单样本 FEM 求解时间 92.61 ms。处理后的 360 张图张量估算为约 350 MiB。完整配置见 `configs/data_config_3d_full.yaml`。

## 6. 模型与测试结果

### 6.1 可信主基线：GraphSAGE

GraphSAGE 使用 3 层、隐藏维度 64、batch size 4、4 个 CPU 线程，参数量 19,713。固定种子 2024 下，第 69 epoch 达到最佳验证损失 0.05422，因早停共运行 94 epoch，训练耗时 439.29 s。无 CUDA 设备，因此报告 CPU 进程 RSS 而非显存。

| 测试项 | 结果 |
| --- | ---: |
| 整体 MAE | **0.0835 K** |
| 相对 \(L^2\) | 0.4945 |
| \(R^2\) | 0.6509 |
| 峰值温升绝对误差 | **0.1206 ± 0.1330 K** |
| 单样本推理 | 7.43 ms |
| 批量摊销推理 | 6.43 ms / 样本 |
| 平均 FEM | 112.25 ms / 样本 |
| 评估 RSS | 600.84 MiB |
| CUDA 峰值显存 | N/A（CPU） |

OOD 分解显示主要瓶颈为差对流外推：`ood_high_power` MAE 0.0489 K、`ood_poor_tim` 0.0283 K、`ood_poor_cooling` **0.5283 K**。后续模型改进应首先围绕顶部对流/散热路径，而不是盲目增加消息传递步数。

### 6.2 紧凑 MeshGraphNet 试验

为避免二维 8 次消息传递、128 隐藏维在三维图上造成边级状态膨胀，试验采用 2 次消息传递、64 维、batch size 8、12 epoch。该试验是成本/学习曲线探针，不是充分调优模型。

| 测试项 | 结果 |
| --- | ---: |
| 参数量 | 73,153 |
| 训练时间 | 343.80 s |
| 训练 RSS 峰值 | 887.57 MiB |
| 整体 MAE | 0.1433 K |
| 峰值误差 | 0.1529 K |
| 加权 OOD MAE | 0.3060 K |
| 单样本推理 | 19.13 ms |

该紧凑 MGN 暂时不如 GraphSAGE（尤其是 OOD），且代价更高。当前结论是：**继续把 GraphSAGE 作为可信 3D 主基线；MGN 只有在增加数据量、改进边/边界编码或采用 GPU 后才值得做系统调参。**

## 7. 2D vs 3D 对比

| 指标 | 2D GraphSAGE | 3D GraphSAGE |
| --- | ---: | ---: |
| 参数量 | 104,321 | 19,713 |
| 节点 / 有向边 | 2,257 / 13,152 | 1,859 / 22,532 |
| MAE | 0.1035 K | **0.0835 K** |
| 峰值误差 | 0.3431 K | **0.1206 K** |
| 加权 OOD MAE | 0.2490 K | **0.1866 K** |
| FEM | 8.50 ms | 112.25 ms |
| 推理 | 54.94 ms | 7.43 ms |
| 进程 RSS | 520.09 MiB | 600.84 MiB |
| CUDA 峰值显存 | N/A | N/A |

表格 CSV/Markdown 位于 `outputs/3d/metrics/comparison_2d_vs_3d_baseline.*`。推理时间不能只解释为二维/三维效应：二维既有检查点为 4 层、128 维，三维主基线为 3 层、64 维。可比较的结论是：在本数据与配置下，三维带来了更低的场 MAE、峰值误差和 OOD MAE，但 FEM 成本显著上升，且三维图的边数增加约 71%。

## 8. 云图结果与待执行可视化

本版已产生三维温度场数据，但尚未运行新的三维可视化代码；因此不能将不存在的三维云图作为已验证结果。获准执行代码后，应为每个代表测试工况（中位 ID、最高 ID、最高 OOD）生成：

1. FEM 与 GNN 预测的 \(y=y_{die,mid}\) x-z 热点平面温升云图；
2. FEM 与 GNN 预测的中心 x-y、y-z 剖面云图；
3. 误差云图 \(\Delta T_{pred}-\Delta T_{FEM}\)；
4. 同一色标下的热点峰值、截面曲线与 OOD 注释。

可视化应使用四面体插值/切面采样，不能直接复用二维 `matplotlib.tri` 三角形渲染逻辑。生成后图像保存到 `outputs/3d/figures/`，并嵌入本节。

## 9. 可公开参考的算例、数据与代码

以下资源已完成网页检索，尚未下载或执行。许可证、数据可获得性与输入格式须在实际接入前再次核验。

| 资源 | 可作为何种对照 | 建议用途 |
| --- | --- | --- |
| [MOOSE heat-conduction verification](https://mooseframework.inl.gov/releases/moose/2022-06-10/getting_started/examples_and_tutorials/tutorial03_verification/index.html) | 解析解/MMS 与 FEM 收敛 | 优先实现与本项目独立的 3D 制造解收敛测试，验证四面体装配和 Robin 边界。 |
| [scikit-fem](https://github.com/kinnala/scikit-fem) | `MeshTet.init_tensor` 与稀疏 FEM 工作流 | 作为当前 solver 的实现级参考；可扩展到外部 `.msh` 网格导入。 |
| [MFIT](https://github.com/AlishKanani/MFIT) | 2.5D/3D chiplet 热 RC/DSS 与 FEM 参考文件 | 优先检查其公开的几何、材料、功耗和 ANSYS 参考文件；适合作为稳态/瞬态 chiplet 对比，而非直接替换本四面体 FEM。 |
| [PODTherm-GP](https://github.com/CompResearchLab/PODTherm-GP) | 三维芯片动态热代理 | 用于比较代理建模思路（POD/GP vs GNN）和动态扩展路线。 |
| [DeepMind MeshGraphNets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) | 原始网格图学习训练/评估管线 | 仅用于架构与数据管线基准；公开示例不是芯片热问题。 |
| [NVIDIA PhysicsNeMo MeshGraphNet examples](https://github.com/NVIDIA/physicsnemo/blob/main/examples/README.md) | 现代 PyTorch MeshGraphNet 数据管线 | 用于对照数据读取、可变网格和 GPU 扩展做法；不应把流体/结构结果直接当作热学精度基准。 |
| [Multi-topology thermal dataset](https://github.com/henkuailederen/Multi-topology-Dataset) | 稳态 FEM 温度场与多拓扑功率模块数据 | 可评估跨拓扑温度场代理；其 COMSOL/HDF5 数据格式与本项目需单独适配。 |

推荐的接入顺序：先完成本地 MMS，再做 MFIT 的稳态 chiplet 输入/输出对照，最后评估 Multi-topology Dataset 的许可和 HDF5 映射。第三方代码只应放入隔离目录并保留原始许可证；对比实验需先固定网格、单位、边界条件、功率定义和评价指标。

## 10. 执行许可后的工作包

在收到许可前不执行下列操作：

1. 新增并运行三维切面/云图生成脚本；
2. 克隆或下载 MFIT、PODTherm-GP、Multi-topology Dataset 等第三方资源；
3. 安装其依赖、转换外部数据或运行第三方示例；
4. 运行 MMS 收敛、外部算例与本项目的对比实验。

建议先批准的最小工作包为：仅运行本仓库新增三维云图与 MMS 验证；确认图像和收敛后，再单独批准第三方资源下载与对比运行。
