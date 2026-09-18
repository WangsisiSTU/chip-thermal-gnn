# 三维芯片热场 MGN + TRANS 原型说明

本说明针对仓库当前的 `MGNTransolverHybrid` 路径及 2026-09-15 的配对网格 benchmark。它是 MeshGraphNet（MGN）消息传递与 **Transolver 启发的轻量 physics-state attention** 组合，不是原论文 Transolver++ 的逐层复现；在本仓库中简称 MGN+TRANS。

## 三维原型与热问题

封装外形为 `20 × 12 × 16 mm`，沿 `y` 方向从底到顶依次为基板 `6 mm`、Cu `3 mm`、TIM `1 mm`、die `2 mm`。die 内有 `x-z` 矩形热点，底面施加环境温度 Dirichlet 条件，顶面施加对流 Robin 条件。每个材料区的导热系数、热点位置/宽度/功率、环境温度与顶面对流系数均可变化。稳态 FEM 解的是

`-∇·(k(x)∇T)=q(x)`，底面 `T=T_ambient`，顶面 `-k ∂T/∂n=h_top(T-T_ambient)`。

![三维封装与中网格剖面](../outputs/3d/mgn_transolver_benchmark/figures/package_mesh_3d.png)

图中可见四层封装及中网格 `z=0` 前剖面的真实网格节点和张量网格线；三维单元内部由四面体组成，完整连接统计见下表。图示星号是热点位置示意，实际每个样本的位置与宽度不同。

## 网格与图数据

FEM 采用 P1 四面体。`MeshTet.init_tensor` 由张量节点网格划分四面体，`ny=13` 使 `y=6/9/10/12 mm` 的材料层界面与节点平面对齐。三个分辨率仅改变 `nx,nz`，几何、材料采样及测试工况不变。

| 网格 | `(nx, ny, nz)` | 节点 | 四面体 | 去重有向图边 | 节点特征 | 边特征 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 粗 | `(9,13,7)` | 819 | 3,456 | 9,412 | 26 | 5 |
| 中（唯一训练网格） | `(13,13,11)` | 1,859 | 8,640 | 22,532 | 26 | 5 |
| 细 | `(17,13,15)` | 3,315 | 16,128 | 41,220 | 26 | 5 |

图边取每个四面体的六条无向边，再生成双向、去重连接。节点还存储由四面体体积按 P1 lumped mass 分摊的正权重，统一归一化为平均值 1；这些权重进入混合模型的 physics-state 聚合，并用于体积加权评估。模型预测每个节点的温升 `ΔT=T-T_ambient`，不是直接预测绝对温度；由中网格训练集的均值和标准差进行标准化与解码。

26 个节点输入包含三维归一化坐标、四种材料 one-hot、局部热源、环境温度、对流节点值、顶/底边界标志，以及热点和材料/边界参数等 14 项全局特征；全局特征包含 `1/k_TIM`、`1/h_top`、`log(h_top)` 与估算的顶部热阻占比。5 个边输入为 `dx,dy,dz`、边长及两端材料谐均导热系数，均按配置中的物理参考尺度归一化。完整字段名与尺度可查看 [图数据构建代码](../src/graph_dataset_3d.py) 和各 `metadata.json`。

## 模型结构与小网格路径

![MGN+TRANS 原型计算路径](../outputs/3d/mgn_transolver_benchmark/figures/architecture_prototype.png)

本次混合模型宽度 128，先做 3 次 MGN 风格的边更新、按目标节点求和聚合及残差节点更新；随后有 2 个 physics-state block，每个 block 把各图节点按可学习的软切片权重聚合到 32 个状态，做 4 头状态间 attention 与 FFN，再把状态结果映射回节点。软切片聚合按节点的 lumped 体积加权；PyG `batch` 图 ID 隔离不同样本，因此一批图可有不同节点数而无需 padding。最后的节点解码器输出一个标准化 `ΔT` 值。

设节点数为 `N`、有向边数为 `E`、隐宽为 `H`、状态数为 `S`，计算主要是局部消息 `O(EH)`、切片/回映 `O(NSH)` 和状态 attention `O(S²H)`；没有全节点的 `O(N²)` attention。这也是小规模网格使用简化状态 Transformer 的直接实现方式。当前 `S=32` 是固定的研究型选择，未证明是所有网格最优；当网格进一步减小，可在训练配置中减少切片数/全局 block，重新训练，不应直接修改已训练 checkpoint 的结构。

对照模型是 3 层、宽度 64 的 GraphSAGE（仅使用节点输入和连接）与 2 轮、宽度 64 的纯 MeshGraphNet（使用节点/边输入）。参数量分别为 19,969、73,281 和 732,803；因此本次是**相同数据与训练协议**的工程能力比较，不是等参数量消融。混合模型架构见 [模型源码](../src/models.py)。

## 数据工况与训练/评估协议

每个网格使用同一随机种子 2024、独立的 split/工况采样流，逐条核验 144 个样本的工况参数完全一致。训练/验证/测试为 `96/16/32`。训练中有 24 个 `h_top=100–400` 的已覆盖弱冷却工况；验证中 4 个，测试中 8 个。测试剩余为正常 ID 12 个、严格 OOD 高功率/低 TIM 导热/极弱冷却各 4 个。严格 OOD 的 `h_top=20–80` 与训练 `100–400`、正常 `500–15000` 不重叠。

三模型仅在中网格训练，使用同一随机种子、每模型最多 80 epoch、训练 batch size（`benchsize`）**2 张图**、Adam `lr=1e-3`、`weight_decay=1e-5`、梯度裁剪 1、同一按样本相对场损失及权重 0.1 的全图最大热点损失，并由中网格验证集选最优 checkpoint。评估在三网格各 32 个测试样本上进行；吞吐 benchmark batch size（`benchsize`）**4 张图**，预热后每个单样本与整批前向重复 **5 次**，同步 CUDA 再计时。模型、图构造和 FEM 离线生成分开计时，推理仅为已处理图的前向耗时，不包括生产系统的数据传输/预处理。

跨网格命令必须显式使用 `--cross_mesh`。它只容许目标训练集的 `dT_train_mean/std` 因网格变化而不同；节点/边字段、物理尺度、维度及 lumped volume contract 仍严格匹配。预测始终使用**中网格 checkpoint** 的输出统计量解码。

## 复现

本机实测环境：Windows、RTX 3060、PyTorch `2.5.1+cu121`、PyG `2.8.0.post1`、scikit-fem `12.0.2`、SciPy `1.13.1`、NumPy `1.26.4`。下列命令从仓库根目录执行；使用已具备这些包的 Python 环境，实际实验使用 `E:\Anaconda\python.exe`。

```powershell
$py = 'E:\Anaconda\python.exe'
foreach ($spec in @(@('coarse',9,7),@('medium',13,11),@('fine',17,15))) {
  $name,$nx,$nz = $spec
  & $py src/generate_dataset_3d.py --config configs/data_config_3d_mgn_transolver_benchmark.yaml --out_dir "data/raw_3d_mgn_transolver_benchmark_$name" --nx $nx --ny 13 --nz $nz
  & $py src/graph_dataset_3d.py --raw_dir "data/raw_3d_mgn_transolver_benchmark_$name" --out_dir "data/processed_3d_mgn_transolver_benchmark_$name"
}
foreach ($model in @('baseline','meshgraphnet','mgn_transolver')) {
  & $py src/train.py --model $model --data_dir data/processed_3d_mgn_transolver_benchmark_medium --config configs/train_config_3d_mgn_transolver_benchmark.yaml --out_dir outputs/3d/mgn_transolver_benchmark --seed 2024
}
foreach ($name in @('coarse','medium','fine')) {
  $extra = if ($name -eq 'medium') { @() } else { @('--cross_mesh') }
  & $py src/evaluate.py --data_dir "data/processed_3d_mgn_transolver_benchmark_$name" --ckpt_dir outputs/3d/mgn_transolver_benchmark/checkpoints --out_dir "outputs/3d/mgn_transolver_benchmark/eval_$name" --models baseline meshgraphnet mgn_transolver --timing_repeats 5 --benchmark_batch_size 4 @extra
}
& $py src/analyze_3d_mesh_reference_shift.py
& $py src/summarize_mgn_transolver_benchmark.py
& $py src/generate_mgn_transolver_benchmark_figures.py
& $py -m pytest -q
```

数值结论、效果对比、FEM 离散化差异与能力边界见 [三维能力测评报告](mgn_transolver_3d_benchmark_report.md)。
