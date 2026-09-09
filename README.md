# chip-thermal-gnn：三维芯片封装热场 GNN 代理模型

这是一个基于有限元网格图的稳态芯片封装热场代理项目。当前主线是**三维 P1 四面体 FEM + GraphSAGE 基线**；二维三角网格流程仍保留，用于方法验证和轻量级实验。

项目覆盖完整闭环：**工况采样 → FEM 求解 → 网格转图 → GNN 训练 → 精度、速度与 OOD 评估 → 切面可视化**。数据均由 `scikit-fem` 实际求解生成，不依赖商业热仿真软件。

## 当前状态

| 流程 | 状态 | 推荐用途 |
| --- | --- | --- |
| 三维四面体热场 | 当前主线 | 封装堆叠、x-z 热点、差散热泛化研究 |
| 三维差散热覆盖实验 | 已完成首轮训练与评估 | 低对流支持范围与严格 OOD 的区分 |
| 二维三角网格热场 | 保留 | 快速冒烟、教学和架构对照 |

三维工作流的完整实验记录见 [差散热覆盖研究](docs/cooling_coverage_study.md)，三维 FEM、数据格式与基线审查见 [三维执行报告](docs/three_dimensional_run_report.md)。

## 三维物理模型

封装域为 `Ω = [0, W] × [0, H] × [0, D]`，其中 `y` 为堆叠方向。自下而上包含基板、铜层、TIM 与 Die；默认网格为 1,859 个节点、8,640 个四面体和 22,532 条有向图边。

```text
substrate → Cu → TIM → die + x-z 矩形热点
          ↓
  P1 tetrahedral FEM
          ↓
  PyG mesh graph → GraphSAGE / MeshGraphNet
```

求解稳态导热方程：

```text
-div(k grad(T)) = q
```

底部为 `T = T_amb` 的 Dirichlet 边界，顶部为对流 Robin 边界，侧面绝热。标签为节点温升 `dT = T - T_amb`，而不是绝对温度。

三维图的节点特征包含坐标、材料、局部热源、边界标记和全局工况参数。差散热研究额外加入 `log(h_top)` 与顶/底热阻分流近似特征，共 26 维。checkpoint 会保存特征契约与归一化参数；评估和可视化会拒绝不兼容数据集。

## 已完成的差散热实验

当前推荐配置将“模型支持的低对流范围”和“严格外推”分开：

- 常规对流：`h_top = 500–15000`
- 已覆盖差散热：`h_top = 100–400`，训练/验证/测试分别为 100/20/30 样本
- 严格差散热 OOD：`h_top = 20–80`，25 个独立测试样本

首轮 3 层、64 隐藏维 GraphSAGE 使用全局峰值损失训练 150 轮，最佳 checkpoint 位于 epoch 141：

| 测试工况 | 样本数 | MAE | 相对 L2 | 峰值误差 |
| --- | ---: | ---: | ---: | ---: |
| ID | 45 | 0.0180 K | 0.1080 | 0.0669 K |
| 已覆盖差散热 | 30 | 0.0463 K | 0.1096 | 0.1607 K |
| 严格 OOD：高功率 | 25 | 0.0730 K | 0.1949 | 0.2972 K |
| 严格 OOD：差 TIM | 25 | 0.1105 K | 0.5728 | 0.2660 K |
| 严格 OOD：差散热 | 25 | 0.1565 K | 0.2066 | 0.4086 K |
| 总体 | 150 | 0.0713 K | 0.2810 | 0.2142 K |

在 4 CPU 线程、预热后每样本执行 3 次前向的口径下，单样本推理为 6.70 ms；4 图批量为 6.23 ms/样本。对应本轮 FEM 平均 20.29 ms，速度比为 3.03× 和 3.26×。

> 这组结果与早期 30 样本三维基线不直接横比：测试范围、样本比例、特征和训练目标均已改变。它说明已覆盖的 `100–400` 差散热范围已具有低误差，`20–80` 严格外推仍是主要短板。

## 快速开始：推荐三维流程

```powershell
pip install -r requirements.txt

# 1. 生成独立的三维差散热覆盖 FEM 数据
python src/generate_dataset_3d.py `
  --config configs/data_config_3d_cooling_coverage.yaml `
  --out_dir data/raw_3d_cooling_coverage

# 2. 四面体网格转为 PyTorch Geometric 图
python src/graph_dataset_3d.py `
  --raw_dir data/raw_3d_cooling_coverage `
  --out_dir data/processed_3d_cooling_coverage

# 3. 训练 GraphSAGE 基线
python src/train.py --model baseline `
  --data_dir data/processed_3d_cooling_coverage `
  --config configs/train_config_3d_cooling_coverage.yaml `
  --out_dir outputs/3d/cooling_coverage --cpu_threads 4

# 4. 固定测试集评估；计时会预热并使用小批次吞吐
python src/evaluate.py `
  --data_dir data/processed_3d_cooling_coverage `
  --ckpt_dir outputs/3d/cooling_coverage/checkpoints `
  --out_dir outputs/3d/cooling_coverage --models baseline `
  --cpu_threads 4 --timing_repeats 3 --benchmark_batch_size 4

# 5. FEM、图数据和训练工具测试
python -m pytest tests -q
```

数据、checkpoint 和重复评估输出默认不纳入 Git；配置、测试和机器可读的实验协议会被版本控制。

## 二维基础流程（保留）

二维流程使用 20 mm × 12 mm 分层截面、P1 三角形 FEM 与 21 维节点特征，适用于快速验证。它不代表当前三维结果。

```powershell
python src/generate_dataset.py --config configs/data_config.yaml --out_dir data/raw
python src/graph_dataset.py --raw_dir data/raw --out_dir data/processed
python src/train.py --model baseline --data_dir data/processed
python src/evaluate.py --data_dir data/processed
python src/visualize.py
```

二维和三维的统一技术细节见 [技术文档](docs/technical_documentation.md)。

## 项目结构

```text
configs/
  data_config_3d_cooling_coverage.yaml   # 当前推荐的三维数据协议
  train_config_3d_cooling_coverage.yaml  # 当前推荐的 GraphSAGE 配置
src/
  fem_solver_3d.py          # 三维分层封装 P1 四面体 FEM
  generate_dataset_3d.py    # 独立随机流、覆盖/OOD 工况采样
  graph_dataset_3d.py       # 四面体网格转图与物理特征
  train.py                  # 训练、全局峰值损失和数据契约保存
  evaluate.py               # 契约验证、精度和预热后计时
  visualize_3d.py           # 三维切面云图
  fem_solver.py             # 保留的二维 FEM 流程
  graph_dataset.py          # 保留的二维图数据流程
tests/                      # FEM、图、采样、训练和评估回归测试
docs/
  cooling_coverage_study.md
  three_dimensional_run_report.md
  technical_documentation.md
```

## 当前限制与下一步

- 所有三维样本共享固定几何与网格，尚未验证跨几何或跨网格泛化。
- 材料均为线性、各向同性；未加入温度相关材料、接触热阻或辐射。
- 严格低对流外推仍是最大误差来源。下一步应在相同 150 样本测试集上做特征、损失和数据覆盖的消融，并用多个训练种子报告均值和方差。