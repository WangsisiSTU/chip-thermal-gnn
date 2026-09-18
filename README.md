# chip-thermal-gnn：三维芯片封装热场 GNN 代理模型

这是一个基于有限元网格图的稳态芯片封装热场代理项目。当前主线是**三维 P1 四面体 FEM + GraphSAGE 基线**；二维三角网格流程仍保留，用于方法验证和轻量级实验。

项目覆盖完整闭环：**工况采样 → FEM 求解 → 网格转图 → GNN 训练 → 精度、速度与 OOD 评估 → 切面可视化**。数据均由 `scikit-fem` 实际求解生成，不依赖商业热仿真软件。

MGN + 轻量 Physics-Attention 的实验路径已接入训练和评估。局部 MeshGraphNet 处理四面体网格边，随后将节点聚合为可学习物理状态，状态间做全局注意力并映射回节点；PyG 批次可包含不同节点数的图。2026-09-15 的初轮三分辨率 benchmark 使用 32 状态、仅中网格训练与单个种子，现保留为历史记录。2026-09-16 修正了图热源与 FEM 装配不一致的问题，增加四面体物理梯度、局部加密留出网格和三种子多网格测评；新的 8 状态轻量混合模型在局部网格的 die 梯度误差优于 GraphSAGE，但全场节点 MAE 尚未稳定更好。

## 当前状态

| 流程 | 状态 | 推荐用途 |
| --- | --- | --- |
| 三维四面体热场 | 当前主线 | 封装堆叠、x-z 热点、差散热泛化研究 |
| 三维差散热覆盖实验 | 已完成首轮训练与评估 | 低对流支持范围与严格 OOD 的区分 |
| MGN+TRANS 三网格 benchmark | 已完成单训练种子配对评估 | 跨分辨率迁移、精度/速度权衡；尚非任意网格泛化 |
| 修正热源后的 MGN+TRANS 四网格测评 | 已完成三种子及局部加密网格留出测试 | 热点梯度研究；全场精度仍以 GraphSAGE 为强基线 |
| ATPlace2.5D Case1 六层多芯粒模型 | 已完成三模型、三种子、布局/网格解耦评测 | 公开多芯粒对象；GraphSAGE、MGN、MGN+TRANS 公平对比 |
| ATPlace 接触/各向异性/温变材料 | 已完成五级收敛、物理消融和多实体图契约 | 界面温跳、非线性 Robin 能量投影；尚未宣称跨材料泛化 |
| 高级材料 MGN+TRANS | 已完成三模型、三种子、三套布局/网格评测 | 真实 checkpoint 云图；MGN+TRANS 场误差领先，MGN 界面温跳领先 |
| ATPlace 非线性算子热启动 | 已接通温度一致 FVM 残差与完整非线性修正 | 修正后约 `9.5e-7 K` MAE；端到端速度仍未超过冷启动 |
| 二维三角网格热场 | 保留 | 快速冒烟、教学和架构对照 |

三维工作流的完整实验记录见 [差散热覆盖研究](docs/cooling_coverage_study.md)，三维 FEM、数据格式与基线审查见 [三维执行报告](docs/three_dimensional_run_report.md)。

最新的 FEM 算子学习路线已经接入四面体刚度、Robin 面积分、材料界面热流、能量残差和三步稀疏算子前向校正；扩展为 384/64/128 个独立物理工况及四训练/两留出网格。实现、benchsize、云图和单种子能力检查见 [FEM 离散算子学习报告](docs/fem_operator_learning_3d_report.md)。

ATPlace2.5D Case1 已建立独立的六层多芯粒参考入口；其材料与边界契约不同于旧四层演示模型，
详情、复现命令和 ML 接口见 [ATPlace 计算对象说明](docs/atplace_computational_objects.md)。
24 个布局的分组训练及 Node MLP、GraphSAGE、守恒 GraphSAGE 对比见
[Case1 首轮 ML 报告](docs/atplace_case1_ml_report.md)。
MGN、MGN+TRANS 接入后的三种子结果，以及布局、网格和二者组合的三套配对评测见
[Case1 MGN+TRANS 泛化报告](docs/atplace_mgn_transolver_generalization_report.md)。
接触热阻、对角各向异性、温变导热率的离散方式、五级 benchsize、受控消融及图契约见
[Case1 高级材料物理报告](docs/atplace_contact_anisotropic_temperature_report.md)。
高级材料图真正接入 GraphSAGE、MeshGraphNet、MGN+TRANS 后的三种子结果和预测云图见
[高级材料 MGN+TRANS 端到端报告](docs/atplace_advanced_mgn_transolver_report.md)。
网络热启动、可重装配离散算子契约、真实残差训练及冷/热启动 benchmark 见
[ATPlace 非线性算子热启动报告](docs/atplace_nonlinear_warm_start_report.md)。该实验当前是 FVM，不应表述为四面体 FEM。

混合模型的 [三维原型说明](docs/mgn_transolver_3d_technical_documentation.md) 与 [初轮能力测评](docs/mgn_transolver_3d_benchmark_report.md)保留原始网格和结构记录；修正、四网格三种子效果对比、benchsize、模型消融、云图和复现命令见[修正后的三维测评报告](docs/mgn_transolver_3d_corrected_benchmark_report.md)。其 `96/16/32` 配对 benchmark 与上面的 `360/60/150` 差散热覆盖实验不是同一数据协议，不能直接横比 MAE；初轮与修正版的热源图输入、模型规模及训练协议也不同，不能只看旧表中的排名。

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

# 实验模型：使用同一份数据协议，输出到独立目录
python src/train.py --model mgn_transolver `
  --data_dir data/processed_3d_cooling_coverage `
  --config configs/train_config_3d_mgn_transolver.yaml `
  --out_dir outputs/3d/mgn_transolver --cpu_threads 4
python src/evaluate.py `
  --data_dir data/processed_3d_cooling_coverage `
  --ckpt_dir outputs/3d/mgn_transolver/checkpoints `
  --out_dir outputs/3d/mgn_transolver --models mgn_transolver `
  --cpu_threads 4 --timing_repeats 3 --benchmark_batch_size 2
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

外部复杂数据的轻量接入、字段映射和 split 风险见
[外部复杂热数据接入报告](docs/external_dataset_onboarding_report.md)。下载的第三方仓库和 HDF5
样本位于 `external/`（默认不提交）；可复现的样本清单和校验契约保存在 `configs/`。

## 当前限制与下一步

- 已完成 Case1 内的布局隔离和中/细网格配对评测；尚未验证跨芯粒数量或跨封装家族泛化。
- ATPlace Case1 已加入温变导热率、对角各向异性和显式接触热阻；参数目前是研究配置，尚未用实测材料卡标定，也尚未完成跨材料 OOD 训练。旧三维四面体主线仍需迁移相同的多实体面契约。
- 严格低对流外推仍是最大误差来源。下一步应在相同 150 样本测试集上做特征、损失和数据覆盖的消融，并用多个训练种子报告均值和方差。
