# 三维热场 MGN+TRANS 修正实验：热点形状改善，全场精度仍未胜出

本报告记录 2026-09-16 在当前仓库完成的修正、三维 FEM 配对测试及能力测评。2026-09-15 的[初轮报告](mgn_transolver_3d_benchmark_report.md)只用了一个训练种子、单一训练网格，而且图输入热源与 FEM 装配不一致；它保留为历史记录，本报告是进一步判断架构效果的依据。这里的 `MGN+TRANS` 是本项目的**轻量 MeshGraphNet + Physics-State Attention 混合实现**，不是原论文的百万点并行 Transolver++ 全量复现。

## 对问题的直接回答

1. **为什么复杂网格的预测云图可能不如 GraphSAGE？** 原先按节点几何位置采样的 `q_node` 与 FEM 按 `∫qφᵢ` 装配的热源不同，在细网格的原始图输入上平均少计约 24.4% 的 FEM 源功率；同时，全局状态混合容易把局部热点扩散成低频、偏宽的云图。修正热源后，该问题不再能归咎于漏热源，但全场 MAE 仍未稳定改善。局部加密测试表明 GraphSAGE 的局部梯度可能更锯齿化，混合模型的梯度更接近 FEM，却也更容易出现宽范围温度偏差。
2. **直接换成完整 Transolver++ 会解决吗？** 不能从论文或当前实验证明。我们试了池化、仅切片、状态自适应温度、Gumbel 切片；单网格/单种子筛选没有一种同时改善场、梯度与热点。原论文重点是百万点规模和大几何上的状态退化问题，本项目仅 819–3,659 节点。[Transolver++ 原文](https://arxiv.org/html/2502.02414v2)描述的自适应机制值得作为消融，而不应预设为小网格热场的必胜修复。
3. **目前更有效的路线是什么？** 用 FEM 同源的热源特征与物理单元梯度，保留轻量 GraphSAGE 为全场精度/速度基线；MGN+全局状态路径用于研究跨网格热点形状，但下一步优先验证热流/弱形式一致性、多尺度区域消息传递和更广泛的局部网格，而不是单纯增加 attention 深度、切片数或梯度损失权重。[RIGNO](https://arxiv.org/html/2501.19205)的区域图与分辨率鲁棒学习、[有限元物理约束算子学习](https://arxiv.org/abs/2405.12465)提供相关方向，但其论文任务与这里的稳态分层封装不完全相同。

## 本轮实际修正

稳态方程为 `-∇·(k∇T)=q`，以三维 P1 四面体 FEM 的温升 `ΔT=T-T_amb` 为监督。旧图热源是节点上的 `q(xᵢ)`；FEM 右端实际是 `bᵢ=∫Ω qφᵢ dV`。现在图使用

```text
qᵢ = bᵢ / mᵢ,   mᵢ = ∫Ω φᵢ dV = Σ_{e∋i} Vₑ/4
Σᵢ qᵢmᵢ = Σᵢ bᵢ = FEM 装配的热功率
```

处理旧原始 FEM 数据时，转图步骤根据物理 case 与保存的四面体网格**重新装配**同一个热源向量，无须改写旧标签。32 个测试工况的 `Σqᵢmᵢ` 与 FEM 源功率的平均相对差：粗 `4.56×10⁻⁸`、中 `5.36×10⁻⁸`、细 `5.62×10⁻⁸`、局部加密 `4.23×10⁻⁸`，所有最大差均低于 `1.9×10⁻⁷`（float32 图数据回算）。这是**离散输入与求解器一致**，不是与实际芯片测量热源一致的验证。

图现在保存每个四面体的 4 节点索引、P1 物理梯度系数、体积及 die 标记。训练可加入 die 内部体积加权的相对单元梯度误差；评估新增 die 梯度 MAE（K/mm）、die 场 MAE、峰值位置误差和逐工况 P95。只在 die 单元计算梯度，不会把 TIM/Cu/die 的真实材料界面硬性抹平。还加入局部自适应四面体加密、跨网格配对训练、不同 attention 状态方式和 checkpoint 输入契约检查。

## 实验范围、三维网格与 benchsize

封装域 `20×12×16 mm`，沿 y 自下而上为基板 6 mm、铜 3 mm、TIM 1 mm、die 2 mm；底部定温、顶部 Robin 对流、侧面绝热。die 中有 x-z 矩形基础热源和热点。物理采样种子及 case 列表冻结在[数据协议](../configs/data_config_3d_mgn_transolver_benchmark.yaml)：每张网格 **144 个 FEM 工况，96 train / 16 val / 32 test**；测试中正常 12、已覆盖弱冷却 8、高功率 OOD 4、差 TIM OOD 4、极弱冷却 OOD 4。四网格 case 的参数、顺序和 split 完全相同。

| 网格 | 构造 | FEM 节点 | P1 四面体 | 去重有向图边 | 训练用途 | 测试工况 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 粗 | 张量 `(9,13,7)` | 819 | 3,456 | 9,412 | 多网格训练 | 32 |
| 中 | 张量 `(13,13,11)` | 1,859 | 8,640 | 22,532 | 训练及 checkpoint 验证 | 32 |
| 细 | 张量 `(17,13,15)` | 3,315 | 16,128 | 41,220 | 多网格训练 | 32 |
| 局部 | 中网格出发，TIM/die 中央局部加密 3 轮 | 3,659 | 18,408 | 45,992 | **完全留出拓扑** | 32 |

多网格训练为每个模型每个种子 `96×3=288` 张图，但只有 **96 个不同物理训练工况**，不能称作 288 个独立 case。验证仅用中网格 16 case；每个模型每个种子在四网格各 32 个相同的物理测试 case 上评估。三个初始化种子 `2024/2025/2026`、上限 60 epoch、**训练 batch size=2 图**；推理计时为预热后单图 3 次、**benchmark batch size=4 图**吞吐 3 次，设备为项目机器 RTX 3060 / PyTorch 2.5.1 + CUDA 12.1。计时只含已处理图的模型前向，不含 FEM 前处理、网格生成或真实工程求解全流程。

局部加密保留同一分层长方体几何与同一随机物理工况，只改变中央 TIM/die 附近的四面体尺寸和连接。它比规则分辨率变化更严格，但**不是任意 CAD 几何、任意接触界面或独立网格生成器的泛化证明**。32 个局部测试 case 在与中网格完全相同的基础节点位置上，FEM 参考标签平均差 `0.01333 K`、P95 `0.03358 K`、平均峰值差 `0.05042 K`；局部网格代理误差的增加不能全部解释为 FEM 标签漂移。

## 模型与损失

主对照是 3 层宽度 64 GraphSAGE（19,969 参数）和宽度 64、3 轮局部 MGN、1 个 4 头/8 切片 Physics-State Attention 残差块的混合模型（142,307 参数）；局部 MGN 使用 mean 邻居聚合以减轻节点度数改变的影响。全图 attention 用学得的状态映射节点↔状态，并以可学残差门控制注入。两者不等参数量，因此性能差不能纯粹归因于 attention。

```text
3D P1 FEM: bᵢ=∫qφᵢ、mᵢ、四面体梯度  →  图节点(qᵢ=bᵢ/mᵢ) + 边/材料/边界
                                             ├─ GraphSAGE ×3 → 节点 ΔT̂
                                             └─ MGN(mean) ×3 → 8 物理状态
                                                               → 1 个 4 头 attention
                                                               → 残差映回节点 → 节点 ΔT̂
```

主目标是按样本相对温升场损失 `+0.1×` 全图最大温升误差 `+0.1×` die 单元相对梯度损失；Adam 初始学习率 `1e-3`，最佳模型由中网格验证集选择。梯度项是 die 单元内 `ΣVₑ||∇T̂ₑ−∇Tₑ||²/(ΣVₑ||∇Tₑ||²+ε)`，不等于报告中以 K/mm 表示的梯度 MAE。单种子把梯度权重提高到 `0.5` 时，细网格混合模型梯度 MAE 从 `0.0497` 降至 `0.0445 K/mm`，但节点 MAE 从 `0.1033` 升至 `0.1321 K`：单靠加大权重是在牺牲温度场。

## 单中网格训练：机制筛选，不作为最终排序

先只在中网格训练 96 case、用局部及细网格零样本测试，统一 2024 种子与修正输入、60 epoch、同损失。`full` 是主 MGN+状态 attention；`pool` 仅状态池化；`slice_only` 去掉状态间 attention；`adaptive` 使用节点自适应切片温度；`adaptive_gumbel` 训练时随机 Gumbel、评估时确定性 softmax。此处旨在判断机制，只有一个种子。

| 模型 | 细网格 MAE K | 细网格 die 梯度 K/mm | 局部 MAE K | 局部 die 梯度 K/mm |
| --- | ---: | ---: | ---: | ---: |
| GraphSAGE | **0.0867** | 0.0483 | **0.1483** | 0.1039 |
| 纯 MGN | 0.1148 | 0.0573 | — | — |
| MGN + pool | 0.1193 | **0.0406** | 0.2123 | **0.0608** |
| MGN + slice_only | 0.1599 | 0.0453 | — | — |
| MGN + full | 0.1033 | 0.0497 | 0.1719 | 0.0684 |
| MGN + adaptive | 0.0924 | 0.0586 | — | — |
| MGN + adaptive_gumbel | 0.0897 | 0.0664 | — | — |

单网格状态诊断中，full 的 8 切片归一化概率熵为 `0.302`（明显非均匀），adaptive 为 `0.951`、adaptive_gumbel 为 `0.962`；状态间余弦相似度仍高 (`0.941/0.995/0.955`)。熵和余弦只是训练后表征诊断，不能单独证明机制因果。在本小网格实验中，自适应版本的确定性切片分配更接近均匀，且梯度指标更差，**不能直接把原论文提出的百万点状态退化修复搬为这里的答案**。[LinearNO 对 Physics-Attention 的分析](https://ojs.aaai.org/index.php/AAAI/article/view/37003)也提醒切片间 attention 应实测消融，而非预设必有收益。

## 三种子、三网格训练、局部网格留出测试

下表均为三个重训种子的平均。`节点 MAE` 是所有测试图的节点绝对温升误差平均；`die 梯度` 是各 die 四面体上按体积加权的 P1 梯度 MAE；`峰值误差` 为每工况 `|max ΔT̂−max ΔT|` 的均值；`P95` 是每次评估 32 个工况节点 MAE 的 95 分位，再跨种子平均。

| 测试网格 | 模型 | 节点 MAE K ↓ | die 梯度 K/mm ↓ | 峰值误差 K ↓ | 工况 MAE P95 K ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| 粗 | GraphSAGE | **0.0824** | 0.0336 | 0.2466 | **0.2722** |
| 粗 | MGN+TRANS | 0.0974 | **0.0322** | **0.2455** | 0.3476 |
| 中 | GraphSAGE | **0.0860** | 0.0341 | 0.3013 | **0.2627** |
| 中 | MGN+TRANS | 0.1032 | **0.0322** | **0.2880** | 0.3511 |
| 细 | GraphSAGE | **0.0830** | 0.0380 | 0.2997 | **0.2657** |
| 细 | MGN+TRANS | 0.0976 | **0.0347** | **0.2752** | 0.3472 |
| 局部加密 | GraphSAGE | **0.1504** | 0.1035 | 0.4039 | **0.4775** |
| 局部加密 | MGN+TRANS | 0.1819 | **0.0563** | **0.2939** | 0.6868 |

差值按同一物理测试 case、同一种子做 `MGN+TRANS − GraphSAGE`；负值为混合模型好。10,000 次配对重采样**同时重采样 3 个训练种子与 32 个测试 case**，以下为探索性 95% percentile 区间，不能当作新几何总体或真实器件误差的置信区间。

| 测试网格 | 节点 MAE 差 K [95%] | die 梯度差 K/mm [95%] | 峰值误差差 K [95%] |
| --- | --- | --- | --- |
| 粗 | `+0.0150 [-0.0110,+0.0575]` | `-0.0014 [-0.0041,+0.0010]` | `-0.0011 [-0.1001,+0.1156]` |
| 中 | `+0.0172 [-0.0090,+0.0579]` | `-0.0018 [-0.0070,+0.0015]` | `-0.0133 [-0.1077,+0.0847]` |
| 细 | `+0.0146 [-0.0107,+0.0584]` | `-0.0033 [-0.0108,+0.0007]` | `-0.0245 [-0.1259,+0.0794]` |
| 局部加密 | `+0.0314 [-0.0115,+0.1061]` | **`-0.0472 [-0.0898,-0.0174]`** | `-0.1100 [-0.2233,+0.0052]` |

因此目前能支持的窄结论是：**在这一个留出的局部加密拓扑与 32 个配对工况上，混合模型的 die 热场梯度形状改善有重复种子证据**；全场温度 MAE、热点峰值误差虽然某些均值方向不同，区间均跨零。局部网格混合模型的工况 P95 反而高于 GraphSAGE。三种子集成平均后，局部 GraphSAGE/MGN+TRANS 分别为 `MAE 0.1325/0.1755 K`、die 梯度 `0.0958/0.0496 K/mm`，仍保持同一权衡，且集成前向需运行三个模型。不能把 2026-09-15 初轮单种子的“平均 MAE 较低”当成稳定架构优势。

![局部加密网格，正常工况：FEM、GraphSAGE、MGN+TRANS 同色标 die 切片和误差](../outputs/3d/benchmark_corrected_summary/figures/local_seed2024_case0.png)

![局部加密网格，极弱冷却工况：三模型同色标 die 切片](../outputs/3d/benchmark_corrected_summary/figures/local_seed2025_case29.png)

以上云图使用相同温升色标并单独给误差色标；图示个案不取代 32 case×3 seed 的统计。原型、完整三维层结构和早期 FEM 网格图可参见[三维原型说明](mgn_transolver_3d_technical_documentation.md)。

三个多网格主模型的切片熵分别为 `0.513/0.717/0.864`，状态余弦 `0.979/0.937/0.928`，随种子明显变化；单一“状态坍塌”诊断不足以解释结果。真正需要定位的是：非均匀拓扑下局部导热系数/面积表示、长程状态混合与物理边界条件的相互作用。

## 能力边界和后续优先级

当前已经解决的是**图热源与 FEM 装配矛盾**，并建立了物理梯度、局部网格和多种子测评。尚未解决的是跨局部拓扑的**全场温度精度优势**，也未证明新封装形状、芯粒阵列、接触热阻、瞬态、真实实验校准或百万点规模能力。当前图边的导热系数是谐均值等近似特征，并非完整 FEM 边刚度；Robin 表面积分也尚未作为明确节点/面特征输入，因此复杂网格上输入物理算子仍有表达缺口。

按信息增益与工程收益排序：

1. **保持双指标门槛。** GraphSAGE 继续作为快速全场基线；混合模型必须同时检查节点/体积 MAE、die 梯度、峰值、工况 P95 和用时，不能只挑单张漂亮云图。
2. **做局部热流与边界的离散一致性。** 在四面体上验证 `k∇T` 与导热/Robin 弱形式残差，补充面面积、材料界面、热通量与 FEM 能量平衡特征；新增损失先用小权重消融，避免重复 `0.5` 梯度项的场精度恶化。
3. **扩展真正的拓扑与工况分布。** 多个不一样的局部加密区域、多个独立 mesh seed、非矩形 die/多热点/界面热阻及不同层厚；按物理 case 配对切分，至少 3 种子，避免训练/测试共享物理 case 被误计为泛化。
4. **再比较区域/多尺度图与全局 attention。** 以 [RIGNO](https://arxiv.org/html/2501.19205)式区域图或粗细层次图作为等预算对照，以纯 MGN/GraphSAGE 扩宽版和小型 LinearNO 类映射做容量消融；只在大网格足以暴露状态瓶颈时，再评估 Transolver++ 的自适应温度/Gumbel 与并行实现。

## 复现与可检查产物

核心修正位于 [FEM 热源与局部网格](../src/fem_solver_3d.py)、[四面体图和梯度几何](../src/graph_dataset_3d.py)、[模型机制](../src/models.py)、[训练](../src/train.py)、[评估](../src/evaluate.py)。训练配置见[修正版配置](../configs/train_config_3d_mgn_transolver_corrected.yaml)。逐种子、逐工况原始 CSV 和评估 JSON 位于 `outputs/3d/benchmark_corrected_multimesh_seed{2024,2025,2026}/eval_{coarse,medium,fine,local}/metrics/`；配对重采样机读结果在 `outputs/3d/benchmark_corrected_summary/paired_seed_case_bootstrap.json`。这些大文件和数据默认被 Git 忽略，需要本地生成，不作为仓库中的已发布 benchmark 数据。

```powershell
# 旧 benchmark 原始 FEM 粗/中/细网格仍可复用；新拓扑按同一物理配置生成
python src/generate_dataset_3d.py --config configs/data_config_3d_mgn_transolver_benchmark.yaml `
  --out_dir data/raw_3d_mgn_transolver_corrected_local --refine_levels 3

# 转图会把每个物理 case 的热源重新投影到 FEM 节点
foreach ($g in @('coarse','medium','fine')) {
  python src/graph_dataset_3d.py --raw_dir "data/raw_3d_mgn_transolver_benchmark_$g" `
    --out_dir "data/processed_3d_mgn_transolver_corrected_$g"
}
python src/graph_dataset_3d.py --raw_dir data/raw_3d_mgn_transolver_corrected_local `
  --out_dir data/processed_3d_mgn_transolver_corrected_local

foreach ($s in @(2024,2025,2026)) {
  foreach ($m in @('baseline','mgn_transolver')) {
    python src/train.py --model $m --seed $s --config configs/train_config_3d_mgn_transolver_corrected.yaml `
      --data_dir data/processed_3d_mgn_transolver_corrected_medium `
      --extra_train_dirs data/processed_3d_mgn_transolver_corrected_coarse `
        data/processed_3d_mgn_transolver_corrected_fine `
      --out_dir "outputs/3d/benchmark_corrected_multimesh_seed$s"
  }
  foreach ($g in @('coarse','medium','fine','local')) {
    python src/evaluate.py --data_dir "data/processed_3d_mgn_transolver_corrected_$g" `
      --ckpt_dir "outputs/3d/benchmark_corrected_multimesh_seed$s/checkpoints" `
      --out_dir "outputs/3d/benchmark_corrected_multimesh_seed$s/eval_$g" `
      --models baseline mgn_transolver --cross_mesh --timing_repeats 3 --benchmark_batch_size 4
  }
}
python src/summarize_corrected_benchmark.py
python -m pytest tests -q
```

最后完整测试为 **45 passed**、1 条预期警告。统计、图与命令均为本项目的合成 FEM benchmark，不能作为芯片实测精度承诺。
