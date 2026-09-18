# ATPlace2.5D Case1 首轮 ML 对比

## 数据协议

- 计算对象：Case1，6 个芯粒，42×42 mm 中介层，官方六层热结构。
- 几何：24 个本项目生成的合法派生布局；保持芯粒尺寸、身份和封装尺寸不变。
- 分组划分：18 个训练布局、3 个验证布局、3 个测试布局，`layout_id` 不跨集合。
- 每个布局 4 个功率向量，共 72/12/12 张 train/val/test 图。
- 单芯粒功率倍率为 0.70–1.30；总功率约 621–956 W。
- 训练采用 medium 网格，约 4.5k–6.3k 个控制体；目标为稳态温升。

这里的测试是 Case1 内部的布局 OOD，不代表跨芯粒数量、跨封装或商业 7 nm 器件泛化。

## 相同中网格、未见布局测试

| 模型 | 体积加权 MAE K | RMSE K | 峰温绝对误差 K | 芯粒均温误差 K | 热平衡相对误差 |
|---|---:|---:|---:|---:|---:|
| Node MLP | 1.720 | 2.377 | 0.561 | 1.070 | 6.07e-2 |
| GraphSAGE | 1.289 | 1.734 | 0.462 | 1.036 | 2.49e-2 |
| 守恒 GraphSAGE | **1.219** | **1.675** | **0.276** | **0.956** | **2.98e-7** |

守恒模型不使用固定次数的局部求解器步进。它只利用导热算子的常数温度零模，对每张图施加
一个解析温升平移，使 Robin 排热量严格等于输入功率；梯度和局部温差不变。

## 未见基准布局、26k 单元细网格

| 模型 | 体积加权 MAE K | RMSE K | 峰温误差 K | 最大单元误差 K | 热平衡相对误差 | RTX 3060 推理 ms |
|---|---:|---:|---:|---:|---:|---:|
| Node MLP | 1.386 | 1.772 | 0.511 | 5.683 | 9.00e-2 | 0.90 |
| GraphSAGE | 0.686 | 0.838 | 0.870 | 2.984 | 1.20e-2 | 3.22 |
| 守恒 GraphSAGE | **0.618** | **0.773** | **0.437** | **2.459** | **5.41e-6** | 3.39 |

一次失败消融也得到保留：固定 5 步 Jacobi 校正在中网格上可训练，但迁移到细网格后 MAE
恶化到 5.42 K。原因是固定离散步数不是网格无关的物理算子，因此最终协议将其关闭。
推理时间包含 5 次预热和 20 次 CUDA 同步计时；不能直接与未包含网格装配的数据生成总时间相比。

## 复现

```powershell
E:\Anaconda\python.exe src/generate_atplace_ml_dataset.py
E:\Anaconda\python.exe src/train_atplace_ml.py
E:\Anaconda\python.exe src/atplace_ml_adapter.py
E:\Anaconda\python.exe src/evaluate_atplace_checkpoints.py
```

训练配置为 `configs/atplace_case1_ml.yaml`，最终汇总分别位于
`outputs/atplace_case1_ml/comparison.json` 和
`outputs/atplace_case1_ml/fine_reference_comparison.json`。
