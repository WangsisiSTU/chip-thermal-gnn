# ATPlace2.5D 三维计算对象接入

## 当前已完成

官方包下载至 `external/atplace_download/ATPlace_pub-main`，保留原始输入和热配置。
运行 `python src/import_atplace_cases.py` 可生成 `data/atplace_import/Case1.json` 至
`Case10.json` 及 manifest。每个案例保存原始文件 SHA-256，长度从微米转换为米，功率保留瓦特。
该入口只解析文本，不运行上游加密布局内核。

| 案例 | 芯粒数 | Interposer mm | 总功率 W |
|---|---:|---|---:|
| Case1 | 6 | 42×42 | 780 |
| Case2 | 6 | 32×32 | 370 |
| Case3 | 8 | 39×39 | 680 |
| Case4 | 11 | 37×37 | 1535 |
| Case5 | 12 | 57×59 | 1020 |
| Case6 | 20 | 49×53 | 852 |
| Case7 | 28 | 30×25 | 260 |
| Case8 | 36 | 26×23 | 240 |
| Case9 | 44 | 59×61 | 1548 |
| Case10 | 61 | 47×47 | 1006 |

十个 `.pl` 的坐标全部为零。这些是布局优化输入，不是可以直接热求解的合法放置。
因此导入结果均标记 `ready_for_fem=false`，尚未生成 FEM 标签或启动 ML 训练。

运行 `python src/build_atplace_layouts.py` 生成本项目派生布局到 `data/atplace_layouts`。
整数微米 MaxRects 算法允许 90° 旋转，固定随机种子 3107；不改变芯粒尺寸、功率或封装尺寸。
Case1、3、5、6、7、8、9、10 已通过重叠、边界、尺寸和功率检查。
Case2、4 在当前启发式搜索预算内未找到布局（不等价于证明不可行），manifest 明确标记失败。
这些是零间距几何基准，不是官方优化结果，也不保证布线、制造间距或热性能。
坐标为 x/z 平面左下角，y 留作堆叠方向；尚未配置完整物理模型，仍不可直接视为 FEM 标签。
检查命令：`python -m unittest discover -s tests -p test_atplace_layouts.py`。

## 热结构来源

官方 `Thermal.py` 的六层厚度，自下而上为：substrate 200 μm、C4 70 μm、
interposer 110 μm、microbump 10 μm、chip 150 μm、TIM 20 μm，总计 560 μm，
不含额外的均热板和散热器。芯粒间隙用 underfill 填充；HBM 在该模型中为等效块，
不能解释为解析 HBM 内部每一层。

材料参数包含热阻率，转换导热率需取倒数。C4、TSV 和 microbump 有混合规则，
不能直接使用层默认材料替代 floorplan 的局部覆盖值。
`thermal/hotspot.config` 的环境温度为 318.15 K；辅助代码会修改 sink/spreader
尺寸。但对流热阻替换条件精确匹配值 `0.1` 的整行，下载的配置值却是 `0.01`，
故按当前文本代码路径不会应用其面积缩放公式，保留 0.01 K/W。未运行上游二进制验证。
公开 `reproduce.py` 使用解析紧凑模型，不直接调用该 Thermal.py 热求解助手；不能混为同一验证链路。
三维重建需要核对有效散热边界和 TIM 是否重复计入，不应沿用旧演示模型的底部定温条件。

## 实施顺序与验收

1. Case1 建立首个计算对象：从上游运行结果获取合法 layout，或自行生成并明确标注为
   “基于公开芯粒定义的派生布局”；检查边界、重叠、朝向与总面积。
2. 根据六层与局部材料定义构造三维网格。热源按 `P/(芯粒面积×厚度)` 分配，
   验证离散积分功率等于原始功率；检查薄层网格质量和网格收敛。
3. 在一致的有效边界下比较 FEM 与 HotSpot，记录峰值、芯粒均温、三维场、总能量误差。
   这是数值模型交叉验证，不等于实测校准。
4. 扩展 Case2–Case10，冻结 case、layout、工况与 mesh 标识。只有十个案例，宜采用
   留一案例验证或预先冻结的案例划分；不能把同一布局的不同功率样本当结构 OOD。
5. 先计时传统求解、复用分解与代理推理，再开展固定预算的布局热优化。
   最终布局由参考求解器复核，评价最高温度、优化时间和参考求解次数。

目前的原四层演示 FEM 不能直接表示上述对象，需要新增专用多芯粒/六层装配入口。
公共案例并未提供商业 7 nm 全工艺热模型，不对其作器件尺度自热精度声明。

## Case1 六层参考模型（已完成）

专用配置位于 `configs/atplace_case1_six_layer.yaml`，包含官方六层厚度、C4/TSV/
microbump 混合材料规则、6 个独立芯粒热源和 318.15 K 环境。边界为顶面等效 Robin，
总对流热阻 0.01 K/W，其余表面绝热；这是六层主体参考模型，不包含显式均热板和散热器。

运行：

```powershell
E:\Anaconda\python.exe src/atplace_six_layer_fvm.py
E:\Anaconda\python.exe src/atplace_ml_adapter.py
```

五档正交网格从 420 到 26,040 个控制体。每档离散输入均为 780 W，最终档热平衡相对误差
为 `4.14e-13`，温度范围为 318.1500–339.8074 K；最终两档峰温差小于 `1e-6 K`，
最大芯粒均温差为 0.0757 K。功率守恒、热平衡、离散最小值原理、峰温和芯粒均温
五项门槛全部通过。完整结果在 `data/atplace_case1_six_layer_fvm/convergence_report.json`。

最细网格已转为 26,040 节点、150,964 条有向边的 PyG 图，位置为
`data/atplace_case1_ml_reference/verification.pt`。它标记为 `reference_test_only`：单个 Case1
不能作为训练集，旧四层 checkpoint 也因材料、边界和特征契约不同而不能直接比较。
模型输出温升 `.npy` 后可运行：

```powershell
E:\Anaconda\python.exe src/evaluate_atplace_prediction.py prediction.npy
```

比较指标包含体积加权 MAE/RMSE、峰温误差、芯粒均温误差和预测场热平衡误差。
六面体 Galerkin FEM 诊断入口保留在 `src/atplace_six_layer_fem.py`；极薄层条件下其粗网格
会违反离散最小值原理，因此不会进入 ML 标签路径，保守有限体积结果作为当前参考标签。

首轮按布局分组的 ML 数据和三模型对比也已完成，详见
`docs/atplace_case1_ml_report.md`。最终采用不含固定局部 Jacobi 步数的守恒 GraphSAGE：
未见布局中网格 MAE 1.219 K，在独立 26k 单元细参考图上 MAE 0.618 K；两者的预测
热平衡相对误差分别为 `2.98e-7` 和 `5.41e-6`。

来源：https://github.com/PKU-IDEA/ATPlace_pub
