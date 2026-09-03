"""Create the research figures embedded in the unified technical report.

The script consumes the frozen 3-D evaluation summaries and the two existing
slice-cloud figures.  It does not train a model or regenerate FEM data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "3d" / "metrics"
SOURCE_FIGURES = ROOT / "outputs" / "3d" / "figures"


def configure_style() -> None:
    """Use a readable CJK-capable style when a Windows font is available."""
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )


def read_summary(model: str) -> dict:
    with (METRICS_DIR / f"{model}_eval_summary.json").open(encoding="utf-8") as f:
        return json.load(f)


def weighted_ood_mae(summary: dict) -> float:
    regimes = [v for name, v in summary["regime_breakdown"].items() if name != "id"]
    return sum(v["n_samples"] * v["mae_K_mean"] for v in regimes) / sum(
        v["n_samples"] for v in regimes
    )


def add_box(ax, xy, width, height, text, color, fontsize=10):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=1.25,
        facecolor=color,
        edgecolor="#243447",
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.35,
    )


def arrow(ax, start, end, color="#4a5568", connection="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.35,
            color=color,
            connectionstyle=connection,
        )
    )


def plot_method_pipeline(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    add_box(ax, (0.4, 3.0), 2.25, 1.7, "参数采样\n热点、材料、边界条件", "#e8f1fb")
    add_box(ax, (3.35, 5.15), 2.35, 1.45, "二维 P1 三角形 FEM\n保留既有流程", "#edf7ed")
    add_box(ax, (3.35, 1.4), 2.35, 1.45, "三维 P1 四面体 FEM\n分层封装 + x-z 热点", "#dff3f5")
    add_box(ax, (6.45, 3.0), 2.2, 1.7, "温升标签 $\\Delta T$\n节点坐标、材料、边界", "#fff4d6")
    add_box(ax, (9.35, 3.0), 2.25, 1.7, "PyG 图数据\n节点 / 有向边 / 全局特征", "#f5e8f4")
    add_box(ax, (12.2, 5.15), 2.25, 1.45, "GraphSAGE\n可信主基线", "#e8f1fb")
    add_box(ax, (12.2, 1.4), 2.25, 1.45, "紧凑 MGN\n受控对照", "#f6ece5")
    add_box(ax, (12.2, 3.0), 2.25, 1.0, "评估：MAE、峰值、\nID/OOD、内存、时间", "#f3f4f6", fontsize=9)

    arrow(ax, (2.65, 4.1), (3.35, 5.85))
    arrow(ax, (2.65, 3.6), (3.35, 2.15))
    arrow(ax, (5.7, 5.85), (6.45, 4.1))
    arrow(ax, (5.7, 2.15), (6.45, 3.6))
    arrow(ax, (8.65, 3.85), (9.35, 3.85))
    arrow(ax, (11.6, 4.1), (12.2, 5.85))
    arrow(ax, (11.6, 3.6), (12.2, 2.15))
    arrow(ax, (13.32, 5.15), (13.32, 4.0))
    arrow(ax, (13.32, 2.85), (13.32, 3.0))

    ax.text(0.4, 7.45, "芯片封装二维/三维热场代理：端到端研究流程", fontsize=17, weight="bold")
    ax.text(0.4, 6.95, "二维流程作为回归基线；三维流程以真实四面体 FEM 标签和跨工况 OOD 测试为核心。", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_dir / "method_pipeline_2d_3d.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_thermal_gnn_prototype(out_dir: Path) -> None:
    """Draw the report's detailed but implementation-faithful architecture prototype."""
    fig, ax = plt.subplots(figsize=(18, 7.2))
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Header and the explicitly separate offline supervision lane.
    ax.text(0.35, 7.52, "三维芯片封装热场 GNN 研究原型", fontsize=19, weight="bold")
    ax.text(
        0.35,
        7.08,
        "主路径：参数化封装 → 三维 PyG 图 → GraphSAGE → 节点温升场；虚线分支仅用于离线 FEM 标签与训练监督。",
        fontsize=10.5,
        color="#455a64",
    )
    add_box(ax, (3.6, 6.05), 2.65, 0.7, "离线：P1 四面体 FEM（scikit-fem）", "#fff4d6", fontsize=9.5)
    add_box(ax, (6.85, 6.05), 1.75, 0.7, "标签 $\\Delta T_i$", "#fff4d6", fontsize=9.5)
    add_box(ax, (13.75, 6.05), 1.75, 0.7, "训练损失\n场 + 0.1×峰值", "#fff4d6", fontsize=8.8)

    # 3-D layered package: deliberately schematic, but preserves the package order.
    x0, y0, width, depth = 0.65, 2.2, 2.35, 0.48
    layers = [
        ("Die + x-z 热点", "#f6c17b"),
        ("TIM", "#f3df9e"),
        ("Cu", "#e8a35b"),
        ("Substrate", "#89b7d2"),
    ]
    for index, (label, color) in enumerate(layers):
        y = y0 + (len(layers) - 1 - index) * 0.78
        front = [(x0, y), (x0 + width, y), (x0 + width, y + 0.52), (x0, y + 0.52)]
        top = [(x0, y + 0.52), (x0 + 0.33, y + 0.72), (x0 + width + 0.33, y + 0.72), (x0 + width, y + 0.52)]
        side = [(x0 + width, y), (x0 + width + 0.33, y + 0.2), (x0 + width + 0.33, y + 0.72), (x0 + width, y + 0.52)]
        ax.add_patch(Polygon(front, facecolor=color, edgecolor="#334e68", linewidth=1.0))
        ax.add_patch(Polygon(top, facecolor=color, alpha=0.72, edgecolor="#334e68", linewidth=0.8))
        ax.add_patch(Polygon(side, facecolor=color, alpha=0.52, edgecolor="#334e68", linewidth=0.8))
        ax.text(x0 + width / 2, y + 0.25, label, ha="center", va="center", fontsize=8.5)
    ax.add_patch(Polygon([(1.25, 5.45), (2.05, 5.45), (2.22, 5.55), (1.42, 5.55)], facecolor="#e6553a", edgecolor="#c13c25"))
    ax.annotate("$q(x,z)$", xy=(1.73, 5.51), xytext=(1.73, 5.93), ha="center", fontsize=9, color="#b83b28", arrowprops={"arrowstyle": "-|>", "color": "#b83b28"})
    ax.text(0.65, 1.57, "参数化分层封装与工况", fontsize=11, weight="bold")
    ax.text(0.65, 1.19, "$k$、$q_{base}$、$q_{hot}$、热点几何、$h_{top}$、$T_{amb}$", fontsize=8.7, color="#455a64")

    # Current PyG graph representation.
    add_box(ax, (3.55, 2.0), 3.05, 3.25, "三维 PyG 图", "#e8f1fb", fontsize=12)
    node_x = [4.1, 4.75, 5.45, 4.35, 5.05, 5.8]
    node_y = [4.25, 4.72, 4.35, 3.4, 3.63, 3.3]
    edge_pairs = [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4), (4, 5), (2, 5)]
    for src, dst in edge_pairs:
        ax.plot([node_x[src], node_x[dst]], [node_y[src], node_y[dst]], color="#4f8fb9", linewidth=1.1, zorder=2)
    node_colors = ["#7ab8f5", "#f2b15e", "#8fcf80", "#7ab8f5", "#f2b15e", "#8fcf80"]
    ax.scatter(node_x, node_y, s=170, c=node_colors, edgecolors="#2f5d7c", linewidths=0.9, zorder=3)
    ax.text(3.85, 2.65, "节点 24 维：坐标、材料、$q$、边界\n+ 平铺的全局工况", fontsize=8.6, va="top")
    ax.text(3.85, 2.22, "四面体邻接 → 1,859 节点 / 22,532 有向边", fontsize=8.2, color="#455a64")

    # Main GraphSAGE network, shown as three current layers rather than generic GNN blocks.
    add_box(ax, (7.5, 2.0), 3.1, 3.25, "GraphSAGE 主基线", "#e9f5e8", fontsize=12)
    for index, label in enumerate(["输入 $X$", "SAGE 1\n64", "SAGE 2\n64", "SAGE 3\n64"]):
        x = 7.85 + index * 0.64
        ax.add_patch(FancyBboxPatch((x, 3.28), 0.48, 1.22, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#cce5c8" if index else "#d9ecfb", edgecolor="#4c7a4c" if index else "#3e759c", linewidth=1.0))
        ax.text(x + 0.24, 3.88, label, ha="center", va="center", fontsize=8)
        if index < 3:
            arrow(ax, (x + 0.48, 3.88), (x + 0.64, 3.88), color="#5c6f5c")
    ax.text(7.82, 2.62, "三层、隐藏维度 64、dropout 0.03\n消息传递使用 edge_index 与节点特征", fontsize=8.5, va="top")
    ax.text(7.82, 2.2, "边属性留给紧凑 MGN 对照，不是当前 SAGE 输入", fontsize=8.2, color="#8a5a2b")

    # Decoder and spatial output map.
    add_box(ax, (11.45, 2.0), 1.75, 3.25, "节点解码", "#f5e8f4", fontsize=11)
    ax.text(12.33, 4.12, "MLP", ha="center", fontsize=10)
    ax.text(12.33, 3.63, "$\\widehat{\\Delta T}_i$", ha="center", fontsize=13, weight="bold")
    ax.text(12.33, 2.65, "逐节点输出\n反标准化为 K", ha="center", fontsize=8.7)

    add_box(ax, (14.05, 2.0), 2.35, 3.25, "热场与峰值", "#f6e8ef", fontsize=11)
    grid = np.exp(-(((np.linspace(-1, 1, 22)[None, :]) - 0.22) ** 2 + ((np.linspace(-1, 1, 18)[:, None]) + 0.08) ** 2) / 0.28)
    ax.imshow(grid, extent=(14.35, 15.75, 3.25, 4.48), cmap="inferno", origin="lower", aspect="auto", zorder=1)
    ax.add_patch(FancyBboxPatch((14.35, 3.25), 1.4, 1.23, boxstyle="round,pad=0.015,rounding_size=0.02", facecolor="none", edgecolor="#5f4b5f", linewidth=0.9, zorder=2))
    ax.text(15.05, 2.66, "节点场、$\\max_i\\widehat{\\Delta T}_i$\n三切面云图", ha="center", fontsize=8.5)

    # The evaluation block is deliberately a research assessment, not a delivery dashboard.
    add_box(ax, (16.95, 2.0), 1.7, 3.25, "研究评估", "#f1f0f5", fontsize=11)
    ax.text(17.8, 4.2, "ID / OOD\n场 MAE\n峰值误差\n推理时间\nRSS", ha="center", va="top", fontsize=9.2, linespacing=1.55)

    # Main online arrows.
    arrow(ax, (3.05, 3.7), (3.55, 3.7))
    arrow(ax, (6.6, 3.7), (7.5, 3.7))
    arrow(ax, (10.6, 3.7), (11.45, 3.7))
    arrow(ax, (13.2, 3.7), (14.05, 3.7))
    arrow(ax, (16.4, 3.7), (16.95, 3.7))

    # Offline dashed supervision branch.
    arrow(ax, (2.25, 5.42), (3.6, 6.38), color="#c5862b", connection="arc3,rad=0.1")
    arrow(ax, (6.25, 6.38), (6.85, 6.38), color="#c5862b")
    arrow(ax, (8.6, 6.38), (13.75, 6.38), color="#c5862b")
    arrow(ax, (12.33, 5.25), (13.75, 6.05), color="#c5862b", connection="arc3,rad=-0.18")
    ax.text(9.5, 6.73, "离线监督分支（训练时使用）", fontsize=8.7, color="#9a651d", ha="center")

    # Legend follows the visual grammar of the supplied reference image.
    ax.add_patch(FancyBboxPatch((0.55, 0.25), 18.0, 0.55, boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#fbfbfb", edgecolor="#aab3ba", linewidth=0.8))
    legend_items = [
        ("#e8f1fb", "物理/图表示"),
        ("#e9f5e8", "GraphSAGE 主基线"),
        ("#fff4d6", "离线 FEM 标签/损失"),
        ("#f5e8f4", "节点温升输出"),
        ("#f6e8ef", "云图与研究评估"),
    ]
    for index, (color, text) in enumerate(legend_items):
        x = 1.05 + index * 3.45
        ax.add_patch(FancyBboxPatch((x, 0.39), 0.33, 0.23, boxstyle="round,pad=0.01,rounding_size=0.05", facecolor=color, edgecolor="#708090", linewidth=0.6))
        ax.text(x + 0.48, 0.505, text, va="center", fontsize=8.7)
    fig.tight_layout(pad=0.45)
    fig.savefig(out_dir / "thermal_gnn_prototype_architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_thermal_gnn_prototype_compact(out_dir: Path) -> None:
    """Draw a portrait-page-friendly version for the network-architecture section."""
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(0.45, 9.52, "三维热场代理原型：离线 FEM 监督与在线 GNN 推理", fontsize=15, weight="bold")
    ax.text(0.45, 9.12, "GraphSAGE 为当前主基线；MGN 仅作为使用边属性的受控对照。", fontsize=8.8, color="#455a64")

    # One physical source feeds both the offline-label and online-surrogate lanes.
    add_box(
        ax,
        (0.55, 4.1),
        2.25,
        2.15,
        "参数化 3D 封装\nDie / TIM / Cu / Substrate\n$q(x,z)$、$k$、$h_{top}$、$T_{amb}$",
        "#e8f1fb",
        fontsize=9.4,
    )
    ax.text(0.55, 6.74, "同一物理工况", fontsize=9, color="#3e759c", weight="bold")

    # Offline supervision lane.
    ax.text(3.15, 8.1, "离线标签生成与训练监督", fontsize=9.5, color="#9a651d", weight="bold")
    add_box(ax, (3.15, 6.55), 2.15, 1.1, "P1 四面体 FEM\nscikit-fem", "#fff4d6", fontsize=9.5)
    add_box(ax, (5.9, 6.55), 1.55, 1.1, "标签\n$\\Delta T_i$", "#fff4d6", fontsize=10)
    add_box(ax, (9.65, 6.55), 1.75, 1.1, "训练损失\n场 + 0.1×峰值", "#fff4d6", fontsize=8.8)
    arrow(ax, (2.8, 5.92), (3.15, 6.98), color="#c5862b", connection="arc3,rad=0.18")
    arrow(ax, (5.3, 7.1), (5.9, 7.1), color="#c5862b")
    arrow(ax, (7.45, 7.1), (9.65, 7.1), color="#c5862b")

    # Online surrogate lane.
    ax.text(3.15, 5.55, "在线图推理路径", fontsize=9.5, color="#3e759c", weight="bold")
    add_box(
        ax,
        (3.15, 2.1),
        2.15,
        2.75,
        "三维 PyG 图\n节点：24 维特征\n边：四面体邻接\n1,859 节点 /\n22,532 有向边",
        "#e8f1fb",
        fontsize=8.9,
    )
    graph_nodes = [(3.7, 3.92), (4.35, 4.32), (4.88, 3.78), (3.98, 2.98), (4.62, 2.75)]
    for s, d in [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)]:
        ax.plot([graph_nodes[s][0], graph_nodes[d][0]], [graph_nodes[s][1], graph_nodes[d][1]], color="#4f8fb9", linewidth=1.0)
    ax.scatter(*zip(*graph_nodes), s=68, c=["#7ab8f5", "#f2b15e", "#8fcf80", "#7ab8f5", "#f2b15e"], edgecolors="#2f5d7c", linewidths=0.6, zorder=3)
    add_box(ax, (5.95, 2.1), 2.25, 2.75, "GraphSAGE 主基线\n3 × SAGEConv\n隐藏维度 64\n\n输入：$x$ 与\nedge_index", "#e9f5e8", fontsize=9.3)
    add_box(ax, (8.85, 2.1), 1.35, 2.75, "节点\n解码器\n\nMLP\n\n$\\widehat{\\Delta T}_i$", "#f5e8f4", fontsize=9.5)
    add_box(ax, (10.75, 2.1), 1.05, 2.75, "温升场\n峰值\n切面云图\n\nID / OOD\nMAE", "#f6e8ef", fontsize=8.4)
    arrow(ax, (2.8, 4.55), (3.15, 4.0), color="#4a6a86", connection="arc3,rad=-0.18")
    arrow(ax, (5.3, 3.48), (5.95, 3.48), color="#4a6a86")
    arrow(ax, (8.2, 3.48), (8.85, 3.48), color="#4a6a86")
    arrow(ax, (10.2, 3.48), (10.75, 3.48), color="#4a6a86")
    arrow(ax, (9.52, 4.85), (9.92, 6.55), color="#c5862b", connection="arc3,rad=0.25")

    # MGN is visibly a controlled comparison instead of an implied ensemble.
    ax.text(0.55, 0.95, "紧凑 MGN 对照：同一图，额外消费 edge_attr（$\\Delta x,\\Delta y,\\Delta z$、边长、等效导热率）。", fontsize=8.15, color="#8a5a2b")
    ax.text(0.55, 0.52, "图例：蓝=物理/图表示；绿=GraphSAGE 主基线；黄=离线 FEM；紫=节点温升输出。", fontsize=8.05, color="#455a64")
    fig.tight_layout(pad=0.35)
    fig.savefig(out_dir / "thermal_gnn_prototype_compact.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_id_ood_clouds(out_dir: Path) -> None:
    id_image = plt.imread(SOURCE_FIGURES / "slice_clouds_baseline_sample19_id.png")
    ood_image = plt.imread(SOURCE_FIGURES / "slice_clouds_baseline_sample27_ood_poor_cooling.png")
    # The first row is the x-z hotspot plane.  Cropping to it gives a direct,
    # legible success/failure comparison; the full three-plane fields remain
    # separate figures in the report.
    cutoff = int(id_image.shape[0] * 0.34)
    id_image = id_image[:cutoff]
    ood_image = ood_image[:cutoff]
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.9))
    entries = [
        (axes[0], id_image, "ID 成功：样本 #19，热点位置与厚度方向扩散均被捕捉"),
        (axes[1], ood_image, "差对流 OOD 失败：样本 #27，整体温升被系统性高估"),
    ]
    for ax, image, title in entries:
        ax.imshow(image)
        ax.set_title(title, loc="left", fontsize=14, pad=12, weight="bold")
        ax.axis("off")
    fig.suptitle("3D GraphSAGE：同一 x-z 热点切面上的 ID 与 OOD 对照", fontsize=17, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / "id_vs_ood_success_failure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ood_breakdown(graphsage: dict, mgn: dict, out_dir: Path) -> None:
    keys = ["id", "ood_high_power", "ood_poor_tim", "ood_poor_cooling"]
    labels = ["ID", "高功率\nOOD", "差 TIM\nOOD", "差对流\nOOD"]
    colors = ["#2274a5", "#e76f51"]
    x = list(range(len(keys)))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    metrics = [("mae_K_mean", "MAE [K]"), ("peak_abs_err_K_mean", "峰值绝对误差 [K]")]
    for ax, (metric, ylabel) in zip(axes, metrics):
        values_a = [graphsage["regime_breakdown"][key][metric] for key in keys]
        values_b = [mgn["regime_breakdown"][key][metric] for key in keys]
        bars_a = ax.bar([v - width / 2 for v in x], values_a, width, label="GraphSAGE", color=colors[0])
        bars_b = ax.bar([v + width / 2 for v in x], values_b, width, label="紧凑 MGN", color=colors[1])
        ax.bar_label(bars_a, fmt="%.3f", padding=3, fontsize=8)
        ax.bar_label(bars_b, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    axes[0].set_title("分项场误差")
    axes[1].set_title("分项峰值误差")
    fig.suptitle("三维测试集：ID 与 OOD 分项误差", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "ood_error_breakdown_3d.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_selection(graphsage: dict, mgn: dict, out_dir: Path) -> None:
    models = ["GraphSAGE", "紧凑 MGN"]
    color = ["#2274a5", "#e76f51"]
    panels = [
        ("整体 MAE [K]", [graphsage["overall_mae_K"], mgn["overall_mae_K"]], "越低越好"),
        ("加权 OOD MAE [K]", [weighted_ood_mae(graphsage), weighted_ood_mae(mgn)], "越低越好"),
        ("单样本推理 [ms]", [graphsage["mean_inference_time_ms"], mgn["mean_inference_time_ms"]], "越低越好"),
        ("评估 RSS [MiB]", [graphsage["process_rss_MiB"], mgn["process_rss_MiB"]], "越低越好"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
    for ax, (title, values, note) in zip(axes, panels):
        bars = ax.bar(models, values, color=color, width=0.62)
        ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], padding=3, fontsize=10, weight="bold")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.text(0.5, -0.18, note, ha="center", transform=ax.transAxes, fontsize=9, color="#4a5568")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelrotation=0)
    fig.suptitle("三维模型选型：GraphSAGE 在相同划分上胜出四项主指标", fontsize=15.5, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "model_selection_graphsage_3d.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SOURCE_FIGURES,
        help="Directory for report figures (default: outputs/3d/figures).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    graphsage = read_summary("baseline")
    mgn = read_summary("meshgraphnet")
    plot_method_pipeline(args.output_dir)
    plot_thermal_gnn_prototype(args.output_dir)
    plot_thermal_gnn_prototype_compact(args.output_dir)
    plot_id_ood_clouds(args.output_dir)
    plot_ood_breakdown(graphsage, mgn, args.output_dir)
    plot_model_selection(graphsage, mgn, args.output_dir)
    print(f"Wrote report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
