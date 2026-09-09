"""
可视化脚本：生成项目要求的全部图片并保存到 outputs/figures/。

生成的图片：
    1. field_triptych_{model}_sample{idx}.png  真实温度场 / 预测温度场 / 误差场 三联图
    2. peak_scatter.png                        测试样本峰值温升 真值-预测值 散点图
    3. metrics_bar.png                         主模型与基线模型指标柱状图
    4. loss_curves.png                         训练/验证损失曲线
    5. package_structure_mesh.png              芯片封装二维结构与网格示意图

用法：
    python src/visualize.py                        # 生成全部图片
    python src/visualize.py --figures field peak   # 只生成指定图片
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import checkpoint_normalization, load_model_from_ckpt
from utils import get_device, load_json, load_processed_metadata, load_split, load_yaml, validate_data_contract

MODEL_LABELS = {"meshgraphnet": "MeshGraphNet", "baseline": "GraphSAGE基线"}


def setup_matplotlib():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


@torch.no_grad()
def predict_dT(model, data, dT_mean, dT_std, device):
    data = data.to(device)
    pred_norm = model(data)
    return (pred_norm * dT_std + dT_mean).cpu().numpy()


def pick_showcase_samples(test_set, test_info):
    """自动挑选三联图展示样本：中等温升 id 样本、最大温升 id 样本、一个 OOD 样本。"""
    dT_maxes = np.array([float(d.dT_max_true.item()) for d in test_set])
    id_idx = [i for i, info in enumerate(test_info) if info["regime"] == "id"]
    ood_idx = [i for i, info in enumerate(test_info) if info["regime"] != "id"]

    picks = []
    if id_idx:
        id_sorted = sorted(id_idx, key=lambda i: dT_maxes[i])
        picks.append(id_sorted[len(id_sorted) // 2])   # 中位温升
        picks.append(id_sorted[-1])                    # 最大温升
    if ood_idx:
        ood_sorted = sorted(ood_idx, key=lambda i: dT_maxes[i])
        picks.append(ood_sorted[-1])                   # 温升最大的 OOD 样本
    return sorted(set(picks))


def fig_field_triptych(fig_dir, data_dir, ckpt_dir, sample_indices=None):
    device = get_device()
    proc_meta = load_processed_metadata(data_dir)
    test_set = load_split(data_dir, "test")
    test_info = load_json(os.path.join(data_dir, "test_info.json"))

    if sample_indices is None:
        sample_indices = pick_showcase_samples(test_set, test_info)

    for model_name in ["meshgraphnet", "baseline"]:
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[跳过] 未找到 {ckpt_path}")
            continue
        model, checkpoint = load_model_from_ckpt(ckpt_path, device)
        validate_data_contract(checkpoint, proc_meta)
        dT_mean, dT_std = checkpoint_normalization(checkpoint, device)

        for idx in sample_indices:
            data = test_set[idx]
            info = test_info[idx]
            pred = predict_dT(model, data, dT_mean, dT_std, device)
            true = data.y.cpu().numpy()
            err = pred - true

            pos = data.pos.cpu().numpy()
            x_mm, y_mm = pos[:, 0] * 1000, pos[:, 1] * 1000
            tris = None
            mesh_path = os.path.join("data", "raw", "mesh.npz")
            if os.path.exists(mesh_path):
                tris = np.load(mesh_path)["tris"].T
            tri = mtri.Triangulation(x_mm, y_mm, triangles=tris)

            fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), constrained_layout=True)
            vmin, vmax = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
            for ax, vals, title, cmap, norm_lim in [
                (axes[0], true, "真实温升场 (FEM)", "inferno", (vmin, vmax)),
                (axes[1], pred, f"预测温升场 ({MODEL_LABELS[model_name]})", "inferno", (vmin, vmax)),
                (axes[2], err, "误差场 (预测-真实)", "coolwarm", None),
            ]:
                if norm_lim is not None:
                    tpc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap, vmin=norm_lim[0], vmax=norm_lim[1])
                else:
                    lim = float(np.abs(vals).max()) + 1e-12
                    tpc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap, vmin=-lim, vmax=lim)
                ax.set_aspect("equal")
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("x [mm]")
                ax.set_ylabel("y [mm]")
                fig.colorbar(tpc, ax=ax, label="ΔT [K]", shrink=0.9)

            fig.suptitle(
                f"测试样本 #{idx}（{info['regime']}），真实峰值温升 {true.max():.2f} K，"
                f"预测峰值温升 {pred.max():.2f} K",
                fontsize=12,
            )
            out = os.path.join(fig_dir, f"field_triptych_{model_name}_sample{idx}.png")
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"已保存: {out}")


def fig_peak_scatter(fig_dir, data_dir, ckpt_dir):
    device = get_device()
    proc_meta = load_processed_metadata(data_dir)
    test_set = load_split(data_dir, "test")
    test_info = load_json(os.path.join(data_dir, "test_info.json"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, model_name in zip(axes, ["meshgraphnet", "baseline"]):
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[跳过] 未找到 {ckpt_path}")
            continue
        model, checkpoint = load_model_from_ckpt(ckpt_path, device)
        validate_data_contract(checkpoint, proc_meta)
        dT_mean, dT_std = checkpoint_normalization(checkpoint, device)

        peaks_true, peaks_pred, regimes = [], [], []
        for i, data in enumerate(test_set):
            pred = predict_dT(model, data, dT_mean, dT_std, device)
            peaks_true.append(float(data.y.max()))
            peaks_pred.append(float(pred.max()))
            regimes.append(test_info[i]["regime"])

        peaks_true, peaks_pred = np.array(peaks_true), np.array(peaks_pred)
        is_ood = np.array([r != "id" for r in regimes])
        lim = [0, max(peaks_true.max(), peaks_pred.max()) * 1.08]
        ax.plot(lim, lim, "k--", lw=1, label="y = x")
        ax.scatter(peaks_true[~is_ood], peaks_pred[~is_ood], c="tab:blue", s=45, alpha=0.8, label="分布内 (id)")
        ax.scatter(peaks_true[is_ood], peaks_pred[is_ood], c="tab:red", marker="^", s=55, alpha=0.85, label="分布外 (OOD)")
        ax.set_xlabel("真实峰值温升 [K]")
        ax.set_ylabel("预测峰值温升 [K]")
        ax.set_title(MODEL_LABELS[model_name])
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("测试集峰值温升：真值 vs 预测", fontsize=13)
    out = os.path.join(fig_dir, "peak_scatter.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out}")


def fig_metrics_bar(fig_dir, metrics_dir):
    comp_path = os.path.join(metrics_dir, "model_comparison.csv")
    if not os.path.exists(comp_path):
        print(f"[跳过] 未找到 {comp_path}，请先运行 evaluate.py")
        return
    df = pd.read_csv(comp_path)

    metrics = [
        ("MAE_K", "节点温度 MAE [K]（越低越好）"),
        ("RelativeL2", "相对 L2 误差（越低越好）"),
        ("PeakAbsErr_K", "峰值温度绝对误差 [K]（越低越好）"),
        ("R2", "R²（越高越好）"),
    ]
    colors = {"meshgraphnet": "tab:orange", "baseline": "tab:blue"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, (col, title) in zip(axes, metrics):
        labels = [MODEL_LABELS[m] for m in df["model"]]
        vals = df[col].values
        bars = ax.bar(labels, vals, color=[colors[m] for m in df["model"]], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("主模型与基线模型测试集指标对比", fontsize=13)
    out = os.path.join(fig_dir, "metrics_bar.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out}")


def fig_loss_curves(fig_dir, metrics_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    found = False
    for ax, model_name in zip(axes, ["meshgraphnet", "baseline"]):
        hist_path = os.path.join(metrics_dir, f"{model_name}_loss_history.csv")
        if not os.path.exists(hist_path):
            print(f"[跳过] 未找到 {hist_path}")
            continue
        found = True
        df = pd.read_csv(hist_path)
        ax.plot(df["epoch"], df["train_loss"], label="训练损失", lw=1.5)
        ax.plot(df["epoch"], df["val_loss"], label="验证损失", lw=1.5)
        best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
        ax.axvline(best_epoch, color="gray", ls=":", lw=1, label=f"最佳 epoch = {int(best_epoch)}")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("损失")
        ax.set_title(MODEL_LABELS[model_name])
        ax.grid(alpha=0.3)
        ax.legend()
    if not found:
        plt.close(fig)
        return
    fig.suptitle("训练 / 验证损失曲线", fontsize=13)
    out = os.path.join(fig_dir, "loss_curves.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out}")


def fig_structure_mesh(fig_dir, raw_dir, data_config_path):
    mesh_path = os.path.join(raw_dir, "mesh.npz")
    if not os.path.exists(mesh_path):
        print(f"[跳过] 未找到 {mesh_path}，请先运行 generate_dataset.py")
        return
    cfg = load_yaml(data_config_path)
    g = cfg["geometry"]
    W = g["width"] * 1000
    y_sub = g["substrate_thickness"] * 1000
    y_cu = y_sub + g["cu_thickness"] * 1000
    y_tim = y_cu + g["tim_thickness"] * 1000
    H = y_tim + g["die_thickness"] * 1000

    mesh = np.load(mesh_path)
    points, tris = mesh["points"] * 1000, mesh["tris"].T
    tri = mtri.Triangulation(points[0], points[1], triangles=tris)

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    layers = [
        (0, y_sub, "#c9b38c", "基板 Substrate"),
        (y_sub, y_cu, "#e88f4a", "铜散热层 Cu"),
        (y_cu, y_tim, "#9e9e9e", "TIM"),
        (y_tim, H, "#7f9fc8", "Die (硅)"),
    ]
    for y0, y1, color, label in layers:
        ax.fill_between([0, W], y0, y1, color=color, alpha=0.85, label=label)

    ax.triplot(tri, color="k", lw=0.25, alpha=0.45)

    # 示例热点区域（取采样范围中值）
    hs = cfg["heat_source"]
    c_frac = 0.5 * (hs["hotspot_center_frac"]["low"] + hs["hotspot_center_frac"]["high"])
    w_frac = 0.5 * (hs["hotspot_width_frac"]["low"] + hs["hotspot_width_frac"]["high"])
    hot_lo, hot_hi = (c_frac - w_frac / 2) * W, (c_frac + w_frac / 2) * W
    ax.plot([hot_lo, hot_hi, hot_hi, hot_lo, hot_lo], [y_tim, y_tim, H, H, y_tim], "r--", lw=2, label="热点区域（位置/宽度随机）")

    ax.annotate("顶部对流边界:  -k∂T/∂n = h·(T - T_amb)", xy=(W / 2, H), xytext=(W / 2, H + 1.8),
                ha="center", fontsize=11, arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red")
    ax.annotate("底部恒温边界:  T = T_amb", xy=(W / 2, 0), xytext=(W / 2, -2.2),
                ha="center", fontsize=11, arrowprops=dict(arrowstyle="->", color="tab:blue"), color="tab:blue")
    ax.text(-0.6, H / 2, "绝热", rotation=90, va="center", fontsize=10, color="gray")
    ax.text(W + 0.3, H / 2, "绝热", rotation=-90, va="center", fontsize=10, color="gray")

    ax.set_xlim(-1.5, W + 1.5)
    ax.set_ylim(-3.2, H + 3.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("二维芯片封装截面结构与规则三角形有限元网格", fontsize=13)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    out = os.path.join(fig_dir, "package_structure_mesh.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out}")


def main():
    parser = argparse.ArgumentParser(description="生成项目全部可视化图片")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures")
    parser.add_argument("--data_config", type=str, default="configs/data_config.yaml")
    parser.add_argument(
        "--figures", nargs="+", default=["all"],
        choices=["all", "field", "peak", "bars", "loss", "structure"],
        help="选择要生成的图片类型",
    )
    parser.add_argument("--samples", nargs="+", type=int, default=None, help="三联图使用的测试样本索引")
    args = parser.parse_args()

    setup_matplotlib()
    os.makedirs(args.fig_dir, exist_ok=True)
    figs = set(args.figures)
    run_all = "all" in figs

    if run_all or "structure" in figs:
        fig_structure_mesh(args.fig_dir, args.raw_dir, args.data_config)
    if run_all or "loss" in figs:
        fig_loss_curves(args.fig_dir, args.metrics_dir)
    if run_all or "bars" in figs:
        fig_metrics_bar(args.fig_dir, args.metrics_dir)
    if run_all or "field" in figs:
        fig_field_triptych(args.fig_dir, args.data_dir, args.ckpt_dir, args.samples)
    if run_all or "peak" in figs:
        fig_peak_scatter(args.fig_dir, args.data_dir, args.ckpt_dir)


if __name__ == "__main__":
    main()
