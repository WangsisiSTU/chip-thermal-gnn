"""Render source-backed figures for the paired 3D MGN+TRANS benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from evaluate import checkpoint_normalization, load_model_from_ckpt
from utils import get_device, load_processed_metadata, load_split, validate_data_contract


MODELS = ("baseline", "meshgraphnet", "mgn_transolver")
LABELS = {"baseline": "GraphSAGE", "meshgraphnet": "MeshGraphNet", "mgn_transolver": "MGN + physics-state attention"}
COLORS = {"baseline": "#3b6ea8", "meshgraphnet": "#e3a83b", "mgn_transolver": "#d6754b"}


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
        "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "savefig.dpi": 170,
    })


def box(ax, x, y, w, h, label, color):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                           edgecolor="#344054", facecolor=color, linewidth=1.1)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, linespacing=1.4)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                 color="#475467", linewidth=1.3))


def architecture(out: Path):
    fig, ax = plt.subplots(figsize=(14, 4.4))
    ax.set(xlim=(0, 14), ylim=(0, 5))
    ax.axis("off")
    steps = [
        (0.2, "3D FEM cases\nP1 tetra mesh", "#e6eef8"),
        (2.4, "Encode\n26 node / 5 edge", "#e6eef8"),
        (4.6, "3 MGN steps\nlocal messages", "#dceae3"),
        (6.8, "Weighted slice\n32 states", "#f8ead0"),
        (9.0, "2 global blocks\n4 attention heads", "#f8ead0"),
        (11.2, "Deslice / decode\nN nodal dT", "#f7e2dd"),
    ]
    for x, label, color in steps:
        box(ax, x, 2.2, 1.9, 1.2, label, color)
    for index in range(len(steps) - 1):
        arrow(ax, (steps[index][0] + 1.9, 2.8), (steps[index + 1][0], 2.8))
    ax.text(7.0, 4.25, "MGN + Transolver-inspired physics-state attention: 3D prototype",
            ha="center", fontsize=14, weight="bold")
    ax.text(7.0, 0.95, "Local graph messages carry mesh connectivity; small learned states exchange long-range thermal information.",
            ha="center", fontsize=10, color="#475467")
    fig.savefig(out / "architecture_prototype.png", bbox_inches="tight")
    plt.close(fig)


def geometry(out: Path, raw_dir: Path):
    raw = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    geom = raw["geometry"]
    mesh = np.load(raw_dir / "mesh.npz")
    pts = mesh["points"] * 1000
    xs, ys, zs = [np.unique(pts[i]) for i in range(3)]
    layers = [
        (0, 6, "Substrate", "#8aaed8"), (6, 9, "Cu", "#e3bc66"),
        (9, 10, "TIM", "#dc8d5d"), (10, 12, "Die", "#cc7193"),
    ]
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    for low, high, label, color in layers:
        corners = np.array([[x, z, y] for x in (0, 20) for y in (low, high) for z in (0, 16)])
        face_ids = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        faces = [corners[list(ids)] for ids in face_ids]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=0.14,
                                             edgecolors=color, linewidths=1.0))
        ax.text(20.7, 8, (low + high) / 2, label, fontsize=10, color="#344054")
    # Actual medium-mesh nodes and x/y edges on the front z=0 cut plane.
    front = np.isclose(pts[2], 0)
    ax.scatter(pts[0, front], pts[2, front], pts[1, front], s=2.6, color="#344054", alpha=0.75)
    for y in ys:
        ax.plot(xs, np.zeros_like(xs), np.full_like(xs, y), color="#7b8794", alpha=0.28, lw=0.55)
    for x in xs:
        ax.plot(np.full_like(ys, x), np.zeros_like(ys), ys, color="#7b8794", alpha=0.28, lw=0.55)
    ax.scatter([10], [8], [12], marker="*", s=95, color="#c54b40", label="Die hot spot")
    ax.set(xlabel="x [mm]", ylabel="z [mm]", zlabel="stack y [mm]",
           xlim=(0, 24), ylim=(0, 16), zlim=(0, 12),
           title=f"Layered 3D package and medium tetrahedral mesh | {raw['n_nodes']:,} nodes, {raw['n_tets']:,} tets")
    ax.set_box_aspect((20, 16, 12))
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper left")
    fig.savefig(out / "package_mesh_3d.png", bbox_inches="tight")
    plt.close(fig)


def evaluation_figures(out: Path, benchmark_dir: Path):
    summaries = {}
    for mesh in ("coarse", "medium", "fine"):
        summaries[mesh] = {
            model: json.loads((benchmark_dir / f"eval_{mesh}" / "metrics" / f"{model}_eval_summary.json").read_text(encoding="utf-8"))
            for model in MODELS
        }
    fig, ax = plt.subplots(figsize=(8.7, 5.0))
    nodes = np.array([819, 1859, 3315])
    for model in MODELS:
        values = [summaries[mesh][model]["overall_mae_K"] for mesh in ("coarse", "medium", "fine")]
        ax.plot(nodes, values, marker="o", ms=7, lw=2.0, label=LABELS[model], color=COLORS[model])
        for n, value in zip(nodes, values):
            ax.annotate(f"{value:.3f}", (n, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="Mesh nodes [count]", ylabel="Nodal MAE [K]", title="Paired test cases across mesh resolutions")
    ax.set_xticks(nodes)
    ax.set_ylim(0, 0.125)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9, frameon=False)
    fig.savefig(out / "accuracy_by_mesh.png", bbox_inches="tight")
    plt.close(fig)

    regimes = ("id", "covered_poor_cooling", "ood_high_power", "ood_poor_tim", "ood_poor_cooling")
    names = ("ID", "Covered low h", "OOD high power", "OOD poor TIM", "OOD low h")
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    x = np.arange(len(regimes))
    for index, model in enumerate(MODELS):
        s = summaries["medium"][model]["regime_breakdown"]
        values = [s[regime]["mae_K_mean"] for regime in regimes]
        ax.bar(x + (index - 1) * 0.25, values, 0.23, label=LABELS[model], color=COLORS[model])
    ax.set(xlabel="Test regime", ylabel="Mean per-case MAE [K]",
           title="Same 32 medium-mesh cases, separated by operating regime")
    ax.set_xticks(x, names)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.16)
    ax.legend(fontsize=9)
    fig.savefig(out / "regime_comparison.png", bbox_inches="tight")
    plt.close(fig)


def field_comparison(out: Path, benchmark_dir: Path, data_dir: Path, sample_index: int):
    device = get_device()
    data = load_split(str(data_dir), "test")[sample_index]
    metadata = load_processed_metadata(str(data_dir))
    test_info = json.loads((data_dir / "test_info.json").read_text(encoding="utf-8"))
    pos = data.pos.numpy() * 1000
    y_layers = np.unique(pos[:, 1])
    slice_y = y_layers[np.argmin(abs(y_layers - 11.0))]
    selected = np.isclose(pos[:, 1], slice_y)
    x, z = pos[selected, 0], pos[selected, 2]
    true = data.y.numpy()[selected]
    predictions = {}
    for model_name in MODELS:
        model, ckpt = load_model_from_ckpt(str(benchmark_dir / "checkpoints" / f"{model_name}_best.pt"), device)
        validate_data_contract(ckpt, metadata)
        mean, std = checkpoint_normalization(ckpt, device)
        with __import__("torch").no_grad():
            predictions[model_name] = (model(data.to(device)) * std + mean).cpu().numpy()[selected]
    common_max = max(float(true.max()), *(float(p.max()) for p in predictions.values()))
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.7), constrained_layout=True)
    fields = [("FEM reference", true), *(({"baseline": "GraphSAGE", "meshgraphnet": "MGN", "mgn_transolver": "MGN + TRANS"}[m], predictions[m]) for m in MODELS),
              ("Hybrid - FEM", predictions["mgn_transolver"] - true)]
    artists = []
    for index, ((title, field), ax) in enumerate(zip(fields, axes)):
        if index == 4:
            limit = max(float(np.max(np.abs(field))), 1e-3)
            artist = ax.tricontourf(x, z, field, levels=np.linspace(-limit, limit, 19), cmap="coolwarm", extend="both")
        else:
            artist = ax.tricontourf(x, z, field, levels=np.linspace(0, common_max + 1e-6, 19), cmap="inferno")
        ax.set(xlabel="x [mm]", ylabel="z [mm]" if index == 0 else None,
               title=title, aspect="equal")
        artists.append(artist)
    fig.colorbar(artists[0], ax=axes[:4], label="dT [K]", shrink=0.76)
    fig.colorbar(artists[4], ax=axes[4], label="Error [K]", shrink=0.76)
    fig.suptitle(f"Die-layer x-z field at y={slice_y:.1f} mm | test #{sample_index}: {test_info[sample_index]['regime']}", fontsize=11)
    fig.savefig(out / f"field_comparison_test{sample_index}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", default="outputs/3d/mgn_transolver_benchmark")
    parser.add_argument("--raw_dir", default="data/raw_3d_mgn_transolver_benchmark_medium")
    parser.add_argument("--data_dir", default="data/processed_3d_mgn_transolver_benchmark_medium")
    args = parser.parse_args()
    style()
    out = Path(args.benchmark_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    architecture(out)
    geometry(out, Path(args.raw_dir))
    evaluation_figures(out, Path(args.benchmark_dir))
    info = json.loads((Path(args.data_dir) / "test_info.json").read_text(encoding="utf-8"))
    for regime in ("id", "ood_poor_cooling"):
        index = next(i for i, row in enumerate(info) if row["regime"] == regime)
        field_comparison(out, Path(args.benchmark_dir), Path(args.data_dir), index)
    print(f"saved benchmark figures to {out}")


if __name__ == "__main__":
    main()
