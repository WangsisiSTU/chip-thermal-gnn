"""Slice-based cloud plots for 3D tetrahedral FEM/GNN temperature fields."""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import checkpoint_normalization, load_model_from_ckpt
from utils import get_device, load_processed_metadata, load_split, validate_data_contract


PLANE_SPECS = {
    1: ("x", "z", "y", (0, 2)),
    0: ("y", "z", "x", (1, 2)),
    2: ("x", "y", "z", (0, 1)),
}


def load_test_info(data_dir: str) -> list[dict]:
    with open(os.path.join(data_dir, "test_info.json"), "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def predict_dT(model, data, dT_mean, dT_std, device):
    prediction = model(data.to(device))
    return (prediction * dT_std + dT_mean).cpu().numpy()


def select_showcases(test_set, test_info: list[dict]) -> list[int]:
    dT_max = np.asarray([float(item.dT_max_true.item()) for item in test_set])
    id_indices = [idx for idx, info in enumerate(test_info) if info["regime"] == "id"]
    ood_indices = [idx for idx, info in enumerate(test_info) if info["regime"] != "id"]
    selections = []
    if id_indices:
        selections.append(sorted(id_indices, key=lambda idx: dT_max[idx])[len(id_indices) // 2])
    if ood_indices:
        selections.append(max(ood_indices, key=lambda idx: dT_max[idx]))
    return selections


def _slice_indices(pos: np.ndarray, axis: int, target: float) -> tuple[np.ndarray, float]:
    coordinates = np.unique(pos[:, axis])
    selected = float(coordinates[np.argmin(np.abs(coordinates - target))])
    return np.flatnonzero(np.isclose(pos[:, axis], selected)), selected


def _plot_field(ax, pos, values, axis, target, title, is_error: bool):
    point_indices, actual_target = _slice_indices(pos, axis, target)
    first_name, second_name, normal_name, remaining = PLANE_SPECS[axis]
    points = pos[point_indices]
    x_values = points[:, remaining[0]] * 1000.0
    y_values = points[:, remaining[1]] * 1000.0
    field = values[point_indices]
    tri = mtri.Triangulation(x_values, y_values)
    if is_error:
        limit = float(np.max(np.abs(field))) + 1e-12
        artist = ax.tripcolor(tri, field, shading="gouraud", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        artist = ax.tripcolor(tri, field, shading="gouraud", cmap="inferno")
    ax.set_title(f"{title}\n{normal_name}={actual_target * 1000:.2f} mm", fontsize=9)
    ax.set_xlabel(f"{first_name} [mm]")
    ax.set_ylabel(f"{second_name} [mm]")
    ax.set_aspect("equal")
    return artist


def plot_sample(fig_dir, model_name, sample_index, info, data, prediction, geometry):
    true = data.y.cpu().numpy()
    error = prediction - true
    pos = data.pos.cpu().numpy()
    targets = {
        1: geometry["substrate_thickness"] + geometry["cu_thickness"] + geometry["tim_thickness"] + geometry["die_thickness"] / 2.0,
        0: geometry["width"] / 2.0,
        2: geometry["depth"] / 2.0,
    }
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    columns = [(true, "FEM dT", False), (prediction, "GNN dT", False), (error, "Prediction - FEM", True)]
    for row, axis in enumerate((1, 0, 2)):
        for col, (field, label, is_error) in enumerate(columns):
            artist = _plot_field(axes[row, col], pos, field, axis, targets[axis], label, is_error)
            fig.colorbar(artist, ax=axes[row, col], label="dT [K]", shrink=0.85)
    fig.suptitle(
        f"3D slice clouds | {model_name} | sample {sample_index} | {info['regime']} | "
        f"FEM peak={true.max():.3f} K, prediction peak={prediction.max():.3f} K",
        fontsize=12,
    )
    out = os.path.join(fig_dir, f"slice_clouds_{model_name}_sample{sample_index}_{info['regime']}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Create x-z, x-y and y-z cloud slices for 3D test fields.")
    parser.add_argument("--model", default="baseline", choices=["baseline", "meshgraphnet", "mgn_transolver"])
    parser.add_argument("--data_dir", default="data/processed_3d_full")
    parser.add_argument("--raw_dir", default="data/raw_3d_full")
    parser.add_argument("--ckpt_dir", default="outputs/3d/checkpoints")
    parser.add_argument("--fig_dir", default="outputs/3d/figures")
    parser.add_argument("--samples", type=int, nargs="+", default=None)
    args = parser.parse_args()

    device = get_device()
    metadata = load_processed_metadata(args.data_dir)
    test_set = load_split(args.data_dir, "test")
    test_info = load_test_info(args.data_dir)
    with open(os.path.join(args.raw_dir, "metadata.json"), "r", encoding="utf-8") as f:
        geometry = json.load(f)["geometry"]
    model, checkpoint = load_model_from_ckpt(os.path.join(args.ckpt_dir, f"{args.model}_best.pt"), device)
    validate_data_contract(checkpoint, metadata)
    dT_mean, dT_std = checkpoint_normalization(checkpoint, device)
    indices = args.samples if args.samples is not None else select_showcases(test_set, test_info)
    os.makedirs(args.fig_dir, exist_ok=True)
    for index in indices:
        if index < 0 or index >= len(test_set):
            raise IndexError(f"test sample index out of range: {index}")
        plot_sample(
            args.fig_dir, args.model, index, test_info[index], test_set[index],
            predict_dT(model, test_set[index], dT_mean, dT_std, device), geometry,
        )


if __name__ == "__main__":
    main()
