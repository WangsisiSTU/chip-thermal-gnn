"""Paired FEM/GraphSAGE/MGN+TRANS die-layer cloud plots with common color scales."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import checkpoint_normalization, load_model_from_ckpt
from utils import get_device, load_processed_metadata, load_split, validate_data_contract


def plot_case(data_dir: Path, ckpt_dir: Path, index: int, output: Path):
    device = get_device()
    samples = load_split(str(data_dir), "test")
    data = samples[index]
    metadata = load_processed_metadata(str(data_dir))
    info = json.loads((data_dir / "test_info.json").read_text(encoding="utf-8"))[index]
    positions = data.pos.numpy() * 1000.0
    y_values = np.unique(positions[:, 1])
    y_plane = y_values[np.argmin(np.abs(y_values - 11.0))]
    on_plane = np.isclose(positions[:, 1], y_plane, atol=1e-6)
    x, z = positions[on_plane, 0], positions[on_plane, 2]
    truth = data.y.numpy()[on_plane]
    fields = {"FEM": truth}
    for name, label in (("baseline", "GraphSAGE"), ("mgn_transolver", "MGN+TRANS")):
        model, ckpt = load_model_from_ckpt(str(ckpt_dir / f"{name}_best.pt"), device)
        validate_data_contract(ckpt, metadata, allow_output_stats_shift=True)
        mean, std = checkpoint_normalization(ckpt, device)
        with torch.no_grad():
            fields[label] = (model(data.to(device)) * std + mean).cpu().numpy()[on_plane]
    limit = max(float(np.max(field)) for field in fields.values())
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.8), constrained_layout=True)
    artists = []
    for ax, label in zip(axes[:3], fields):
        artist = ax.tricontourf(x, z, fields[label], levels=np.linspace(0, limit + 1e-6, 20), cmap="inferno")
        ax.set(title=label, xlabel="x [mm]", aspect="equal")
        artists.append(artist)
    axes[0].set_ylabel("z [mm]")
    errors = (fields["GraphSAGE"] - truth, fields["MGN+TRANS"] - truth)
    err_limit = max(float(np.max(np.abs(error))) for error in errors) + 1e-6
    for ax, error, title in zip(axes[3:], errors, ("GraphSAGE - FEM", "MGN+TRANS - FEM")):
        artist = ax.tricontourf(x, z, error, levels=np.linspace(-err_limit, err_limit, 20), cmap="coolwarm", extend="both")
        ax.set(title=title, xlabel="x [mm]", aspect="equal")
    fig.colorbar(artists[0], ax=axes[:3], label="dT [K]", shrink=0.72)
    fig.colorbar(artist, ax=axes[3:], label="Error [K]", shrink=0.72)
    fig.suptitle(f"Paired die-layer cloud | y={y_plane:.2f} mm | case #{index} {info['regime']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot corrected thermal benchmark clouds")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(plot_case(Path(args.data_dir), Path(args.ckpt_dir), args.index, Path(args.out)))


if __name__ == "__main__":
    main()
