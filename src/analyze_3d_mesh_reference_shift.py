"""Quantify FEM label shift when the same physical test cases change mesh.

The medium-mesh P1 nodal field is trilinearly interpolated to each target
mesh's nodes. This is a diagnostic, not a continuum ground-truth error bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def tensor_field(points: np.ndarray, values: np.ndarray):
    axes = [np.unique(points[axis]) for axis in range(3)]
    indices = [np.searchsorted(axes[axis], points[axis]) for axis in range(3)]
    field = np.full(tuple(len(axis) for axis in axes), np.nan)
    field[tuple(indices)] = values
    if np.isnan(field).any():
        raise ValueError("mesh nodes do not form a complete tensor grid")
    return axes, field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_prefix", default="data/raw_3d_mgn_transolver_benchmark_")
    parser.add_argument("--out", default="outputs/3d/mgn_transolver_benchmark/metrics/mesh_reference_shift.json")
    args = parser.parse_args()

    datasets = {}
    for name in ("coarse", "medium", "fine"):
        folder = Path(args.raw_prefix + name)
        datasets[name] = {
            "folder": folder,
            "metadata": json.loads((folder / "metadata.json").read_text(encoding="utf-8")),
            "points": np.load(folder / "mesh.npz")["points"],
        }
    base = datasets["medium"]["metadata"]["samples"]
    if not all(
        len(base) == len(datasets[name]["metadata"]["samples"])
        and all(
            (a["split"], a["regime"], a["case"]) == (b["split"], b["regime"], b["case"])
            for a, b in zip(base, datasets[name]["metadata"]["samples"])
        )
        for name in ("coarse", "fine")
    ):
        raise ValueError("mesh datasets do not have paired physical cases")

    out = {"method": "medium_P1_nodal_trilinear_interpolation_to_target_nodes", "meshes": {}}
    for name in ("coarse", "fine"):
        target = datasets[name]
        records = []
        for sample in base:
            if sample["split"] != "test":
                continue
            path = sample["file"]
            medium_field = np.load(datasets["medium"]["folder"] / path)["T"]
            target_field = np.load(target["folder"] / path)["T"]
            axes, tensor = tensor_field(datasets["medium"]["points"], medium_field)
            medium_at_target = RegularGridInterpolator(axes, tensor)(target["points"].T)
            difference = medium_at_target - target_field
            records.append({
                "file": path,
                "regime": sample["regime"],
                "mae_K": float(np.mean(np.abs(difference))),
                "relative_l2": float(np.linalg.norm(difference) / (np.linalg.norm(target_field - sample["case"]["t_ambient"]) + 1e-12)),
                "peak_difference_K": float(abs(medium_at_target.max() - target_field.max())),
            })
        out["meshes"][name] = {
            "n_test": len(records),
            "mean_mae_K": float(np.mean([record["mae_K"] for record in records])),
            "mean_relative_l2": float(np.mean([record["relative_l2"] for record in records])),
            "mean_peak_difference_K": float(np.mean([record["peak_difference_K"] for record in records])),
            "per_case": records,
        }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, stats in out["meshes"].items():
        print(f"{name}: n={stats['n_test']}, FEM label shift MAE={stats['mean_mae_K']:.5f} K")


if __name__ == "__main__":
    main()
