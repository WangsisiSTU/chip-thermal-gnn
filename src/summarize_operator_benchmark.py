"""Paired physical-case bootstrap for the expanded FEM-operator pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/3d/operator_expanded_seed3107")
    parser.add_argument("--grids", nargs="+", default=["medium", "fine", "local_offset"])
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--out", default="outputs/3d/operator_expanded_seed3107/paired_case_bootstrap.json")
    args = parser.parse_args()
    rng = np.random.default_rng(3107)
    fields = ("mae_K", "die_gradient_mae_K_per_mm", "peak_abs_err_K",
              "fem_energy_residual_relative", "interface_flux_relative_l2")
    result = {"training_seeds": [3107], "bootstrap_repeats": args.repeats,
              "resampling_unit": "matched physical test case; training-seed uncertainty excluded", "grids": {}}
    for grid in args.grids:
        root = Path(args.base) / f"eval_{grid}" / "metrics"
        left = read_rows(root / "baseline_per_sample_metrics.csv")
        right = read_rows(root / "mgn_transolver_per_sample_metrics.csv")
        if len(left) != len(right) or not all(
            a["file"] == b["file"] and a["regime"] == b["regime"] for a, b in zip(left, right)
        ):
            raise ValueError(f"unpaired test cases on {grid}")
        stats = {}
        for field in fields:
            diff = np.asarray([float(b[field]) - float(a[field]) for a, b in zip(left, right)])
            sample = diff[rng.integers(0, len(diff), size=(args.repeats, len(diff)))].mean(1)
            stats[field] = {
                "mean_difference": float(diff.mean()),
                "ci95": [float(x) for x in np.percentile(sample, [2.5, 97.5])],
                "hybrid_wins_fraction": float((diff < 0).mean()),
            }
        result["grids"][grid] = {"n_cases": len(left), "paired": stats}
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
