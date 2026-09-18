"""Paired case-bootstrap comparison of three-mesh benchmark model errors.

Intervals resample physical test cases only. They do not include training-seed
variation and should not be interpreted as model-stability confidence bounds.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def paired_stats(hybrid: np.ndarray, comparator: np.ndarray, rng: np.random.Generator, reps: int) -> dict:
    difference = hybrid - comparator
    draws = rng.integers(0, len(difference), size=(reps, len(difference)))
    resampled_means = difference[draws].mean(axis=1)
    return {
        "n_cases": len(difference),
        "mean_hybrid_minus_comparator_K": float(difference.mean()),
        "paired_case_bootstrap_95pct_interval_K": [float(x) for x in np.quantile(resampled_means, [0.025, 0.975])],
        "hybrid_lower_error_case_fraction": float(np.mean(difference < 0)),
        "resampled_mean_nonnegative_fraction": float(np.mean(resampled_means >= 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", default="outputs/3d/mgn_transolver_benchmark")
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=240915)
    args = parser.parse_args()
    if args.reps < 100:
        raise ValueError("bootstrap reps must be at least 100")
    directory = Path(args.benchmark_dir)
    result = {"method": "paired_physical_case_bootstrap_on_nodal_mae", "reps": args.reps,
              "seed": args.seed, "scope": "test-case sampling only; one training seed", "meshes": {}}
    rng = np.random.default_rng(args.seed)
    for mesh in ("coarse", "medium", "fine"):
        metric_dir = directory / f"eval_{mesh}" / "metrics"
        records = {model: rows(metric_dir / f"{model}_per_sample_metrics.csv")
                   for model in ("baseline", "meshgraphnet", "mgn_transolver")}
        ids = [(r["file"], r["regime"]) for r in records["mgn_transolver"]]
        if not all(ids == [(r["file"], r["regime"]) for r in values] for values in records.values()):
            raise ValueError(f"test samples are not paired for mesh {mesh}")
        hybrid = np.asarray([float(r["mae_K"]) for r in records["mgn_transolver"]])
        mesh_result = {"n_test": len(ids), "comparisons": {}}
        for comparator in ("baseline", "meshgraphnet"):
            other = np.asarray([float(r["mae_K"]) for r in records[comparator]])
            mesh_result["comparisons"][comparator] = paired_stats(hybrid, other, rng, args.reps)
        result["meshes"][mesh] = mesh_result
    out = directory / "metrics" / "paired_case_bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for mesh, item in result["meshes"].items():
        for name, comparison in item["comparisons"].items():
            low, high = comparison["paired_case_bootstrap_95pct_interval_K"]
            print(f"{mesh} hybrid-{name}: {comparison['mean_hybrid_minus_comparator_K']:.4f} K, bootstrap interval [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    main()
