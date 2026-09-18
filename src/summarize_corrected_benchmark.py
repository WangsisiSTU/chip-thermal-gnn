"""Paired seed-and-physical-case uncertainty for the corrected 3D benchmark."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def paired_bootstrap(values: np.ndarray, rng: np.random.Generator, repeats: int = 10000):
    """Resample model initialization and held-out physical cases independently."""
    if values.ndim != 2:
        raise ValueError("paired values must have shape (seeds, physical cases)")
    seeds, cases = values.shape
    sampled_seeds = rng.integers(0, seeds, size=(repeats, seeds))
    sampled_cases = rng.integers(0, cases, size=(repeats, cases))
    sampled = values[sampled_seeds[:, :, None], sampled_cases[:, None, :]].mean((1, 2))
    return {"mean_difference": float(values.mean()),
            "ci95": [float(x) for x in np.percentile(sampled, [2.5, 97.5])],
            "hybrid_wins_fraction": float((values < 0).mean())}


def summarize(base: Path, seeds: list[int], grids: list[str], repeats: int):
    rng = np.random.default_rng(2024)
    output = {"seeds": seeds, "grids": {}, "bootstrap_repeats": repeats,
              "models": ["baseline", "mgn_transolver"],
              "resampling_unit": "training seed and matched physical test case"}
    fields = ("mae_K", "die_gradient_mae_K_per_mm", "peak_abs_err_K", "die_field_mae_K")
    for grid in grids:
        seed_pairs = []
        summaries = []
        for seed in seeds:
            result_dir = Path(f"{base}_seed{seed}") / f"eval_{grid}" / "metrics"
            left = read_rows(result_dir / "baseline_per_sample_metrics.csv")
            right = read_rows(result_dir / "mgn_transolver_per_sample_metrics.csv")
            if len(left) != len(right) or not all(
                a["file"] == b["file"] and a["regime"] == b["regime"] for a, b in zip(left, right)
            ):
                raise ValueError(f"test cases are not paired: grid={grid}, seed={seed}")
            seed_pairs.append((left, right))
            summaries.append({
                "seed": seed,
                "baseline": json.loads((result_dir / "baseline_eval_summary.json").read_text(encoding="utf-8")),
                "mgn_transolver": json.loads((result_dir / "mgn_transolver_eval_summary.json").read_text(encoding="utf-8")),
            })
        paired = {}
        for field in fields:
            values = np.asarray([
                [float(b[field]) - float(a[field]) for a, b in zip(left, right)]
                for left, right in seed_pairs
            ])
            paired[field] = paired_bootstrap(values, rng, repeats)
        output["grids"][grid] = {"n_cases": len(seed_pairs[0][0]), "paired": paired,
                                "per_seed": summaries}
    return output


def main():
    parser = argparse.ArgumentParser(description="Summarize corrected paired 3D runs")
    parser.add_argument("--base", default="outputs/3d/benchmark_corrected_multimesh")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--grids", nargs="+", default=["coarse", "medium", "fine", "local"])
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--out", default="outputs/3d/benchmark_corrected_summary/paired_seed_case_bootstrap.json")
    args = parser.parse_args()
    result = summarize(Path(args.base), args.seeds, args.grids, args.repeats)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for grid, value in result["grids"].items():
        print(grid, {name: {"mean": round(stat["mean_difference"], 4),
                            "ci95": [round(x, 4) for x in stat["ci95"]]}
                     for name, stat in value["paired"].items()})


if __name__ == "__main__":
    main()
