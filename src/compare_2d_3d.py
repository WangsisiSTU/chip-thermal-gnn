"""Create a compact, reproducible 2D-vs-3D experiment comparison table."""
from __future__ import annotations

import argparse
import csv
import json
import os


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def weighted_ood_mae(result: dict) -> float | None:
    breakdown = result.get("regime_breakdown", {})
    rows = [v for name, v in breakdown.items() if name.startswith("ood_")]
    if not rows:
        return None
    total = sum(row["n_samples"] for row in rows)
    return sum(row["n_samples"] * row["mae_K_mean"] for row in rows) / total


def build_row(label: str, raw_dir: str, processed_dir: str, eval_path: str) -> dict:
    raw, processed, result = load_json(os.path.join(raw_dir, "metadata.json")), load_json(
        os.path.join(processed_dir, "metadata.json")
    ), load_json(eval_path)
    return {
        "dimension": label,
        "model": result["model"],
        "n_params": result["n_params"],
        "nodes": raw["n_nodes"],
        "elements": raw.get("n_tris", raw.get("n_tets")),
        "directed_edges": processed["n_edges"],
        "mean_fem_ms": result["mean_fem_solve_time_ms"],
        "mean_inference_ms": result["mean_inference_time_ms"],
        "mae_K": result["overall_mae_K"],
        "peak_abs_err_K": result["peak_abs_err_mean_K"],
        "ood_mae_K": weighted_ood_mae(result),
        "runtime_device": result.get("runtime_device", "unknown"),
        "cuda_peak_memory_MiB": result.get("cuda_peak_memory_MiB"),
        "process_rss_MiB": result.get("process_rss_MiB"),
    }


def _format(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown_table(rows: list[dict], path: str) -> None:
    columns = [
        ("dimension", "Dimension"),
        ("model", "Model"),
        ("n_params", "Parameters"),
        ("nodes", "Nodes"),
        ("directed_edges", "Directed edges"),
        ("mae_K", "MAE [K]"),
        ("peak_abs_err_K", "Peak error [K]"),
        ("ood_mae_K", "OOD MAE [K]"),
        ("mean_fem_ms", "FEM [ms]"),
        ("mean_inference_ms", "Inference [ms]"),
        ("process_rss_MiB", "Process RSS [MiB]"),
        ("cuda_peak_memory_MiB", "CUDA peak [MiB]"),
    ]
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_format(row[key]) for key, _ in columns) + " |" for row in rows]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join([header, divider, *body]) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare existing 2D and 3D model evaluations.")
    parser.add_argument("--model", default="baseline", choices=["baseline", "meshgraphnet"])
    parser.add_argument("--raw_2d", default="data/raw")
    parser.add_argument("--processed_2d", default="data/processed")
    parser.add_argument("--metrics_2d", default="outputs/metrics")
    parser.add_argument("--raw_3d", default="data/raw_3d_full")
    parser.add_argument("--processed_3d", default="data/processed_3d_full")
    parser.add_argument("--metrics_3d", default="outputs/3d/metrics")
    parser.add_argument("--out_dir", default="outputs/3d/metrics")
    args = parser.parse_args()

    eval_name = f"{args.model}_eval_summary.json"
    rows = [
        build_row("2D", args.raw_2d, args.processed_2d, os.path.join(args.metrics_2d, eval_name)),
        build_row("3D", args.raw_3d, args.processed_3d, os.path.join(args.metrics_3d, eval_name)),
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"comparison_2d_vs_3d_{args.model}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = os.path.join(args.out_dir, f"comparison_2d_vs_3d_{args.model}.md")
    write_markdown_table(rows, markdown_path)
    print(f"comparison saved: {csv_path}")
    print(f"markdown table saved: {markdown_path}")
    for row in rows:
        print(
            f"{row['dimension']} {row['model']}: nodes={row['nodes']}, edges={row['directed_edges']}, "
            f"MAE={row['mae_K']:.4f} K, OOD_MAE={row['ood_mae_K']:.4f} K"
        )


if __name__ == "__main__":
    main()
