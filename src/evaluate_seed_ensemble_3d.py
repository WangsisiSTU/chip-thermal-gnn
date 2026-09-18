"""Evaluate a paired three-seed thermal ensemble for initialization robustness."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import checkpoint_normalization, evaluate_model, load_model_from_ckpt
from utils import get_device, load_json, load_processed_metadata, load_split, validate_data_contract


class SeedEnsemble(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    def forward(self, data):
        return torch.stack([member(data) for member in self.members], dim=0).mean(0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate paired multi-seed model averages")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt_base", default="outputs/3d/benchmark_corrected_multimesh")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--models", nargs="+", default=["baseline", "mgn_transolver"])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cross_mesh", action="store_true")
    args = parser.parse_args()
    device = get_device()
    metadata = load_processed_metadata(args.data_dir)
    test_set = load_split(args.data_dir, "test")
    test_info = load_json(os.path.join(args.data_dir, "test_info.json"))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        members, scales = [], []
        for seed in args.seeds:
            path = f"{args.ckpt_base}_seed{seed}/checkpoints/{name}_best.pt"
            model, checkpoint = load_model_from_ckpt(path, device)
            validate_data_contract(checkpoint, metadata, allow_output_stats_shift=args.cross_mesh)
            members.append(model)
            scales.append((checkpoint["dT_mean"], checkpoint["dT_std"]))
        if any(scale != scales[0] for scale in scales[1:]):
            raise ValueError("ensemble members use different output normalization")
        ensemble = SeedEnsemble(members).to(device).eval()
        mean, std = checkpoint_normalization({"dT_mean": scales[0][0], "dT_std": scales[0][1]}, device)
        cases, summary = evaluate_model(ensemble, test_set, test_info, mean, std, device,
                                        timing_repeats=3, benchmark_batch_size=4)
        with (output / f"{name}_ensemble_per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
        result = {"model": name, "seeds": args.seeds,
                  "total_parameters": sum(p.numel() for p in ensemble.parameters()), **summary}
        (output / f"{name}_ensemble_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, {key: round(summary[key], 4) for key in
                     ("overall_mae_K", "die_gradient_mae_mean_K_per_mm", "peak_abs_err_mean_K")})


if __name__ == "__main__":
    main()
