"""
评估脚本：在测试集上评估主模型（MeshGraphNet）与基线模型（GCN/GraphSAGE），
输出节点温度 MAE、相对 L2 误差、峰值温度绝对误差、R²，以及 FEM 求解耗时与
模型推理耗时对比，结果保存到 outputs/metrics/。

用法：
    python src/evaluate.py --data_dir data/processed --ckpt_dir outputs/checkpoints --out_dir outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# Windows + Anaconda 下 torch 与 MKL 可能各自加载一份 OpenMP 运行时，导致 OMP Error #15；
# 该环境变量是官方文档提及的常用规避方式，需在导入 torch/numpy 前设置。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import psutil
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model
from utils import get_device, load_json, load_processed_metadata, load_split, validate_data_contract


def checkpoint_normalization(ckpt, device):
    """Decode predictions using the training checkpoint's normalization."""
    mean = torch.tensor(ckpt["dT_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(ckpt["dT_std"], dtype=torch.float32, device=device)
    if mean.numel() != 1 or std.numel() != 1:
        raise ValueError("Checkpoint normalization must be scalar")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or std.item() <= 0:
        raise ValueError("Checkpoint normalization must have a finite mean and positive finite std")
    return mean, std


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["node_in_dim"], ckpt["edge_in_dim"], ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_model(
    model,
    test_set,
    test_info,
    dT_mean,
    dT_std,
    device,
    timing_repeats: int = 5,
    benchmark_batch_size: int = 4,
):
    """Evaluate fields and measure warmed single-sample and mini-batch inference."""
    if not test_set:
        raise ValueError("Cannot evaluate an empty test set")
    if timing_repeats < 1 or benchmark_batch_size < 1:
        raise ValueError("timing_repeats and benchmark_batch_size must be positive")

    per_sample = []
    all_true, all_pred, all_weights = [], [], []
    process = psutil.Process()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Exclude one-off dispatch and lazy-kernel costs from all reported timings.
    model(test_set[0].to(device))
    _synchronize(device)

    for i, data in enumerate(test_set):
        data = data.to(device)
        _synchronize(device)
        t0 = time.perf_counter()
        for _ in range(timing_repeats):
            pred_norm = model(data)
        _synchronize(device)
        infer_time = (time.perf_counter() - t0) / timing_repeats

        pred_dT = (pred_norm * dT_std + dT_mean).cpu().numpy()
        true_dT = data.y.cpu().numpy()
        weights = data.node_volume.cpu().numpy() if hasattr(data, "node_volume") else np.ones_like(true_dT)
        mae = float(np.mean(np.abs(pred_dT - true_dT)))
        volume_mae = float(np.sum(weights * np.abs(pred_dT - true_dT)) / np.sum(weights))
        rel_l2 = float(np.linalg.norm(pred_dT - true_dT) / (np.linalg.norm(true_dT) + 1e-12))
        volume_rel_l2 = float(np.sqrt(np.sum(weights * (pred_dT - true_dT) ** 2)) /
                              (np.sqrt(np.sum(weights * true_dT ** 2)) + 1e-12))
        peak_true = float(true_dT.max())
        peak_pred = float(pred_dT.max())

        info = test_info[i]
        per_sample.append(
            {
                "index": i,
                "file": info["file"],
                "regime": info["regime"],
                "mae_K": mae,
                "volume_mae_K": volume_mae,
                "rel_l2": rel_l2,
                "volume_rel_l2": volume_rel_l2,
                "peak_true_K": peak_true,
                "peak_pred_K": peak_pred,
                "peak_abs_err_K": abs(peak_pred - peak_true),
                "inference_time_s": infer_time,
                "fem_solve_time_s": float(data.solve_time_s.item()),
            }
        )
        if all(getattr(data, name, None) is not None for name in ("tet_index", "tet_grad_phi", "tet_volume", "tet_die")):
            corners = data.tet_index.long()
            grad_phi = data.tet_grad_phi
            pred_t = pred_norm * dT_std + dT_mean
            pred_grad = (pred_t[corners].transpose(0, 1).unsqueeze(-1) * grad_phi).sum(1)
            true_grad = (data.y[corners].transpose(0, 1).unsqueeze(-1) * grad_phi).sum(1)
            die = data.tet_die.bool()
            die_vol = data.tet_volume[die]
            grad_err = torch.linalg.vector_norm(pred_grad[die] - true_grad[die], dim=-1)
            per_sample[-1]["die_gradient_mae_K_per_mm"] = float(
                (die_vol * grad_err).sum().item() / die_vol.sum().item() / 1000.0
            )
            per_sample[-1]["die_gradient_rel_l2"] = float(
                torch.sqrt((die_vol * grad_err.square()).sum()).item() /
                (torch.sqrt((die_vol * true_grad[die].square().sum(-1)).sum()).item() + 1e-12)
            )
            die_node = torch.zeros(data.y.numel(), dtype=torch.bool, device=device)
            die_node[corners[:, die].reshape(-1)] = True
            per_sample[-1]["die_field_mae_K"] = float((pred_t[die_node] - data.y[die_node]).abs().mean().item())
        if getattr(data, "pos", None) is not None:
            p = data.pos
            per_sample[-1]["peak_location_error_mm"] = float(
                torch.linalg.vector_norm(p[pred_norm.argmax()] - p[data.y.argmax()]).item() * 1000.0
            )
        if all(getattr(data, name, None) is not None for name in (
            "fem_operator_index", "fem_operator_value", "fem_operator_diag",
            "fem_source_load", "fem_free_node",
        )):
            row, col = data.fem_operator_index.long()
            pred_t = pred_norm * dT_std + dT_mean
            residual = torch.zeros_like(pred_t)
            residual.scatter_add_(0, row, data.fem_operator_value * pred_t[col])
            residual = (residual - data.fem_source_load) / data.fem_operator_diag.clamp_min(1e-12)
            free = data.fem_free_node.bool()
            rms = torch.sqrt(residual[free].square().mean())
            rhs_scaled = data.fem_source_load / data.fem_operator_diag.clamp_min(1e-12)
            signal = torch.sqrt(rhs_scaled[free].square().mean()).clamp_min(1e-12)
            per_sample[-1]["fem_energy_residual_rms_K"] = float(rms.item())
            per_sample[-1]["fem_energy_residual_relative"] = float((rms / signal).item())
        if all(getattr(data, name, None) is not None for name in (
            "interface_node_index", "interface_normal", "interface_area",
            "interface_grad_phi_left", "interface_grad_phi_right",
            "interface_k_left", "interface_k_right",
        )) and data.interface_node_index.shape[1] > 0:
            nodes = data.interface_node_index.long()

            def interface_flux(values):
                left_grad = (values[nodes[:4]].transpose(0, 1).unsqueeze(-1) *
                             data.interface_grad_phi_left).sum(1)
                right_grad = (values[nodes[4:]].transpose(0, 1).unsqueeze(-1) *
                              data.interface_grad_phi_right).sum(1)
                left = data.interface_k_left * (left_grad * data.interface_normal).sum(-1)
                right = data.interface_k_right * (right_grad * data.interface_normal).sum(-1)
                return left, right

            pred_left, pred_right = interface_flux(pred_norm)
            true_left, true_right = interface_flux((data.y - dT_mean) / dT_std)
            error = ((pred_left - pred_right) - (true_left - true_right)).square()
            signal = 0.5 * (true_left.square() + true_right.square())
            per_sample[-1]["interface_flux_relative_l2"] = float(torch.sqrt(
                (data.interface_area * error).sum() /
                (data.interface_area * signal).sum().clamp_min(1e-12)
            ).item())
        all_true.append(true_dT)
        all_pred.append(pred_dT)
        all_weights.append(weights)

    all_true_cat = np.concatenate(all_true)
    all_pred_cat = np.concatenate(all_pred)
    all_weights_cat = np.concatenate(all_weights)
    overall_mae = float(np.mean(np.abs(all_pred_cat - all_true_cat)))
    overall_rel_l2 = float(np.linalg.norm(all_pred_cat - all_true_cat) / (np.linalg.norm(all_true_cat) + 1e-12))
    ss_res = float(np.sum((all_pred_cat - all_true_cat) ** 2))
    ss_tot = float(np.sum((all_true_cat - all_true_cat.mean()) ** 2))
    peak_errs = [sample["peak_abs_err_K"] for sample in per_sample]

    summary = {
        "overall_mae_K": overall_mae,
        "overall_volume_mae_K": float(np.sum(all_weights_cat * np.abs(all_pred_cat - all_true_cat)) / np.sum(all_weights_cat)),
        "overall_rel_l2": overall_rel_l2,
        "overall_volume_rel_l2": float(np.sqrt(np.sum(all_weights_cat * (all_pred_cat - all_true_cat) ** 2)) /
                                         (np.sqrt(np.sum(all_weights_cat * all_true_cat ** 2)) + 1e-12)),
        "overall_r2": 1.0 - ss_res / (ss_tot + 1e-12),
        "peak_abs_err_mean_K": float(np.mean(peak_errs)),
        "peak_abs_err_std_K": float(np.std(peak_errs)),
        "mean_inference_time_ms": float(np.mean([sample["inference_time_s"] for sample in per_sample]) * 1000),
        "mean_fem_solve_time_ms": float(np.mean([sample["fem_solve_time_s"] for sample in per_sample]) * 1000),
        "timing_repeats": timing_repeats,
        "benchmark_batch_size": benchmark_batch_size,
    }
    for field, output in (("die_gradient_mae_K_per_mm", "die_gradient_mae_mean_K_per_mm"),
                          ("die_gradient_rel_l2", "die_gradient_rel_l2_mean"),
                          ("die_field_mae_K", "die_field_mae_mean_K"),
                          ("peak_location_error_mm", "peak_location_error_mean_mm"),
                          ("fem_energy_residual_rms_K", "fem_energy_residual_rms_mean_K"),
                          ("fem_energy_residual_relative", "fem_energy_residual_relative_mean"),
                          ("interface_flux_relative_l2", "interface_flux_relative_l2_mean")):
        if field in per_sample[0]:
            summary[output] = float(np.mean([sample[field] for sample in per_sample]))
    summary["per_case_mae_p95_K"] = float(np.percentile([sample["mae_K"] for sample in per_sample], 95))
    summary["per_case_peak_error_p95_K"] = float(np.percentile(peak_errs, 95))
    summary["speedup_vs_fem"] = summary["mean_fem_solve_time_ms"] / max(summary["mean_inference_time_ms"], 1e-9)

    batches = [
        Batch.from_data_list(list(test_set[start:start + benchmark_batch_size])).to(device)
        for start in range(0, len(test_set), benchmark_batch_size)
    ]
    for batch in batches:
        model(batch)
    _synchronize(device)
    t0 = time.perf_counter()
    for _ in range(timing_repeats):
        for batch in batches:
            model(batch)
    _synchronize(device)
    batched_total = time.perf_counter() - t0
    summary["batched_inference_time_per_sample_ms"] = float(batched_total / (len(test_set) * timing_repeats) * 1000)
    summary["speedup_vs_fem_batched"] = summary["mean_fem_solve_time_ms"] / max(
        summary["batched_inference_time_per_sample_ms"], 1e-9
    )
    summary["runtime_device"] = str(device)
    summary["process_rss_MiB"] = process.memory_info().rss / 2 ** 20
    summary["cuda_peak_memory_MiB"] = (
        torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
    )
    return per_sample, summary


def regime_breakdown(per_sample):
    regimes = {}
    for s in per_sample:
        regimes.setdefault(s["regime"], []).append(s)
    out = {}
    for regime, samples in regimes.items():
        maes = [s["mae_K"] for s in samples]
        rel_l2s = [s["rel_l2"] for s in samples]
        peak_errs = [s["peak_abs_err_K"] for s in samples]
        out[regime] = {
            "n_samples": len(samples),
            "mae_K_mean": float(np.mean(maes)),
            "volume_mae_K_mean": float(np.mean([s["volume_mae_K"] for s in samples])),
            "rel_l2_mean": float(np.mean(rel_l2s)),
            "volume_rel_l2_mean": float(np.mean([s["volume_rel_l2"] for s in samples])),
            "peak_abs_err_K_mean": float(np.mean(peak_errs)),
        }
        if "die_gradient_mae_K_per_mm" in samples[0]:
            out[regime]["die_gradient_mae_K_per_mm_mean"] = float(np.mean(
                [sample["die_gradient_mae_K_per_mm"] for sample in samples]
            ))
    return out


def main():
    parser = argparse.ArgumentParser(description="在测试集上评估主模型与基线模型")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--cpu_threads", type=int, default=None, help="limit CPU worker threads during timing")
    parser.add_argument("--timing_repeats", type=int, default=5, help="warmed forward passes per single-sample timing")
    parser.add_argument("--benchmark_batch_size", type=int, default=4, help="graphs per warmed throughput batch")
    parser.add_argument("--cross_mesh", action="store_true", help="allow mesh-dependent target statistics to differ; checkpoint normalization is still used")
    parser.add_argument(
        "--models", nargs="+", choices=["meshgraphnet", "baseline", "mgn_transolver",
                                      "mgn_global_pool", "mgn_transolver_slice_only",
                                      "mgn_transolver_adaptive", "mgn_transolver_adaptive_gumbel"],
        default=["meshgraphnet", "baseline"],
        help="models to evaluate; use one name for a focused timing run",
    )
    args = parser.parse_args()

    device = get_device()
    if device.type == "cpu" and args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("cpu thread count must be positive")
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        print(f"CPU threads: {torch.get_num_threads()}")
    print(f"使用设备: {device}")

    test_set = load_split(args.data_dir, "test")
    test_info = load_json(os.path.join(args.data_dir, "test_info.json"))
    processed_metadata = load_processed_metadata(args.data_dir)

    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    model_files = {name: f"{name}_best.pt" for name in (
        "meshgraphnet", "baseline", "mgn_transolver", "mgn_global_pool",
        "mgn_transolver_slice_only", "mgn_transolver_adaptive", "mgn_transolver_adaptive_gumbel",
    )}
    comparison_rows = []

    for model_name in args.models:
        ckpt_file = model_files[model_name]
        ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
        if not os.path.exists(ckpt_path):
            print(f"警告: 未找到权重 {ckpt_path}，跳过 {model_name}")
            continue
        model, ckpt = load_model_from_ckpt(ckpt_path, device)
        validate_data_contract(ckpt, processed_metadata, allow_output_stats_shift=args.cross_mesh)
        dT_mean, dT_std = checkpoint_normalization(ckpt, device)
        per_sample, summary = evaluate_model(
            model, test_set, test_info, dT_mean, dT_std, device,
            timing_repeats=args.timing_repeats, benchmark_batch_size=args.benchmark_batch_size,
        )
        breakdown = regime_breakdown(per_sample)

        per_sample_path = os.path.join(metrics_dir, f"{model_name}_per_sample_metrics.csv")
        with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample)

        result = {
            "model": model_name,
            "n_params": sum(p.numel() for p in model.parameters()),
            "best_epoch": ckpt.get("epoch"),
            **summary,
            "regime_breakdown": breakdown,
        }
        with open(os.path.join(metrics_dir, f"{model_name}_eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n== {model_name} ==")
        print(f"参数量: {result['n_params']}")
        print(f"整体 MAE: {summary['overall_mae_K']:.4f} K | 相对L2: {summary['overall_rel_l2']:.4f} | "
              f"R2: {summary['overall_r2']:.4f}")
        print(f"峰值温度绝对误差: {summary['peak_abs_err_mean_K']:.4f} +/- {summary['peak_abs_err_std_K']:.4f} K")
        print(f"平均推理耗时(单样本): {summary['mean_inference_time_ms']:.4f} ms | "
              f"平均FEM求解耗时: {summary['mean_fem_solve_time_ms']:.4f} ms | "
              f"加速比: {summary['speedup_vs_fem']:.2f}x")
        print(f"批量推理摊销耗时: {summary['batched_inference_time_per_sample_ms']:.4f} ms/样本 | "
              f"批量加速比: {summary['speedup_vs_fem_batched']:.2f}x")
        for regime, stat in breakdown.items():
            print(f"  [{regime}] n={stat['n_samples']}, MAE={stat['mae_K_mean']:.4f} K, "
                  f"相对L2={stat['rel_l2_mean']:.4f}, 峰值误差={stat['peak_abs_err_K_mean']:.4f} K")

        comparison_rows.append(
            {
                "model": model_name,
                "n_params": result["n_params"],
                "MAE_K": summary["overall_mae_K"],
                "VolumeMAE_K": summary["overall_volume_mae_K"],
                "RelativeL2": summary["overall_rel_l2"],
                "VolumeRelativeL2": summary["overall_volume_rel_l2"],
                "PeakAbsErr_K": summary["peak_abs_err_mean_K"],
                "R2": summary["overall_r2"],
                "MeanInferenceTime_ms": summary["mean_inference_time_ms"],
                "BatchedInferenceTime_ms": summary["batched_inference_time_per_sample_ms"],
                "MeanFEMSolveTime_ms": summary["mean_fem_solve_time_ms"],
                "SpeedupVsFEM": summary["speedup_vs_fem"],
                "SpeedupVsFEM_Batched": summary["speedup_vs_fem_batched"],
                "DieGradientMAE_K_per_mm": summary.get("die_gradient_mae_mean_K_per_mm"),
                "DieFieldMAE_K": summary.get("die_field_mae_mean_K"),
                "PeakLocationError_mm": summary.get("peak_location_error_mean_mm"),
                "FEMEnergyResidualRMS_K": summary.get("fem_energy_residual_rms_mean_K"),
                "FEMEnergyResidualRelative": summary.get("fem_energy_residual_relative_mean"),
                "InterfaceFluxRelativeL2": summary.get("interface_flux_relative_l2_mean"),
                "PerCaseMAE_P95_K": summary["per_case_mae_p95_K"],
                "PerCasePeakError_P95_K": summary["per_case_peak_error_p95_K"],
            }
        )

    if comparison_rows:
        comp_path = os.path.join(metrics_dir, "model_comparison.csv")
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"\n模型对比表已保存至: {comp_path}")


if __name__ == "__main__":
    main()
