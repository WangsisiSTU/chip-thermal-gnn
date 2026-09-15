"""
训练脚本：训练 MeshGraphNet 主模型或 GCN/GraphSAGE 基线模型。

用法：
    python src/train.py --model meshgraphnet --data_dir data/processed --out_dir outputs
    python src/train.py --model baseline     --data_dir data/processed --out_dir outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# 规避 Windows + Anaconda 下的 OpenMP 重复加载问题（OMP Error #15）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import psutil
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model
from utils import build_data_contract, get_device, load_processed_metadata, load_split, load_yaml, set_seed


def _graph_ids(values: torch.Tensor, num_graphs: int, batch: torch.Tensor | None) -> torch.Tensor:
    if batch is not None:
        if batch.numel() != values.numel():
            raise ValueError("batch must contain one graph id per node")
        return batch
    if num_graphs < 1 or values.numel() % num_graphs:
        raise ValueError("without batch, every graph must have the same number of nodes")
    return torch.arange(num_graphs, device=values.device).repeat_interleave(values.numel() // num_graphs)


def peak_loss_fn(
    pred_norm: torch.Tensor,
    true_norm: torch.Tensor,
    num_graphs: int,
    peak_loss_type: str = "true_peak_node",
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare per-graph peak temperatures in normalized space.

    ``true_peak_node`` retains the historical loss at the true hottest node.
    ``global_max`` also penalizes spurious predicted hot spots and matches the
    peak metric reported by evaluation.
    """
    graph_id = _graph_ids(pred_norm, num_graphs, batch)
    if peak_loss_type == "true_peak_node":
        # Small graph batch: exact argmax node, preserving the historical loss.
        peak_indices = torch.stack([true_norm.masked_fill(graph_id != i, float("-inf")).argmax()
                                    for i in range(num_graphs)])
        pred_peak = pred_norm[peak_indices]
        true_peak = true_norm[peak_indices]
    elif peak_loss_type == "global_max":
        pred_peak = scatter(pred_norm, graph_id, dim=0, dim_size=num_graphs, reduce="max")
        true_peak = scatter(true_norm, graph_id, dim=0, dim_size=num_graphs, reduce="max")
    else:
        raise ValueError(f"Unknown peak loss type: {peak_loss_type}")
    return torch.mean((pred_peak - true_peak) ** 2)


def field_loss_fn(pred_norm: torch.Tensor, true_norm: torch.Tensor, num_graphs: int,
                  loss_type: str, batch: torch.Tensor | None = None) -> torch.Tensor:
    """温度场主损失（标准化空间）。

    zscore_mse:      全局标准化温升的逐节点 MSE。
    per_sample_rel:  在标准化温升基础上，将每个样本的误差平方和再除以该样本自身的
                     信号能量（约等于逐样本相对 L2 误差的平方），避免大温升样本主导损失。
    """
    graph_id = _graph_ids(pred_norm, num_graphs, batch)
    counts = scatter(torch.ones_like(true_norm), graph_id, dim=0, dim_size=num_graphs, reduce="sum")
    error = scatter((pred_norm - true_norm) ** 2, graph_id, dim=0, dim_size=num_graphs, reduce="sum")
    if loss_type == "zscore_mse":
        return torch.mean(error / counts)
    elif loss_type == "per_sample_rel":
        graph_means = scatter(true_norm, graph_id, dim=0, dim_size=num_graphs, reduce="sum") / counts
        signal = scatter((true_norm - graph_means[graph_id]) ** 2,
                         graph_id, dim=0, dim_size=num_graphs, reduce="sum")
        return torch.mean(error / (signal + 1e-4))
    else:
        raise ValueError(f"未知损失类型: {loss_type}")


def run_epoch(model, loader, device, dT_mean, dT_std, peak_weight, loss_type, peak_loss_type="true_peak_node", optimizer=None, grad_clip=None):
    is_train = optimizer is not None
    model.train(is_train)
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    n_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        y_norm = (batch.y - dT_mean) / dT_std
        with torch.set_grad_enabled(is_train):
            pred_norm = model(batch)
            field = field_loss_fn(pred_norm, y_norm, batch.num_graphs, loss_type, batch.batch)
            pk = peak_loss_fn(pred_norm, y_norm, batch.num_graphs, peak_loss_type, batch.batch)
            loss = field + peak_weight * pk
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        totals += torch.stack((loss.detach(), field.detach(), pk.detach())).to(torch.float64) * batch.num_graphs
        n_graphs += batch.num_graphs
    if n_graphs == 0:
        raise ValueError("Cannot run an epoch on an empty dataset")
    return tuple((totals / n_graphs).cpu().tolist())


def main():
    parser = argparse.ArgumentParser(description="训练芯片封装温度场图神经网络代理模型")
    parser.add_argument("--model", type=str, choices=["meshgraphnet", "baseline", "mgn_transolver"], required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override the configured epoch count (useful for benchmarks)")
    parser.add_argument("--cpu_threads", type=int, default=None, help="limit CPU worker threads for reproducible CPU runs")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        cfg = {**cfg, "train": {**cfg["train"], "epochs": args.epochs}}
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)

    device = get_device()
    cpu_threads = args.cpu_threads if args.cpu_threads is not None else cfg["train"].get("cpu_num_threads")
    if device.type == "cpu" and cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("cpu thread count must be positive")
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        print(f"CPU threads: {torch.get_num_threads()}")
    process = psutil.Process()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    peak_process_rss = process.memory_info().rss
    print(f"使用设备: {device}")

    proc_meta = load_processed_metadata(args.data_dir)
    dT_mean = torch.tensor(proc_meta["dT_train_mean"], dtype=torch.float32, device=device)
    dT_std = torch.tensor(proc_meta["dT_train_std"], dtype=torch.float32, device=device)

    train_set = load_split(args.data_dir, "train")
    val_set = load_split(args.data_dir, "val")

    node_in_dim = train_set[0].x.shape[1]
    edge_in_dim = train_set[0].edge_attr.shape[1]

    tcfg = cfg["train"]
    train_loader = DataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=tcfg["batch_size"], shuffle=False)

    model = build_model(args.model, node_in_dim, edge_in_dim, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {args.model}, 参数量: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg["lr_scheduler_factor"], patience=tcfg["lr_scheduler_patience"]
    )

    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    metrics_dir = os.path.join(args.out_dir, "metrics")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, f"{args.model}_best.pt")

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    t0 = time.perf_counter()
    loss_type = tcfg.get("loss_type", "per_sample_rel")
    peak_loss_type = tcfg.get("peak_loss_type", "true_peak_node")
    for epoch in range(1, tcfg["epochs"] + 1):
        train_loss, train_field, train_peak = run_epoch(
            model, train_loader, device, dT_mean, dT_std, tcfg["peak_loss_weight"], loss_type, peak_loss_type,
            optimizer=optimizer, grad_clip=tcfg["grad_clip_norm"],
        )
        val_loss, val_field, val_peak = run_epoch(
            model, val_loader, device, dT_mean, dT_std, tcfg["peak_loss_weight"], loss_type, peak_loss_type, optimizer=None,
        )
        scheduler.step(val_loss)
        peak_process_rss = max(peak_process_rss, process.memory_info().rss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch, "train_loss": train_loss, "train_field": train_field, "train_peak": train_peak,
                "val_loss": val_loss, "val_field": val_field, "val_peak": val_peak, "lr": current_lr,
            }
        )

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "node_in_dim": node_in_dim,
                    "edge_in_dim": edge_in_dim,
                    "model_name": args.model,
                    "config": cfg,
                    "dT_mean": proc_meta["dT_train_mean"],
                    "dT_std": proc_meta["dT_train_std"],
                    "data_contract": build_data_contract(proc_meta),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            print(
                f"[{args.model}] epoch {epoch:4d} | train_loss {train_loss:.5f} | "
                f"val_loss {val_loss:.5f} (field {val_field:.5f}, peak {val_peak:.5f}) | "
                f"lr {current_lr:.2e}" + (" *" if improved else "")
            )

        if patience_counter >= tcfg["patience"]:
            print(f"验证集损失连续 {tcfg['patience']} 轮未改善，提前停止于 epoch {epoch}")
            break

    total_time = time.perf_counter() - t0
    print(f"训练完成，用时 {total_time:.1f} s，最佳 epoch={best_epoch}，最佳验证 loss={best_val_loss:.5f}")

    history_path = os.path.join(metrics_dir, f"{args.model}_loss_history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": args.model,
        "loss_type": loss_type,
        "peak_loss_type": peak_loss_type,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs_run": len(history),
        "total_train_time_s": total_time,
        "seed": seed,
        "runtime_device": str(device),
        "process_rss_MiB": peak_process_rss / 2 ** 20,
        "cuda_peak_memory_MiB": (
            torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
        ),
    }
    with open(os.path.join(metrics_dir, f"{args.model}_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"损失曲线已保存至: {history_path}")
    print(f"最佳权重已保存至: {best_ckpt_path}")


if __name__ == "__main__":
    main()
