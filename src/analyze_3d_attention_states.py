"""Inspect whether learned physics-state slices distinguish thermal regions.

The diagnostic uses deterministic evaluation weights; the Gumbel variant is
stochastic only during training. Entropy close to one means uniform assignment,
and state cosine close to one means nearly identical learned states.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model_from_ckpt
from utils import get_device, load_split


def analyze_model(model, samples, device):
    blocks = getattr(model, "global_blocks", None)
    if blocks is None:
        raise ValueError("model has no physics-state blocks")
    measurements = [[] for _ in blocks]
    handles = []

    for block_index, block in enumerate(blocks):
        def inspect(module, inputs, block_index=block_index):
            if module.slice_logits is None:
                return
            x, graph_id, node_weight = inputs
            if torch.any(graph_id != 0):
                raise ValueError("analyze one physical graph at a time")
            logits = module.slice_logits(x).view(x.size(0), module.heads, module.slices)
            adjustment = module.temperature_adjust(x).unsqueeze(-1) if module.temperature_adjust is not None else 0
            temperature = F.softplus(module.temperature + adjustment) + 1e-4
            weights = F.softmax(logits / temperature, dim=-1)
            entropy = -(weights * weights.clamp_min(1e-20).log()).sum(-1).mean() / np.log(module.slices)
            weighted = weights * node_weight.reshape(-1, 1, 1) if node_weight is not None else weights
            values = module.state_values(x).view(x.size(0), module.heads, module.head_dim)
            states = (weighted.unsqueeze(-1) * values.unsqueeze(2)).sum(0) / weighted.sum(0).clamp_min(1e-8).unsqueeze(-1)
            unit = F.normalize(states, dim=-1)
            cosine = unit @ unit.transpose(-2, -1)
            off_diagonal = (cosine.sum((-1, -2)) - module.slices) / (module.slices * (module.slices - 1))
            measurements[block_index].append((float(entropy), float(weights.max(-1).values.mean()),
                                              float(off_diagonal.mean())))

        handles.append(block.attn.register_forward_pre_hook(inspect))

    try:
        for data in samples:
            with torch.no_grad():
                model(data.to(device))
    finally:
        for handle in handles:
            handle.remove()

    result = []
    for index, values in enumerate(measurements):
        if not values:
            result.append({"block": index + 1, "mode": "pool", "n_cases": len(samples)})
            continue
        array = np.asarray(values)
        result.append({
            "block": index + 1,
            "mode": blocks[index].attn.mode,
            "n_cases": len(values),
            "n_slices": blocks[index].attn.slices,
            "normalized_entropy_mean": float(array[:, 0].mean()),
            "normalized_entropy_min": float(array[:, 0].min()),
            "normalized_entropy_max": float(array[:, 0].max()),
            "max_slice_probability_mean": float(array[:, 1].mean()),
            "within_head_state_cosine_mean": float(array[:, 2].mean()),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze 3D learned physics-state distinction")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = get_device()
    samples = load_split(args.data_dir, "test")
    result = {}
    for name in args.models:
        model, _ = load_model_from_ckpt(os.path.join(args.ckpt_dir, f"{name}_best.pt"), device)
        result[name] = analyze_model(model, samples, device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
