"""Evaluate trained Case1 models on the independent fine reference graph."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_atplace_prediction import evaluate_array
from train_atplace_ml import build_atplace_model


def load_model(checkpoint, device):
    meta, cfg, name = checkpoint['metadata'], checkpoint['config'], checkpoint['model_name']
    model = build_atplace_model(name, meta, cfg)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), name, meta


def main():
    parser = argparse.ArgumentParser(description='Cross-resolution ATPlace checkpoint evaluation')
    parser.add_argument('--models', type=Path, default=Path('outputs/atplace_case1_ml'))
    parser.add_argument('--graph', type=Path, default=Path('data/atplace_case1_ml_reference/verification.pt'))
    parser.add_argument('--reference', type=Path,
                        default=Path('data/atplace_case1_six_layer_fvm/verification.npz'))
    parser.add_argument('--out', type=Path, default=Path('outputs/atplace_case1_ml/fine_reference_comparison.json'))
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(args.graph, weights_only=False).to(device)
    results = {}
    folders = [path.name for path in args.models.iterdir()
               if path.is_dir() and (path/'checkpoint.pt').exists()]
    for folder in sorted(folders):
        checkpoint = torch.load(args.models/folder/'checkpoint.pt', map_location='cpu', weights_only=False)
        model, name, meta = load_model(checkpoint, device)
        with torch.no_grad():
            for _ in range(5):
                model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            pred = model(data)*float(meta['dT_train_std'])+float(meta['dT_train_mean'])
        results[name] = evaluate_array(args.reference, pred.detach().cpu().numpy(), 'rise_K')
        results[name]['inference_ms'] = 1000*elapsed/20
    payload = {'device': str(device), 'reference_role': 'unseen baseline layout and finer mesh', 'models': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
