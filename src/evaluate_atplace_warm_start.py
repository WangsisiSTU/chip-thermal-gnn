"""Benchmark a trained ATPlace network as a nonlinear FVM warm start."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from atplace_nonlinear_operator import nonlinear_warm_start_solve, require_operator_contract
from train_atplace_ml import build_atplace_model


def field_metrics(prediction, graph):
    true = graph.y.numpy().astype(float)
    volume = graph.node_volume.numpy().astype(float)
    error = np.asarray(prediction)-true
    return {
        'volume_weighted_mae_K': float(np.sum(volume*np.abs(error))/volume.sum()),
        'volume_weighted_rmse_K': float(np.sqrt(np.sum(volume*error**2)/volume.sum())),
        'absolute_peak_error_K': float(abs(np.max(prediction)-np.max(true))),
        'max_absolute_cell_error_K': float(np.max(np.abs(error))),
    }


def evaluate(checkpoint_path: Path, data_path: Path, out_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    graphs = torch.load(data_path, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_atplace_model(
        checkpoint['model_name'], checkpoint['metadata'], checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    mean = float(checkpoint['metadata']['dT_train_mean'])
    std = float(checkpoint['metadata']['dT_train_std'])
    cases = []
    for graph in graphs:
        require_operator_contract(graph)
        gpu_graph = graph.clone().to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            prediction = model(gpu_graph)*std+mean
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.perf_counter()-started
        prediction = prediction.detach().cpu().numpy().astype(float)
        warm = nonlinear_warm_start_solve(graph, prediction)
        cold = nonlinear_warm_start_solve(graph, np.zeros(graph.num_nodes, dtype=float))
        cases.append({
            'physical_case_id': int(graph.physical_case_id),
            'layout_id': int(graph.layout_id),
            'n_nodes': int(graph.num_nodes),
            'network': field_metrics(prediction, graph),
            'warm_corrected': field_metrics(warm.temperature_rise_K, graph),
            'network_inference_ms': 1000*inference_time,
            'warm_correction_ms': 1000*warm.solve_time_s,
            'warm_end_to_end_ms': 1000*(inference_time+warm.solve_time_s),
            'cold_solve_ms': 1000*cold.solve_time_s,
            'warm_nonlinear_iterations': warm.nonlinear_iterations,
            'cold_nonlinear_iterations': cold.nonlinear_iterations,
            'warm_linear_iterations': warm.linear_iterations,
            'cold_linear_iterations': cold.linear_iterations,
            'network_initial_residual_rel_l2': warm.initial_residual_rel_l2,
            'corrected_residual_rel_l2': warm.final_residual_rel_l2,
            'corrected_energy_balance_rel_error': warm.energy_balance_rel_error,
        })
    aggregate = {}
    scalar_fields = [key for key in cases[0] if isinstance(cases[0][key], (int, float))]
    for key in scalar_fields:
        if key in ('physical_case_id', 'layout_id', 'n_nodes'):
            continue
        values = np.asarray([case[key] for case in cases], dtype=float)
        aggregate[key] = {'mean': float(values.mean()), 'max': float(values.max())}
    for group in ('network', 'warm_corrected'):
        aggregate[group] = {}
        for key in cases[0][group]:
            values = np.asarray([case[group][key] for case in cases], dtype=float)
            aggregate[group][key] = {'mean': float(values.mean()), 'max': float(values.max())}
    cold_time = aggregate['cold_solve_ms']['mean']
    aggregate['warm_correction_speedup_vs_cold'] = (
        cold_time/aggregate['warm_correction_ms']['mean'])
    aggregate['warm_end_to_end_speedup_vs_cold'] = (
        cold_time/aggregate['warm_end_to_end_ms']['mean'])
    payload = {
        'device': str(device), 'checkpoint': str(checkpoint_path),
        'dataset': str(data_path), 'n_cases': len(cases),
        'solver': {
            'nonlinear_tolerance_K': 1e-7, 'nonlinear_relaxation': 0.70,
            'linear_solver': 'Jacobi-preconditioned conjugate gradient',
            'linear_rtol': 1e-10,
        },
        'aggregate': aggregate, 'cases': cases,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('outputs/atplace_case1_contact_material_ml/seed_4108/mgn_transolver/checkpoint.pt'))
    parser.add_argument('--data', type=Path,
                        default=Path('data/atplace_case1_contact_material_ml/eval_layout_mesh_fine.pt'))
    parser.add_argument('--out', type=Path,
                        default=Path('outputs/atplace_case1_contact_material_ml/warm_start_layout_mesh_fine.json'))
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.data, args.out)
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
