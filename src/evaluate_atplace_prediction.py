"""Score an ML temperature prediction against the converged Case1 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate_array(reference: Path, prediction: np.ndarray, prediction_kind: str = 'rise_K') -> dict:
    raw = np.load(reference)
    pred = np.asarray(prediction, dtype=float).reshape(-1)
    true = raw['temperature_K'].astype(float)
    if pred.size != true.size or not np.all(np.isfinite(pred)):
        raise ValueError(f'prediction must contain {true.size} finite cell values')
    ambient = float(raw['ambient_K'])
    if prediction_kind == 'rise_K':
        pred = pred + ambient
    elif prediction_kind != 'temperature_K':
        raise ValueError('prediction_kind must be rise_K or temperature_K')
    volume = raw['cell_volume_m3'].astype(float)
    weight = volume / volume.sum()
    error = pred-true
    chip_id = raw['chip_id'].astype(int)
    chip_mean_error = {}
    for cid in sorted(set(chip_id)-{-1}):
        mask = chip_id == cid
        chip_mean_error[str(cid)] = float(
            np.average(pred[mask], weights=volume[mask]) - np.average(true[mask], weights=volume[mask]))
    top = raw['top_cell'].astype(int)
    if all(key in raw for key in (
            'top_face_area_m2', 'top_half_distance_m', 'boundary_h_W_m2K',
            'conductivity_reference_diag_W_mK', 'temperature_coefficient_per_K',
            'conductivity_reference_temperature_K', 'conductivity_factor_bounds')):
        bounds = raw['conductivity_factor_bounds'].astype(float)
        factor = np.clip(
            1.0 + raw['temperature_coefficient_per_K'][top].astype(float) *
            (pred[top]-float(raw['conductivity_reference_temperature_K'])),
            bounds[0], bounds[1])
        k_y = raw['conductivity_reference_diag_W_mK'][top, 1].astype(float)*factor
        top_g = raw['top_face_area_m2'].astype(float) / (
            raw['top_half_distance_m'].astype(float)/k_y +
            1.0/float(raw['boundary_h_W_m2K']))
    else:
        top_g = raw['top_conductance_W_K'].astype(float)
    predicted_heat_rejection = float(top_g @ (pred[top]-ambient))
    source_power = float(raw['source_power_W'].sum())
    return {
        'n_cells': int(true.size),
        'volume_weighted_mae_K': float(np.sum(weight*np.abs(error))),
        'volume_weighted_rmse_K': float(np.sqrt(np.sum(weight*error**2))),
        'peak_temperature_error_K': float(pred.max()-true.max()),
        'max_absolute_cell_error_K': float(np.abs(error).max()),
        'chip_mean_temperature_error_K': chip_mean_error,
        'predicted_heat_rejection_W': predicted_heat_rejection,
        'energy_balance_rel_error': abs(predicted_heat_rejection-source_power)/source_power,
    }


def evaluate(reference: Path, prediction: Path, prediction_kind: str = 'rise_K') -> dict:
    return evaluate_array(reference, np.load(prediction), prediction_kind)


def main():
    parser = argparse.ArgumentParser(description='Evaluate an ML prediction on ATPlace Case1')
    parser.add_argument('prediction', type=Path, help='1-D .npy array in verification graph cell order')
    parser.add_argument('--reference', type=Path,
                        default=Path('data/atplace_case1_six_layer_fvm/verification.npz'))
    parser.add_argument('--prediction-kind', choices=['rise_K', 'temperature_K'], default='rise_K')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    metrics = evaluate(args.reference, args.prediction, args.prediction_kind)
    text = json.dumps(metrics, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
