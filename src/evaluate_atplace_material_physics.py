"""Run controlled ATPlace ablations for contact, anisotropy, and k(T)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from atplace_six_layer_fem import load_problem
from atplace_six_layer_fvm import solve_level


def variant_config(base: dict, *, contact: bool, anisotropy: bool,
                   temperature_dependence: bool) -> dict:
    cfg = copy.deepcopy(base)
    advanced = cfg.setdefault('advanced_physics', {})
    if not contact:
        advanced.setdefault('contact_resistance', {})['interfaces'] = []
    if not anisotropy:
        advanced.setdefault('anisotropy', {})['material_multipliers'] = {}
    if not temperature_dependence:
        advanced.setdefault('temperature_dependence', {})['material_alpha_per_K'] = {}
    return cfg


def run(config_path: Path, level: str, out_path: Path) -> dict:
    cfg, layout, chips, width, depth = load_problem(config_path)
    variants = {
        'linear_isotropic_no_contact': (False, False, False),
        'contact_only': (True, False, False),
        'anisotropy_only': (False, True, False),
        'temperature_only': (False, False, True),
        'combined': (True, True, True),
    }
    results = {}
    for name, (contact, anisotropy, temperature) in variants.items():
        metrics, arrays = solve_level(
            variant_config(cfg, contact=contact, anisotropy=anisotropy,
                           temperature_dependence=temperature),
            chips, width, depth, level)
        chip_means = np.asarray([
            item['mean_temperature_K'] for item in metrics['chiplets'].values()])
        results[name] = {
            'contact_resistance': contact,
            'anisotropy': anisotropy,
            'temperature_dependence': temperature,
            'n_cells': metrics['n_cells'],
            'n_directed_edges': metrics['n_directed_edges'],
            'solve_time_s': metrics['solve_time_s'],
            'nonlinear_iterations': metrics['nonlinear_iterations'],
            'temperature_max_K': metrics['temperature_max_K'],
            'temperature_rise_max_K': metrics['temperature_max_K']-metrics['ambient_K'],
            'mean_chip_temperature_K': float(chip_means.mean()),
            'energy_balance_rel_error': metrics['energy_balance_rel_error'],
            'minimum_principle_margin_K': metrics['minimum_principle_margin_K'],
            'anisotropy_ratio_max': metrics['anisotropy_ratio_max'],
            'n_contact_faces': metrics['n_contact_faces'],
            'contact_temperature_jump_mean_abs_K':
                metrics['contact_temperature_jump_mean_abs_K'],
            'contact_temperature_jump_max_abs_K':
                metrics['contact_temperature_jump_max_abs_K'],
            'conductivity_factor_min': float(np.min(
                arrays['conductivity_final_diag_W_mK'] /
                arrays['conductivity_reference_diag_W_mK'])),
            'conductivity_factor_max': float(np.max(
                arrays['conductivity_final_diag_W_mK'] /
                arrays['conductivity_reference_diag_W_mK'])),
        }
        print(name, f"Tmax={metrics['temperature_max_K']:.6f} K",
              f"balance={metrics['energy_balance_rel_error']:.3e}",
              f"iterations={metrics['nonlinear_iterations']}")

    baseline = results['linear_isotropic_no_contact']
    for values in results.values():
        values['delta_peak_vs_linear_isotropic_no_contact_K'] = (
            values['temperature_max_K']-baseline['temperature_max_K'])
        values['delta_mean_chip_vs_linear_isotropic_no_contact_K'] = (
            values['mean_chip_temperature_K']-baseline['mean_chip_temperature_K'])
    report = {
        'case_id': layout['case_id'],
        'layout_id': layout['layout_id'],
        'mesh_level': level,
        'source_config': str(config_path),
        'controlled_variables': (
            'same geometry, mesh, powers, Robin boundary, and base material constants'),
        'variants': results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path('configs/atplace_case1_contact_material.yaml'))
    parser.add_argument('--level', default='medium')
    parser.add_argument('--out', type=Path,
                        default=Path('outputs/atplace_case1_contact_material/physics_ablation.json'))
    args = parser.parse_args()
    run(args.config, args.level, args.out)


if __name__ == '__main__':
    main()
