"""Conservative cell-centred reference solver for the ATPlace six-layer stack.

The orthogonal finite-volume operator is an M-matrix: positive chip powers
cannot produce temperatures below ambient.  It is used as the trustworthy ML
label path; the hexahedral FEM entry remains available for numerical comparison.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from atplace_six_layer_fem import (
    MATERIAL_NAMES, _axis_with_edges, classify, layer_bounds, load_problem,
    material_constants,
)


def grid_axes(cfg, chips, width, depth, level):
    spec = cfg['mesh_levels'][level]
    x = _axis_with_edges(width, spec['max_xz_cell_m'], (v for c in chips for v in (c.x0, c.x1)))
    z = _axis_with_edges(depth, spec['max_xz_cell_m'], (v for c in chips for v in (c.z0, c.z1)))
    parts, y0 = [[0.0]], 0.0
    for layer in cfg['layers']:
        y1 = y0 + float(layer['thickness_m'])
        parts.append(np.linspace(y0, y1, int(spec['cells_per_layer']) + 1)[1:])
        y0 = y1
    return x, np.unique(np.round(np.concatenate(parts), 14)), z


def material_transport_model(cfg):
    """Return reference diagonal conductivity and temperature coefficients."""
    scalar = material_constants(cfg)
    diagonal = np.repeat(scalar[:, None], 3, axis=1)
    advanced = cfg.get('advanced_physics', {})
    anisotropy = advanced.get('anisotropy', {}).get('material_multipliers', {})
    name_to_id = {name: index for index, name in enumerate(MATERIAL_NAMES)}
    for name, multiplier in anisotropy.items():
        if name not in name_to_id:
            raise ValueError(f'unknown anisotropic material {name}')
        multiplier = np.asarray(multiplier, dtype=float)
        if multiplier.shape != (3,) or np.any(multiplier <= 0):
            raise ValueError(f'anisotropy multiplier for {name} must contain three positive values')
        diagonal[name_to_id[name]] *= multiplier
    temperature = advanced.get('temperature_dependence', {})
    alpha = np.zeros(len(MATERIAL_NAMES), dtype=float)
    for name, value in temperature.get('material_alpha_per_K', {}).items():
        if name not in name_to_id:
            raise ValueError(f'unknown temperature-dependent material {name}')
        alpha[name_to_id[name]] = float(value)
    reference_temperature = float(temperature.get('reference_temperature_K', cfg['ambient_K']))
    factor_bounds = (float(temperature.get('min_factor', 0.25)),
                     float(temperature.get('max_factor', 4.0)))
    if not 0 < factor_bounds[0] <= factor_bounds[1]:
        raise ValueError('temperature-dependent conductivity factor bounds are invalid')
    return diagonal, alpha, reference_temperature, factor_bounds


def contact_resistance_map(cfg):
    """Map adjacent layer-id pairs to area-normalized contact resistance."""
    layer_ids = {layer['name']: index for index, layer in enumerate(cfg['layers'])}
    result = {}
    entries = cfg.get('advanced_physics', {}).get('contact_resistance', {}).get('interfaces', [])
    for entry in entries:
        names = entry.get('between_layers', [])
        if len(names) != 2 or any(name not in layer_ids for name in names):
            raise ValueError(f'invalid contact interface layers: {names}')
        pair = tuple(sorted((layer_ids[names[0]], layer_ids[names[1]])))
        if abs(pair[0]-pair[1]) != 1:
            raise ValueError(f'contact interface must join adjacent layers: {names}')
        resistance = float(entry['resistance_m2K_W'])
        if resistance < 0:
            raise ValueError('contact resistance must be non-negative')
        if pair in result:
            raise ValueError(f'duplicate contact interface {names}')
        result[pair] = resistance
    return result


def solve_level(cfg, chips, width, depth, level):
    x, y, z = grid_axes(cfg, chips, width, depth, level)
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    xc, yc, zc = (a[:-1] + np.diff(a)/2 for a in (x, y, z))
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')
    shape = X.shape
    centers = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
    layer, chip_id, material = classify(centers[0], centers[1], centers[2], cfg, chips)
    k_reference_by_material, alpha_by_material, reference_temperature, factor_bounds = (
        material_transport_model(cfg))
    k_reference = k_reference_by_material[material]
    alpha = alpha_by_material[material]
    contacts = contact_resistance_map(cfg)
    volume = (dx[:, None, None] * dy[None, :, None] * dz[None, None, :]).ravel()
    source = np.zeros(centers.shape[1])
    for i, chip in enumerate(chips):
        mask = (layer == 4) & (chip_id == i)
        if not mask.any():
            raise ValueError(f'no cells represent chiplet {chip.name}')
        source[mask] = chip.power_W * volume[mask] / volume[mask].sum()

    def index(i, j, k_):
        return np.ravel_multi_index((i, j, k_), shape)

    I, J, K = np.indices(shape)
    top = index(I[:, -1, :], J[:, -1, :], K[:, -1, :]).ravel()
    top_area = (dx[:, None] * dz[None, :]).ravel()
    h = 1.0 / (float(cfg['boundary']['r_convec_total_K_W']) * width * depth)

    def conductivity_at(temperature):
        factor = np.clip(1.0 + alpha*(temperature-reference_temperature),
                         factor_bounds[0], factor_bounds[1])
        return k_reference*factor[:, None]

    def assemble(k_diagonal):
        rows, cols, values = [], [], []
        diagonal = np.zeros(centers.shape[1])
        graph_src, graph_dst, graph_g, graph_area, graph_rtc = [], [], [], [], []
        graph_axis, graph_distance_src, graph_distance_dst = [], [], []

        def connect(a, b, area, distance_a, distance_b, axis, contact_r=None):
            edge_shape = np.asarray(a).shape
            a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
            area = np.broadcast_to(area, edge_shape).ravel()
            distance_a = np.broadcast_to(distance_a, edge_shape).ravel()
            distance_b = np.broadcast_to(distance_b, edge_shape).ravel()
            if contact_r is None:
                contact_r = np.zeros_like(area)
            else:
                contact_r = np.broadcast_to(contact_r, edge_shape).ravel()
            resistance_area = (distance_a/k_diagonal[a, axis] + contact_r +
                               distance_b/k_diagonal[b, axis])
            conductance = area/resistance_area
            rows.extend(np.concatenate((a, b)).tolist())
            cols.extend(np.concatenate((b, a)).tolist())
            values.extend(np.concatenate((-conductance, -conductance)).tolist())
            np.add.at(diagonal, a, conductance)
            np.add.at(diagonal, b, conductance)
            graph_src.extend(np.concatenate((a, b)).tolist())
            graph_dst.extend(np.concatenate((b, a)).tolist())
            graph_g.extend(np.concatenate((conductance, conductance)).tolist())
            graph_area.extend(np.concatenate((area, area)).tolist())
            graph_rtc.extend(np.concatenate((contact_r, contact_r)).tolist())
            graph_axis.extend(np.full(2*a.size, axis, dtype=np.int8).tolist())
            graph_distance_src.extend(np.concatenate((distance_a, distance_b)).tolist())
            graph_distance_dst.extend(np.concatenate((distance_b, distance_a)).tolist())

        if shape[0] > 1:
            a, b = index(I[:-1], J[:-1], K[:-1]), index(I[1:], J[1:], K[1:])
            connect(a, b, dy[None, :, None]*dz[None, None, :],
                    dx[:-1, None, None]/2, dx[1:, None, None]/2, 0)
        if shape[1] > 1:
            a, b = index(I[:, :-1], J[:, :-1], K[:, :-1]), index(I[:, 1:], J[:, 1:], K[:, 1:])
            pair_r = np.zeros(np.asarray(a).shape, dtype=float)
            flat_a, flat_b = np.asarray(a).ravel(), np.asarray(b).ravel()
            for pair, resistance in contacts.items():
                mask = ((layer[flat_a] == pair[0]) & (layer[flat_b] == pair[1])) | (
                    (layer[flat_a] == pair[1]) & (layer[flat_b] == pair[0]))
                pair_r.ravel()[mask] = resistance
            connect(a, b, dx[:, None, None]*dz[None, None, :],
                    dy[None, :-1, None]/2, dy[None, 1:, None]/2, 1, pair_r)
        if shape[2] > 1:
            a, b = index(I[:, :, :-1], J[:, :, :-1], K[:, :, :-1]), index(I[:, :, 1:], J[:, :, 1:], K[:, :, 1:])
            connect(a, b, dx[:, None, None]*dy[None, :, None],
                    dz[None, None, :-1]/2, dz[None, None, 1:]/2, 2)

        top_g = top_area/(dy[-1]/(2*k_diagonal[top, 1]) + 1.0/h)
        np.add.at(diagonal, top, top_g)
        rows.extend(np.arange(centers.shape[1]).tolist())
        cols.extend(np.arange(centers.shape[1]).tolist())
        values.extend(diagonal.tolist())
        operator = coo_matrix((values, (rows, cols)), shape=(centers.shape[1],)*2).tocsr()
        return {
            'operator': operator, 'top_g': top_g,
            'graph_src': np.asarray(graph_src, dtype=np.int64),
            'graph_dst': np.asarray(graph_dst, dtype=np.int64),
            'graph_g': np.asarray(graph_g), 'graph_area': np.asarray(graph_area),
            'graph_rtc': np.asarray(graph_rtc),
            'graph_axis': np.asarray(graph_axis, dtype=np.int8),
            'graph_distance_src': np.asarray(graph_distance_src),
            'graph_distance_dst': np.asarray(graph_distance_dst),
        }

    reference_assembly = assemble(k_reference)
    nonlinear = cfg.get('advanced_physics', {}).get('temperature_dependence', {}).get('solver', {})
    max_iterations = int(nonlinear.get('max_iterations', 50))
    tolerance = float(nonlinear.get('tolerance_K', 1e-7))
    relaxation = float(nonlinear.get('relaxation', 1.0))
    if max_iterations < 1 or tolerance <= 0 or not 0 < relaxation <= 1:
        raise ValueError('invalid nonlinear conductivity solver settings')
    started = time.perf_counter()
    if not np.any(alpha):
        # Avoid artificial Picard iterations when the material law is linear.
        # This also makes ablation solve-time comparisons interpretable.
        theta_linear = spsolve(reference_assembly['operator'], source)
        temperature_iterate = float(cfg['ambient_K']) + theta_linear
        nonlinear_iterations, nonlinear_update = 1, 0.0
    else:
        temperature_iterate = np.full(centers.shape[1], float(cfg['ambient_K']))
        nonlinear_update = float('inf')
        for nonlinear_iterations in range(1, max_iterations+1):
            assembly = assemble(conductivity_at(temperature_iterate))
            theta_candidate = spsolve(assembly['operator'], source)
            temperature_candidate = float(cfg['ambient_K']) + theta_candidate
            nonlinear_update = float(np.max(np.abs(temperature_candidate-temperature_iterate)))
            if nonlinear_update <= tolerance:
                temperature_iterate = temperature_candidate
                break
            temperature_iterate = ((1.0-relaxation)*temperature_iterate +
                                   relaxation*temperature_candidate)
        else:
            raise RuntimeError(
                f'temperature-dependent conductivity did not converge in {max_iterations} iterations; '
                f'last update={nonlinear_update:.3e} K')
    k_final = conductivity_at(temperature_iterate)
    if not np.any(alpha):
        final_assembly, theta = reference_assembly, theta_linear
    else:
        final_assembly = assemble(k_final)
        theta = spsolve(final_assembly['operator'], source)
    operator, top_g = final_assembly['operator'], final_assembly['top_g']
    solve_time = time.perf_counter() - started
    temperature = float(cfg['ambient_K']) + theta
    residual = operator @ theta - source
    power_out = float(top_g @ theta[top])
    chip_metrics = {}
    for i, chip in enumerate(chips):
        mask = (layer == 4) & (chip_id == i)
        chip_metrics[chip.name] = {
            'power_W': chip.power_W,
            'mean_temperature_K': float(np.average(temperature[mask], weights=volume[mask])),
            'max_temperature_K': float(temperature[mask].max()),
        }
    metrics = {
        'level': level, 'n_cells': int(centers.shape[1]),
        'n_directed_edges': int(final_assembly['graph_src'].size),
        'solve_time_s': solve_time, 'ambient_K': float(cfg['ambient_K']),
        'source_power_W': float(source.sum()),
        'specified_power_W': float(sum(c.power_W for c in chips)), 'convective_power_W': power_out,
        'source_power_rel_error': abs(source.sum()-sum(c.power_W for c in chips))/sum(c.power_W for c in chips),
        'energy_balance_rel_error': abs(source.sum()-power_out)/source.sum(),
        'algebraic_residual_rel_l2': float(np.linalg.norm(residual)/np.linalg.norm(source)),
        'nonlinear_iterations': nonlinear_iterations,
        'nonlinear_last_update_K': nonlinear_update,
        'anisotropy_ratio_max': float(np.max(
            np.max(k_reference_by_material, axis=1)/np.min(k_reference_by_material, axis=1))),
        'temperature_min_K': float(temperature.min()), 'temperature_max_K': float(temperature.max()),
        'minimum_principle_margin_K': float(temperature.min()-cfg['ambient_K']), 'chiplets': chip_metrics,
    }
    graph_src, graph_dst = final_assembly['graph_src'], final_assembly['graph_dst']
    contact_mask = (final_assembly['graph_rtc'] > 0) & (graph_src < graph_dst)
    contact_a, contact_b = graph_src[contact_mask], graph_dst[contact_mask]
    contact_area = final_assembly['graph_area'][contact_mask]
    contact_r = final_assembly['graph_rtc'][contact_mask]
    contact_flow = final_assembly['graph_g'][contact_mask]*(temperature[contact_a]-temperature[contact_b])
    contact_jump = contact_flow/contact_area*contact_r if contact_area.size else np.empty(0)
    contact_flow_reference = (reference_assembly['graph_g'][contact_mask] *
                              (temperature[contact_a]-temperature[contact_b]))
    contact_jump_reference = (contact_flow_reference/contact_area*contact_r
                              if contact_area.size else np.empty(0))
    metrics.update({
        'n_contact_faces': int(contact_area.size),
        'contact_temperature_jump_mean_abs_K': float(np.mean(np.abs(contact_jump))) if contact_jump.size else 0.0,
        'contact_temperature_jump_max_abs_K': float(np.max(np.abs(contact_jump))) if contact_jump.size else 0.0,
    })
    arrays = dict(
        pos_m=centers.T, cell_volume_m3=volume, temperature_K=temperature,
        ambient_K=np.asarray(float(cfg['ambient_K'])),
        source_power_W=source, conductivity_W_mK=np.cbrt(np.prod(k_final, axis=1)),
        conductivity_reference_diag_W_mK=k_reference,
        conductivity_final_diag_W_mK=k_final,
        temperature_coefficient_per_K=alpha,
        material_id=material,
        layer_id=layer, chip_id=chip_id,
        edge_index=np.asarray([graph_src, graph_dst], dtype=np.int64),
        edge_conductance_W_K=final_assembly['graph_g'],
        edge_conductance_reference_W_K=reference_assembly['graph_g'],
        edge_face_area_m2=final_assembly['graph_area'],
        edge_contact_resistance_m2K_W=final_assembly['graph_rtc'],
        edge_axis=final_assembly['graph_axis'],
        edge_half_distance_src_m=final_assembly['graph_distance_src'],
        edge_half_distance_dst_m=final_assembly['graph_distance_dst'],
        top_cell=top, top_conductance_W_K=top_g,
        top_conductance_reference_W_K=reference_assembly['top_g'],
        top_face_area_m2=top_area,
        top_half_distance_m=np.full(top.size, dy[-1]/2),
        boundary_h_W_m2K=np.asarray(h),
        conductivity_reference_temperature_K=np.asarray(reference_temperature),
        conductivity_factor_bounds=np.asarray(factor_bounds),
        contact_pair_index=np.asarray([contact_a, contact_b], dtype=np.int64),
        contact_face_area_m2=contact_area,
        contact_resistance_m2K_W=contact_r,
        contact_conductance_W_K=final_assembly['graph_g'][contact_mask],
        contact_conductance_reference_W_K=reference_assembly['graph_g'][contact_mask],
        contact_heat_flow_W=contact_flow,
        contact_temperature_jump_K=contact_jump,
        contact_temperature_jump_reference_operator_K=contact_jump_reference,
    )
    return metrics, arrays


def run_convergence(config_path: Path, out_dir: Path):
    cfg, layout, chips, width, depth = load_problem(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in cfg['mesh_levels']:
        metrics, arrays = solve_level(cfg, chips, width, depth, level)
        np.savez_compressed(out_dir / f'{level}.npz', **arrays)
        rows.append(metrics)
    for previous, current in zip(rows, rows[1:]):
        current['delta_peak_from_previous_K'] = abs(current['temperature_max_K']-previous['temperature_max_K'])
        current['max_chip_mean_delta_from_previous_K'] = max(
            abs(current['chiplets'][name]['mean_temperature_K']-previous['chiplets'][name]['mean_temperature_K'])
            for name in current['chiplets'])
    tol, final = cfg['convergence'], rows[-1]
    checks = {
        'source_power': all(r['source_power_rel_error'] <= tol['source_power_rel_tol'] for r in rows),
        'energy_balance': all(r['energy_balance_rel_error'] <= tol['energy_balance_rel_tol'] for r in rows),
        'minimum_principle': all(r['minimum_principle_margin_K'] >= -1e-9 for r in rows),
        'peak_temperature': final['delta_peak_from_previous_K'] <= tol['peak_temperature_abs_tol_K'],
        'chip_mean_temperature': final['max_chip_mean_delta_from_previous_K'] <= tol['chip_mean_temperature_abs_tol_K'],
    }
    report = {
        'case_id': layout['case_id'], 'layout_id': layout['layout_id'],
        'layout_provenance': layout['provenance'],
        'model': 'six_layer_cell_centered_finite_volume_v1',
        'boundary_model': cfg['boundary'], 'material_names': MATERIAL_NAMES,
        'levels': rows, 'checks': checks, 'all_checks_passed': all(checks.values()),
        'ml_gate': 'ready_for_graph_conversion' if all(checks.values()) else 'blocked_until_convergence_passes',
    }
    (out_dir / 'convergence_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Conservative ATPlace Case1 six-layer reference solve')
    parser.add_argument('--config', type=Path, default=Path('configs/atplace_case1_six_layer.yaml'))
    parser.add_argument('--out', type=Path, default=Path('data/atplace_case1_six_layer_fvm'))
    args = parser.parse_args()
    report = run_convergence(args.config, args.out)
    for row in report['levels']:
        print(row['level'], row['n_cells'], f"P={row['source_power_W']:.6f} W",
              f"T={row['temperature_min_K']:.4f}..{row['temperature_max_K']:.4f} K",
              f"balance={row['energy_balance_rel_error']:.3e}")
    print('checks:', report['checks'])
    if not report['all_checks_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
