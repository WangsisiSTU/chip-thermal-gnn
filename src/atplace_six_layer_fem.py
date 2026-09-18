"""ATPlace multi-chiplet six-layer 3-D steady thermal FEM diagnostic entry.

This is deliberately separate from the legacy four-layer demonstration solver.
The Q1 hexahedral mesh is aligned to every layer and chiplet edge.  This path
is retained for cross-discretization checks; conservative ML labels come from
``atplace_six_layer_fvm.py`` because extreme layer aspect ratios can violate
the Galerkin discrete maximum principle.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementHex1, FacetBasis, LinearForm, MeshHex
from skfem.helpers import grad

MATERIAL_NAMES = (
    'substrate', 'c4_effective', 'tsv_silicon_effective',
    'ubump_effective', 'ubump_underfill', 'chip_silicon',
    'chip_underfill', 'tim',
)


@dataclass(frozen=True)
class Chiplet:
    name: str
    x0: float
    x1: float
    z0: float
    z1: float
    power_W: float


@dataclass
class AtplaceResult:
    mesh: MeshHex
    temperature_K: np.ndarray
    source_load_W: np.ndarray
    conductivity_node_W_mK: np.ndarray
    material_id_node: np.ndarray
    material_id_tet: np.ndarray
    chip_id_tet: np.ndarray
    layer_id_tet: np.ndarray
    top_node: np.ndarray
    metrics: dict


def _effective_k(k_metal: float, k_matrix: float, diameter: float, pitch: float) -> float:
    """Parallel-area mixture, algebraically identical to ATPlace resistivity rule."""
    area_ratio = (pitch / diameter) ** 2 - 1.0
    rho_metal, rho_matrix = 1.0 / k_metal, 1.0 / k_matrix
    rho = (1.0 + area_ratio) * rho_metal * rho_matrix / (rho_matrix + area_ratio * rho_metal)
    return 1.0 / rho


def load_problem(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    project_root = config_path.resolve().parents[1]
    layout_path = Path(cfg['layout'])
    if not layout_path.is_absolute():
        layout_path = project_root / layout_path
    layout = json.loads(layout_path.read_text(encoding='utf-8'))
    if not layout.get('geometric_checks_passed'):
        raise ValueError('layout has not passed geometric validation')
    width, depth = (v * 1e-6 for v in layout['interposer_size_um'])
    chips = []
    for c in layout['chiplets']:
        x0, z0 = (v * 1e-6 for v in c['origin_um'])
        w, d = (v * 1e-6 for v in c['size_um'])
        chips.append(Chiplet(c['name'], x0, x0+w, z0, z0+d, float(c['power_W'])))
    if not math.isclose(sum(c.power_W for c in chips), float(layout['total_power_W']), rel_tol=1e-12):
        raise ValueError('layout power total is inconsistent')
    return cfg, layout, chips, width, depth


def _axis_with_edges(length: float, max_cell: float, edges) -> np.ndarray:
    length = round(float(length), 14)
    base = np.linspace(0.0, length, math.ceil(length / max_cell) + 1)
    # Decimal input edges and linspace can differ by ~1e-18 m; merge those
    # representations or they create near-zero-width cells and a singular mesh.
    axis = np.unique(np.round(np.concatenate((base, np.asarray(list(edges), dtype=float), [0.0, length])), 14))
    axis = axis[(axis >= -1e-14) & (axis <= length + 1e-14)]
    axis[0], axis[-1] = 0.0, length
    return axis


def build_mesh(cfg: dict, chips: list[Chiplet], width: float, depth: float, level: str):
    spec = cfg['mesh_levels'][level]
    x = _axis_with_edges(width, spec['max_xz_cell_m'], (v for c in chips for v in (c.x0, c.x1)))
    z = _axis_with_edges(depth, spec['max_xz_cell_m'], (v for c in chips for v in (c.z0, c.z1)))
    y_parts, y0 = [[0.0]], 0.0
    for layer in cfg['layers']:
        y1 = y0 + float(layer['thickness_m'])
        y_parts.append(np.linspace(y0, y1, int(spec['cells_per_layer']) + 1)[1:])
        y0 = y1
    y = np.unique(np.concatenate(y_parts))
    alpha = float(cfg.get('vertical_coordinate_scale', 1.0))
    if not 0.0 < alpha <= 1.0:
        raise ValueError('vertical_coordinate_scale must be in (0, 1]')
    mesh = MeshHex.init_tensor(x, y / alpha, z)
    return mesh, Basis(mesh, ElementHex1()), y


def layer_bounds(cfg: dict):
    bounds, y = [], 0.0
    for layer in cfg['layers']:
        y_next = y + float(layer['thickness_m'])
        bounds.append((y, y_next))
        y = y_next
    return bounds


def _chip_ids(x, z, chips: list[Chiplet]):
    ids = np.full(np.broadcast(x, z).shape, -1, dtype=np.int16)
    for i, c in enumerate(chips):
        inside = (x >= c.x0 - 1e-14) & (x <= c.x1 + 1e-14) & (z >= c.z0 - 1e-14) & (z <= c.z1 + 1e-14)
        ids[inside] = i
    return ids


def material_constants(cfg: dict):
    mat = cfg['materials']
    k_cu, k_uf, k_si = mat['copper']['k_W_mK'], mat['underfill']['k_W_mK'], mat['silicon']['k_W_mK']
    return np.asarray([
        cfg['layers'][0]['k_W_mK'],
        _effective_k(k_cu, k_uf, mat['c4']['diameter_m'], mat['c4']['pitch_m']),
        _effective_k(k_cu, k_si, mat['tsv']['diameter_m'], mat['tsv']['pitch_m']),
        _effective_k(k_cu, k_uf, mat['ubump']['diameter_m'], mat['ubump']['pitch_m']),
        k_uf, k_si, k_uf, cfg['layers'][5]['k_W_mK'],
    ], dtype=float)


def classify(x, y, z, cfg: dict, chips: list[Chiplet]):
    bounds = layer_bounds(cfg)
    layer = np.zeros(np.broadcast(x, y, z).shape, dtype=np.int8)
    for i, (_, y1) in enumerate(bounds[:-1]):
        layer = np.where(y > y1 + 1e-14, i + 1, layer)
    chip = _chip_ids(x, z, chips)
    material = np.choose(layer, [
        np.full_like(layer, 0), np.full_like(layer, 1), np.full_like(layer, 2),
        np.where(chip >= 0, 3, 4), np.where(chip >= 0, 5, 6), np.full_like(layer, 7),
    ])
    return layer, chip, material


def solve_level(cfg: dict, chips: list[Chiplet], width: float, depth: float, level: str) -> AtplaceResult:
    mesh, basis, y_axis = build_mesh(cfg, chips, width, depth, level)
    k_values = material_constants(cfg)
    bounds = layer_bounds(cfg)
    chip_y0, chip_y1 = bounds[4]
    ambient = float(cfg['ambient_K'])
    r_total = float(cfg['boundary']['r_convec_total_K_W'])
    h_top = 1.0 / (r_total * width * depth)
    alpha = float(cfg.get('vertical_coordinate_scale', 1.0))

    def k_at(x, y, z):
        return k_values[classify(x, y, z, cfg, chips)[2]]

    @BilinearForm
    def conduction(u, v, w):
        k = k_at(w.x[0], alpha*w.x[1], w.x[2])
        gu, gv = grad(u), grad(v)
        return k * (alpha*gu[0]*gv[0] + gu[1]*gv[1]/alpha + alpha*gu[2]*gv[2])

    @BilinearForm
    def robin(u, v, w):
        return h_top * u * v

    top_facets = mesh.facets_satisfying(lambda p: np.isclose(p[1], y_axis[-1] / alpha))
    top_basis = FacetBasis(mesh, basis.elem, facets=top_facets)
    started = time.perf_counter()
    K = conduction.assemble(basis).tocsr()
    R = robin.assemble(top_basis).tocsr()
    f = np.zeros(mesh.p.shape[1], dtype=float)
    raw_chip_power = {}
    for chip in chips:
        volume = (chip.x1-chip.x0) * (chip.z1-chip.z0) * (chip_y1-chip_y0)

        @LinearForm
        def chip_source(v, w, chip=chip, volume=volume):
            y_physical = alpha*w.x[1]
            inside = ((w.x[0] >= chip.x0-1e-14) & (w.x[0] <= chip.x1+1e-14) &
                      (w.x[2] >= chip.z0-1e-14) & (w.x[2] <= chip.z1+1e-14) &
                      (y_physical >= chip_y0-1e-14) & (y_physical <= chip_y1+1e-14))
            return alpha * (chip.power_W / volume) * inside * v

        chip_load = np.asarray(chip_source.assemble(basis), dtype=float)
        quadrature_power = float(chip_load.sum())
        if quadrature_power <= 0.0:
            raise ValueError(f'chip source has zero discrete support: {chip.name}')
        raw_chip_power[chip.name] = quadrature_power
        f += chip_load * (chip.power_W / quadrature_power)
    ones = np.ones(mesh.p.shape[1])
    # Constant temperature is the exact null mode of conduction.  Micron-thick
    # layers cause cancellation in that row sum, so restore it conservatively.
    K = (K - diags(np.asarray(K @ ones).ravel())).tocsr()
    A = K + R
    # Write theta=u+c with u[0]=0.  Remaining FEM rows determine u, while the
    # omitted row is replaced by exact global Robin energy balance.
    free = np.arange(1, mesh.p.shape[1])
    robin_constant = np.asarray(R @ ones).ravel()
    conductance = float(robin_constant.sum())
    block = bmat([
        [A[free][:, free], csr_matrix(robin_constant[free, None])],
        [csr_matrix(robin_constant[free][None, :]), csr_matrix([[conductance]])],
    ], format='csr')
    rhs = np.concatenate((f[free], [f.sum()]))
    lu = splu(block.tocsc(), options={'Equil': True, 'IterRefine': 'EXTRA'})
    solution = lu.solve(rhs)
    # SuperLU's built-in refinement is insufficient for some highly anisotropic
    # meshes on Windows; explicit residual corrections are cheap and auditable.
    for _ in range(12):
        block_residual = rhs - block @ solution
        if np.linalg.norm(block_residual) <= 1e-11 * np.linalg.norm(rhs):
            break
        solution += lu.solve(block_residual)
    theta = np.empty(mesh.p.shape[1])
    theta[0] = solution[-1]
    theta[free] = solution[:-1] + solution[-1]
    solve_time = time.perf_counter() - started
    temperature = ambient + theta

    physical_points = mesh.p.copy()
    physical_points[1] *= alpha
    centers = physical_points[:, mesh.t].mean(axis=1)
    layer_t, chip_t, material_t = classify(centers[0], centers[1], centers[2], cfg, chips)
    vertices = physical_points[:, mesh.t].transpose(2, 1, 0)
    volumes = np.prod(vertices.max(axis=1) - vertices.min(axis=1), axis=1)
    tet_temp = temperature[mesh.t].mean(axis=0)
    chip_metrics = {}
    for i, chip in enumerate(chips):
        mask = (layer_t == 4) & (chip_t == i)
        chip_metrics[chip.name] = {
            'power_W': chip.power_W,
            'mean_temperature_K': float(np.average(tet_temp[mask], weights=volumes[mask])),
            'max_temperature_K': float(temperature[np.unique(mesh.t[:, mask])].max()),
        }

    power_in = float(f.sum())
    power_out = float(ones @ (R @ theta))
    residual = A @ theta - f
    mass = np.zeros(mesh.p.shape[1])
    for corner in mesh.t:
        np.add.at(mass, corner, volumes / mesh.t.shape[0])
    k_node = np.zeros(mesh.p.shape[1])
    mat_vote = np.zeros((len(MATERIAL_NAMES), mesh.p.shape[1]))
    for corner in mesh.t:
        np.add.at(k_node, corner, k_values[material_t] * volumes / mesh.t.shape[0])
        for mid in range(len(MATERIAL_NAMES)):
            np.add.at(mat_vote[mid], corner, (material_t == mid) * volumes / mesh.t.shape[0])
    k_node /= mass
    node_material = mat_vote.argmax(axis=0).astype(np.int8)
    top_node = np.isclose(physical_points[1], y_axis[-1])
    metrics = {
        'level': level, 'n_nodes': int(mesh.p.shape[1]), 'n_elements': int(mesh.t.shape[1]),
        'solve_time_s': solve_time, 'ambient_K': ambient, 'h_top_W_m2K': h_top,
        'source_power_W': power_in, 'specified_power_W': float(sum(c.power_W for c in chips)),
        'convective_power_W': power_out,
        'source_power_rel_error': abs(power_in-sum(c.power_W for c in chips))/sum(c.power_W for c in chips),
        'energy_balance_rel_error': abs(power_in-power_out)/power_in,
        'algebraic_residual_rel_l2': float(np.linalg.norm(residual) / np.linalg.norm(f)),
        'pre_normalization_chip_power_W': raw_chip_power,
        'max_source_normalization_rel_correction': max(
            abs(raw_chip_power[c.name]-c.power_W)/c.power_W for c in chips),
        'temperature_min_K': float(temperature.min()), 'temperature_max_K': float(temperature.max()),
        'chiplets': chip_metrics,
    }
    physical_mesh = MeshHex(physical_points, mesh.t.copy())
    return AtplaceResult(physical_mesh, temperature, f, k_node, node_material, material_t, chip_t, layer_t, top_node, metrics)


def save_result(result: AtplaceResult, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    level = result.metrics['level']
    np.savez_compressed(
        out_dir / f'{level}.npz', points_m=result.mesh.p, cells=result.mesh.t,
        temperature_K=result.temperature_K, source_load_W=result.source_load_W,
        conductivity_node_W_mK=result.conductivity_node_W_mK,
        material_id_node=result.material_id_node, material_id_tet=result.material_id_tet,
        chip_id_tet=result.chip_id_tet, layer_id_tet=result.layer_id_tet,
        is_top=result.top_node,
    )


def run_convergence(config_path: Path, out_dir: Path):
    cfg, layout, chips, width, depth = load_problem(config_path)
    results = []
    for level in cfg['mesh_levels']:
        result = solve_level(cfg, chips, width, depth, level)
        save_result(result, out_dir)
        results.append(result)
    rows = [r.metrics for r in results]
    for previous, current in zip(rows, rows[1:]):
        current['delta_peak_from_previous_K'] = abs(current['temperature_max_K']-previous['temperature_max_K'])
        current['max_chip_mean_delta_from_previous_K'] = max(
            abs(current['chiplets'][name]['mean_temperature_K']-previous['chiplets'][name]['mean_temperature_K'])
            for name in current['chiplets'])
    tol = cfg['convergence']
    final = rows[-1]
    checks = {
        'source_power': all(r['source_power_rel_error'] <= tol['source_power_rel_tol'] for r in rows),
        'energy_balance': all(r['energy_balance_rel_error'] <= tol['energy_balance_rel_tol'] for r in rows),
        'minimum_principle': all(r['temperature_min_K'] >= r['ambient_K']-1e-9 for r in rows),
        'peak_temperature': final['delta_peak_from_previous_K'] <= tol['peak_temperature_abs_tol_K'],
        'chip_mean_temperature': final['max_chip_mean_delta_from_previous_K'] <= tol['chip_mean_temperature_abs_tol_K'],
    }
    report = {
        'case_id': layout['case_id'], 'layout_id': layout['layout_id'],
        'layout_provenance': layout['provenance'], 'model': 'six_layer_q1_hexahedral_fem_v1',
        'boundary_model': cfg['boundary'], 'material_names': MATERIAL_NAMES,
        'levels': rows, 'checks': checks, 'all_checks_passed': all(checks.values()),
        'ml_gate': 'ready_for_graph_conversion' if all(checks.values()) else 'blocked_until_convergence_passes',
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'convergence_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='Solve ATPlace Case1 with a six-layer 3-D FEM model')
    parser.add_argument('--config', type=Path, default=Path('configs/atplace_case1_six_layer.yaml'))
    parser.add_argument('--out', type=Path, default=Path('data/atplace_case1_six_layer'))
    args = parser.parse_args()
    report = run_convergence(args.config, args.out)
    for row in report['levels']:
        print(row['level'], row['n_nodes'], row['n_elements'], f"P={row['source_power_W']:.6f} W",
              f"Tmax={row['temperature_max_K']:.4f} K", f"balance={row['energy_balance_rel_error']:.3e}")
    print('checks:', report['checks'])
    if not report['all_checks_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
