"""Generate grouped-layout Case1 power sweeps for the first ML comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import yaml
from torch_geometric.data import Data

from atplace_six_layer_fem import Chiplet, MATERIAL_NAMES, load_problem
from atplace_six_layer_fvm import solve_level


NODE_FEATURE_NAMES = (
    ['x_norm', 'y_norm', 'z_norm', 'log10_k_norm', 'q_density_scaled',
     'source_power_scaled', 'total_power_scaled'] +
    [f'material_{x}' for x in MATERIAL_NAMES] +
    [f'layer_{i}' for i in range(6)] + ['chip_none'] + [f'chip_{i}' for i in range(6)]
)
EDGE_FEATURE_NAMES = ['dx_norm', 'dy_norm', 'dz_norm', 'distance_norm', 'log10_conductance_scaled']
ADVANCED_NODE_FEATURE_NAMES = NODE_FEATURE_NAMES + [
    'log10_kx_norm', 'log10_ky_norm', 'log10_kz_norm',
    'temperature_coefficient_scaled',
]
ADVANCED_EDGE_FEATURE_NAMES = EDGE_FEATURE_NAMES + [
    'contact_resistance_scaled', 'is_contact_interface',
]


def overlap(a: Chiplet, b: Chiplet) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.z0 < b.z1 and b.z0 < a.z1


def random_layout(base: list[Chiplet], width: float, depth: float, rng, grid: float, attempts: int):
    chips = list(base)
    accepted = 0
    for _ in range(attempts):
        i = int(rng.integers(len(chips)))
        old = chips[i]
        w, d = old.x1-old.x0, old.z1-old.z0
        if rng.random() < 0.5:
            w, d = d, w
        nx, nz = int(np.floor((width-w)/grid)), int(np.floor((depth-d)/grid))
        if min(nx, nz) < 0:
            continue
        x0, z0 = int(rng.integers(nx+1))*grid, int(rng.integers(nz+1))*grid
        candidate = Chiplet(old.name, x0, x0+w, z0, z0+d, old.power_W)
        if any(j != i and overlap(candidate, other) for j, other in enumerate(chips)):
            continue
        chips[i] = candidate
        accepted += 1
    if accepted == 0:
        raise ValueError('layout randomization accepted no legal moves')
    return chips, accepted


def graph_from_arrays(arrays, width: float, height: float, depth: float, layout_id: int,
                      profile_id: int, split: str, mesh_level: str = 'unknown',
                      physical_case_id: int | None = None,
                      advanced_material_features: bool = False):
    pos, volume = arrays['pos_m'], arrays['cell_volume_m3']
    source = arrays['source_power_W']
    if advanced_material_features:
        if 'conductivity_reference_diag_W_mK' not in arrays:
            raise ValueError('advanced material features require diagonal reference conductivity')
        k_diagonal = arrays['conductivity_reference_diag_W_mK']
        k = np.cbrt(np.prod(k_diagonal, axis=1))
    else:
        k = arrays['conductivity_W_mK']
    material, layer, chip = arrays['material_id'], arrays['layer_id'], arrays['chip_id']
    total_power = float(source.sum())
    pos_norm = pos / np.asarray([width, height, depth])
    logk_norm = (np.log10(k)-np.log10(0.3)) / (np.log10(400.0)-np.log10(0.3))
    q_density = source / volume
    onehot_material = np.eye(8)[material]
    onehot_layer = np.eye(6)[layer]
    onehot_chip = np.eye(7)[chip+1]
    node_columns = [
        pos_norm, logk_norm, q_density/1.0e10, source/10.0,
        np.full(pos.shape[0], total_power/780.0), onehot_material, onehot_layer, onehot_chip,
    ]
    if advanced_material_features:
        logk_diagonal = ((np.log10(k_diagonal)-np.log10(0.3)) /
                         (np.log10(400.0)-np.log10(0.3)))
        node_columns.extend((logk_diagonal,
                             arrays['temperature_coefficient_per_K'][:, None]/0.005))
    x = np.column_stack(node_columns).astype(np.float32)
    edge_index = arrays['edge_index'].astype(np.int64)
    src, dst = edge_index
    delta = (pos[dst]-pos[src]) / np.asarray([width, height, depth])
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    edge_g = (arrays['edge_conductance_reference_W_K']
              if 'edge_conductance_reference_W_K' in arrays else arrays['edge_conductance_W_K'])
    edge_columns = [delta, distance, np.log10(edge_g)[:, None]/8.0]
    if advanced_material_features:
        edge_r = arrays['edge_contact_resistance_m2K_W']
        edge_columns.extend((np.log10(1.0+edge_r/1.0e-6)[:, None]/3.0,
                             (edge_r > 0)[:, None].astype(float)))
    edge_attr = np.column_stack(edge_columns).astype(np.float32)
    top_g_node = np.zeros(pos.shape[0], dtype=np.float32)
    top_g = (arrays['top_conductance_reference_W_K']
             if 'top_conductance_reference_W_K' in arrays else arrays['top_conductance_W_K'])
    top_g_node[arrays['top_cell']] = top_g.astype(np.float32)
    diag = np.zeros(pos.shape[0], dtype=np.float64)
    np.add.at(diag, src, edge_g)
    diag += top_g_node
    data = Data(
        x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy((arrays['temperature_K']-float(arrays['ambient_K'])).astype(np.float32)),
        pos=torch.from_numpy(pos.astype(np.float32)), node_volume=torch.from_numpy(volume.astype(np.float32)),
        source_power=torch.from_numpy(source.astype(np.float32)), chip_id=torch.from_numpy(chip.astype(np.int64)),
        edge_conductance=torch.from_numpy(edge_g.astype(np.float32)),
        top_conductance=torch.from_numpy(top_g_node), thermal_diag=torch.from_numpy(diag.astype(np.float32)),
    )
    data.layout_id = torch.tensor([layout_id], dtype=torch.long)
    data.power_profile_id = torch.tensor([profile_id], dtype=torch.long)
    data.physical_case_id = torch.tensor(
        [physical_case_id if physical_case_id is not None else profile_id], dtype=torch.long)
    data.total_power_W = torch.tensor([total_power], dtype=torch.float32)
    data.t_ambient = torch.tensor([float(arrays['ambient_K'])], dtype=torch.float32)
    data.split_name = split
    data.mesh_level_name = mesh_level
    if advanced_material_features:
        top_area_node = np.zeros(pos.shape[0], dtype=np.float32)
        top_half_node = np.zeros(pos.shape[0], dtype=np.float32)
        top_area_node[arrays['top_cell']] = arrays['top_face_area_m2'].astype(np.float32)
        top_half_node[arrays['top_cell']] = arrays['top_half_distance_m'].astype(np.float32)
        data.top_face_area = torch.from_numpy(top_area_node)
        data.top_half_distance = torch.from_numpy(top_half_node)
        data.conductivity_reference_y = torch.from_numpy(k_diagonal[:, 1].astype(np.float32))
        data.conductivity_reference_diag = torch.from_numpy(k_diagonal.astype(np.float32))
        data.temperature_coefficient = torch.from_numpy(
            arrays['temperature_coefficient_per_K'].astype(np.float32))
        data.boundary_h = torch.tensor([float(arrays['boundary_h_W_m2K'])], dtype=torch.float32)
        data.conductivity_reference_temperature = torch.tensor(
            [float(arrays['conductivity_reference_temperature_K'])], dtype=torch.float32)
        data.conductivity_factor_bounds = torch.from_numpy(
            arrays['conductivity_factor_bounds'].astype(np.float32)[None, :])
        required_operator_geometry = (
            'edge_axis', 'edge_half_distance_src_m', 'edge_half_distance_dst_m',
            'edge_face_area_m2', 'edge_contact_resistance_m2K_W')
        missing = [name for name in required_operator_geometry if name not in arrays]
        if missing:
            raise ValueError(
                f'advanced material graph requires nonlinear operator geometry: {missing}')
        data.edge_axis = torch.from_numpy(arrays['edge_axis'].astype(np.int64))
        data.edge_half_distance_src = torch.from_numpy(
            arrays['edge_half_distance_src_m'].astype(np.float32))
        data.edge_half_distance_dst = torch.from_numpy(
            arrays['edge_half_distance_dst_m'].astype(np.float32))
        data.edge_face_area = torch.from_numpy(arrays['edge_face_area_m2'].astype(np.float32))
        data.edge_contact_resistance = torch.from_numpy(
            arrays['edge_contact_resistance_m2K_W'].astype(np.float32))
    if 'contact_pair_index' in arrays:
        data.contact_pair_index = torch.from_numpy(arrays['contact_pair_index'].astype(np.int64))
        data.contact_face_area = torch.from_numpy(arrays['contact_face_area_m2'].astype(np.float32))
        data.contact_resistance = torch.from_numpy(
            arrays['contact_resistance_m2K_W'].astype(np.float32))
        contact_g = (arrays['contact_conductance_reference_W_K']
                     if 'contact_conductance_reference_W_K' in arrays
                     else arrays['contact_conductance_W_K'])
        data.contact_conductance = torch.from_numpy(contact_g.astype(np.float32))
        data.contact_temperature_jump_target = torch.from_numpy(
            (arrays['contact_temperature_jump_reference_operator_K']
             if 'contact_temperature_jump_reference_operator_K' in arrays
             else arrays['contact_temperature_jump_K']).astype(np.float32))
        data.contact_temperature_jump_physical = torch.from_numpy(
            arrays['contact_temperature_jump_K'].astype(np.float32))
    return data


def generate(config_path: Path, out_dir: Path):
    run_cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    root = config_path.resolve().parents[1]
    physics_path = Path(run_cfg['physics_config'])
    if not physics_path.is_absolute():
        physics_path = root / physics_path
    physics, _, base, width, depth = load_problem(physics_path)
    height = sum(float(x['thickness_m']) for x in physics['layers'])
    rng = np.random.default_rng(int(run_cfg['seed']))
    splits = {name: [] for name in ('train', 'val', 'test')}
    evaluation = run_cfg.get('evaluation', {})
    evaluation_level = evaluation.get('mesh_level')
    evaluation_sets = {'mesh_only': [], 'layout_mesh': []}
    advanced_material_features = bool(run_cfg.get('advanced_material_features', False))
    mesh_only_split = evaluation.get('mesh_only_source_split', 'train')
    mesh_only_layouts = int(evaluation.get('mesh_only_layouts', 0))
    layout_mesh_split = evaluation.get('layout_mesh_source_split', 'test')
    manifest, layout_id = [], 0
    for split in splits:
        for split_layout_index in range(int(run_cfg['layouts'][split])):
            chips, accepted = random_layout(
                base, width, depth, rng, float(run_cfg['layout_move_grid_m']),
                int(run_cfg['layout_move_attempts']),
            )
            layout_record = {'layout_id': layout_id, 'split': split, 'accepted_moves': accepted,
                             'chiplets': [{'name': c.name, 'origin_m': [c.x0, c.z0],
                                           'size_m': [c.x1-c.x0, c.z1-c.z0]} for c in chips],
                             'power_profiles': []}
            manifest.append(layout_record)
            for profile in range(int(run_cfg['power_profiles_per_layout'])):
                if profile == 0:
                    multipliers = np.ones(len(chips))
                else:
                    bounds = run_cfg['power_multiplier']
                    multipliers = rng.uniform(float(bounds['low']), float(bounds['high']), len(chips))
                powered = [Chiplet(c.name, c.x0, c.x1, c.z0, c.z1, c.power_W*float(multipliers[i]))
                           for i, c in enumerate(chips)]
                physical_case_id = layout_id * int(run_cfg['power_profiles_per_layout']) + profile
                layout_record['power_profiles'].append({
                    'profile_id': profile,
                    'physical_case_id': physical_case_id,
                    'multipliers': [float(x) for x in multipliers],
                    'chip_power_W': [float(c.power_W) for c in powered],
                })
                _, arrays = solve_level(physics, powered, width, depth, run_cfg['mesh_level'])
                splits[split].append(graph_from_arrays(
                    arrays, width, height, depth, layout_id, profile, split,
                    run_cfg['mesh_level'], physical_case_id, advanced_material_features))
                if evaluation_level:
                    target = None
                    if split == mesh_only_split and split_layout_index < mesh_only_layouts:
                        target = 'mesh_only'
                    elif split == layout_mesh_split:
                        target = 'layout_mesh'
                    if target is not None:
                        _, eval_arrays = solve_level(
                            physics, powered, width, depth, evaluation_level)
                        evaluation_sets[target].append(graph_from_arrays(
                            eval_arrays, width, height, depth, layout_id, profile,
                            f'eval_{target}', evaluation_level, physical_case_id,
                            advanced_material_features))
            layout_id += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        torch.save(data, out_dir / f'{split}.pt')
    for name, data in evaluation_sets.items():
        if data:
            torch.save(data, out_dir / f'eval_{name}_{evaluation_level}.pt')
    train_y = torch.cat([d.y for d in splits['train']])
    metadata = {
        'case_id': 'Case1', 'seed': run_cfg['seed'], 'mesh_level': run_cfg['mesh_level'],
        'split_strategy': 'grouped_by_layout_id',
        'layout_counts': run_cfg['layouts'],
        'split_counts': {k: len(v) for k, v in splits.items()},
        'evaluation_mesh_level': evaluation_level,
        'evaluation_counts': {k: len(v) for k, v in evaluation_sets.items()},
        'evaluation_protocol': {
            'layout_only': f'unseen test layouts on training mesh {run_cfg["mesh_level"]}',
            'mesh_only': f'seen training layouts and identical physical cases on {evaluation_level}',
            'layout_mesh': f'unseen test layouts and identical physical cases on {evaluation_level}',
        } if evaluation_level else {},
        'node_feature_dim': len(ADVANCED_NODE_FEATURE_NAMES if advanced_material_features else NODE_FEATURE_NAMES),
        'edge_feature_dim': len(ADVANCED_EDGE_FEATURE_NAMES if advanced_material_features else EDGE_FEATURE_NAMES),
        'node_feature_names': ADVANCED_NODE_FEATURE_NAMES if advanced_material_features else NODE_FEATURE_NAMES,
        'edge_feature_names': ADVANCED_EDGE_FEATURE_NAMES if advanced_material_features else EDGE_FEATURE_NAMES,
        'target': 'temperature_rise_K', 'dT_train_mean': float(train_y.mean()),
        'dT_train_std': float(train_y.std()), 'dimension': 3,
        'node_volume_weighting': True, 'heat_source_projection': 'conservative_cell_power_v1',
        'discrete_operator': 'orthogonal_fvm_conductance_v1',
        'advanced_material_features': advanced_material_features,
        'contact_interface_entities': advanced_material_features,
        'training_scope': 'layout_OOD within Case1; power range 0.70-1.30 per chip',
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out_dir / 'layout_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate grouped ATPlace Case1 ML dataset')
    parser.add_argument('--config', type=Path, default=Path('configs/atplace_case1_ml.yaml'))
    parser.add_argument('--out', type=Path, default=Path('data/atplace_case1_ml'))
    args = parser.parse_args()
    print(json.dumps(generate(args.config, args.out), indent=2))


if __name__ == '__main__':
    main()
