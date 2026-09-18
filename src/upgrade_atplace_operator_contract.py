"""Upgrade existing advanced ATPlace graphs with exact nonlinear FVM geometry."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from atplace_six_layer_fem import Chiplet, load_problem
from atplace_six_layer_fvm import grid_axes


GRAPH_FILES = (
    'train.pt', 'val.pt', 'test.pt',
    'eval_mesh_only_fine.pt', 'eval_layout_mesh_fine.pt',
)


def _nearest_indices(values, centers):
    right = np.searchsorted(centers, values).clip(0, len(centers)-1)
    left = np.maximum(right-1, 0)
    choose_left = np.abs(values-centers[left]) <= np.abs(values-centers[right])
    return np.where(choose_left, left, right)


def upgrade_graph(graph, physics, base_by_name, layout_by_id, width, depth, metadata):
    layout = layout_by_id[int(graph.layout_id)]
    chips = []
    for item in layout['chiplets']:
        x0, z0 = item['origin_m']
        w, d = item['size_m']
        chips.append(Chiplet(item['name'], x0, x0+w, z0, z0+d,
                             base_by_name[item['name']].power_W))
    level = str(graph.mesh_level_name)
    x, y, z = grid_axes(physics, chips, width, depth, level)
    axes = (x, y, z)
    centers = tuple((axis[:-1]+axis[1:])/2 for axis in axes)
    widths = tuple(np.diff(axis) for axis in axes)
    pos = graph.pos.numpy().astype(float)
    indices = [_nearest_indices(pos[:, axis], centers[axis]) for axis in range(3)]
    cell_widths = np.column_stack([widths[axis][indices[axis]] for axis in range(3)])
    reconstructed_volume = np.prod(cell_widths, axis=1)
    if not np.allclose(reconstructed_volume, graph.node_volume.numpy(), rtol=2e-5, atol=1e-18):
        raise ValueError(f'layout {int(graph.layout_id)} {level}: cell-volume reconstruction failed')

    edge_index = graph.edge_index.numpy().astype(np.int64)
    src, dst = edge_index
    delta = np.abs(pos[dst]-pos[src])
    edge_axis = np.argmax(delta, axis=1).astype(np.int64)
    if np.any(np.sum(delta > 1e-10, axis=1) != 1):
        raise ValueError('operator upgrade requires orthogonal cell-neighbor edges')
    distance_src = 0.5*cell_widths[src, edge_axis]
    distance_dst = 0.5*cell_widths[dst, edge_axis]
    all_dimensions = np.prod(cell_widths[src], axis=1)
    edge_area = all_dimensions/cell_widths[src, edge_axis]

    edge_rtc = np.zeros(src.size, dtype=np.float64)
    contact_lookup = {}
    pair = graph.contact_pair_index.numpy().astype(np.int64)
    resistance = graph.contact_resistance.numpy().astype(float)
    for index in range(pair.shape[1]):
        a, b = int(pair[0, index]), int(pair[1, index])
        contact_lookup[(a, b)] = resistance[index]
        contact_lookup[(b, a)] = resistance[index]
    for index, nodes in enumerate(zip(src.tolist(), dst.tolist())):
        edge_rtc[index] = contact_lookup.get(nodes, 0.0)

    feature_names = metadata['node_feature_names']
    conductivity_columns = [feature_names.index(f'log10_k{axis}_norm') for axis in 'xyz']
    normalized = graph.x[:, conductivity_columns].numpy().astype(float)
    log_min, log_max = np.log10(0.3), np.log10(400.0)
    conductivity = 10.0**(normalized*(log_max-log_min)+log_min)

    graph.edge_axis = torch.from_numpy(edge_axis)
    graph.edge_half_distance_src = torch.from_numpy(distance_src.astype(np.float32))
    graph.edge_half_distance_dst = torch.from_numpy(distance_dst.astype(np.float32))
    graph.edge_face_area = torch.from_numpy(edge_area.astype(np.float32))
    graph.edge_contact_resistance = torch.from_numpy(edge_rtc.astype(np.float32))
    graph.conductivity_reference_diag = torch.from_numpy(conductivity.astype(np.float32))
    return graph


def upgrade(source_dir: Path, output_dir: Path, config_path: Path):
    metadata = json.loads((source_dir/'metadata.json').read_text(encoding='utf-8'))
    if not metadata.get('advanced_material_features'):
        raise ValueError('operator upgrade requires an advanced-material graph dataset')
    manifest = json.loads((source_dir/'layout_manifest.json').read_text(encoding='utf-8'))
    layout_by_id = {int(item['layout_id']): item for item in manifest}
    physics, _, base, width, depth = load_problem(config_path)
    base_by_name = {chip.name: chip for chip in base}
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for filename in GRAPH_FILES:
        source = source_dir/filename
        if not source.exists():
            continue
        graphs = torch.load(source, weights_only=False)
        upgraded = [upgrade_graph(
            graph, physics, base_by_name, layout_by_id, width, depth, metadata)
            for graph in graphs]
        torch.save(upgraded, output_dir/filename)
        counts[filename] = len(upgraded)
    metadata = dict(metadata)
    metadata['discrete_operator'] = 'temperature_consistent_orthogonal_fvm_v2'
    metadata['nonlinear_operator_geometry'] = True
    (output_dir/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    shutil.copy2(source_dir/'layout_manifest.json', output_dir/'layout_manifest.json')
    return {'output': str(output_dir), 'graphs': counts,
            'operator': metadata['discrete_operator']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('data/atplace_case1_contact_material_ml'))
    parser.add_argument('--out', type=Path,
                        default=Path('data/atplace_case1_contact_material_ml_operator'))
    parser.add_argument('--config', type=Path,
                        default=Path('configs/atplace_case1_contact_material.yaml'))
    args = parser.parse_args()
    print(json.dumps(upgrade(args.source, args.out, args.config), indent=2))


if __name__ == '__main__':
    main()
