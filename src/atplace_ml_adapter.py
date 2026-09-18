"""Convert a converged ATPlace finite-volume reference into a PyG graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from atplace_six_layer_fem import load_problem
from generate_atplace_ml_dataset import (
    ADVANCED_EDGE_FEATURE_NAMES, ADVANCED_NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, graph_from_arrays,
)


def convert(reference_dir: Path, out_dir: Path, level: str = 'verification',
            config_path: Path = Path('configs/atplace_case1_six_layer.yaml'),
            advanced_material_features: bool = False):
    report = json.loads((reference_dir / 'convergence_report.json').read_text(encoding='utf-8'))
    if not report.get('all_checks_passed') or report.get('ml_gate') != 'ready_for_graph_conversion':
        raise ValueError('reference solution has not passed the ML quality gate')
    raw = np.load(reference_dir / f'{level}.npz')
    config_path = config_path.resolve()
    cfg, _, _, width, depth = load_problem(config_path)
    height = sum(float(x['thickness_m']) for x in cfg['layers'])
    data = graph_from_arrays(raw, width, height, depth, -1, 0, 'reference', level, -1,
                             advanced_material_features)
    data.t_ambient = torch.tensor([float(raw['ambient_K'])], dtype=torch.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_dir / f'{level}.pt')
    metadata = {
        'case_id': report['case_id'], 'layout_id': report['layout_id'], 'level': level,
        'role': 'reference_test_only',
        'training_warning': 'One Case1 graph is not a training dataset. Generate layout/power/material variants with grouped splits.',
        'node_feature_names': ADVANCED_NODE_FEATURE_NAMES if advanced_material_features else NODE_FEATURE_NAMES,
        'edge_feature_names': ADVANCED_EDGE_FEATURE_NAMES if advanced_material_features else EDGE_FEATURE_NAMES,
        'node_feature_dim': len(ADVANCED_NODE_FEATURE_NAMES if advanced_material_features else NODE_FEATURE_NAMES),
        'edge_feature_dim': len(ADVANCED_EDGE_FEATURE_NAMES if advanced_material_features else EDGE_FEATURE_NAMES),
        'advanced_material_features': advanced_material_features,
        'target': 'temperature_rise_K', 'n_nodes': int(data.num_nodes),
        'n_directed_edges': int(data.edge_index.shape[1]), 'source_report': str(reference_dir / 'convergence_report.json'),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Build ML reference graph from converged ATPlace Case1')
    parser.add_argument('--reference', type=Path, default=Path('data/atplace_case1_six_layer_fvm'))
    parser.add_argument('--out', type=Path, default=Path('data/atplace_case1_ml_reference'))
    parser.add_argument('--level', default='verification')
    parser.add_argument('--config', type=Path, default=Path('configs/atplace_case1_six_layer.yaml'))
    parser.add_argument('--advanced-material-features', action='store_true')
    args = parser.parse_args()
    print(json.dumps(convert(args.reference, args.out, args.level, args.config,
                             args.advanced_material_features), indent=2))


if __name__ == '__main__':
    main()
