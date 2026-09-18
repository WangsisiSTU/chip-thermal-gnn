import json
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('skfem')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from atplace_six_layer_fem import _axis_with_edges, load_problem, material_constants
from atplace_six_layer_fvm import solve_level
from atplace_nonlinear_operator import (
    dynamic_conductances, nonlinear_relative_residual_loss, nonlinear_residual,
    nonlinear_residual_loss,
    nonlinear_warm_start_solve,
)
from evaluate_atplace_prediction import evaluate_array
from generate_atplace_ml_dataset import graph_from_arrays
from train_atplace_ml import (
    PhysicsCorrectedSAGE, aggregate_seed_results, build_atplace_model,
    contact_jump_loss,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/atplace_case1_six_layer.yaml'
ADVANCED_CONFIG = ROOT / 'configs/atplace_case1_contact_material.yaml'


def test_axis_merges_float_aliases_and_keeps_outer_boundary():
    axis = _axis_with_edges(0.041999999999999996, 0.006, [0.012, 0.012000000000000002])
    assert axis[-1] == 0.042
    assert np.count_nonzero(np.isclose(axis, 0.012, atol=1e-14)) == 1
    assert np.diff(axis).min() > 1e-6


def test_atplace_effective_materials_are_positive_and_ordered():
    cfg, _, _, _, _ = load_problem(CONFIG)
    k = material_constants(cfg)
    assert len(k) == 8
    assert np.all(k > 0)
    assert k[5] == 100.0 and k[7] == 4.0
    assert 1.6 < k[3] < 400.0


def test_case1_coarse_reference_conserves_power_and_respects_minimum_principle():
    cfg, _, chips, width, depth = load_problem(CONFIG)
    metrics, arrays = solve_level(cfg, chips, width, depth, 'coarse')
    assert metrics['source_power_W'] == pytest.approx(780.0, rel=1e-12)
    assert metrics['energy_balance_rel_error'] < 1e-10
    assert metrics['minimum_principle_margin_K'] >= -1e-9
    assert set(np.unique(arrays['layer_id'])) == set(range(6))
    assert set(np.unique(arrays['chip_id'])) >= {-1, 0, 1, 2, 3, 4, 5}


def test_advanced_material_solver_resolves_contact_jump_anisotropy_and_temperature_dependence():
    cfg, _, chips, width, depth = load_problem(ADVANCED_CONFIG)
    metrics, arrays = solve_level(cfg, chips, width, depth, 'coarse')
    assert metrics['energy_balance_rel_error'] < 1e-10
    assert metrics['minimum_principle_margin_K'] >= -1e-9
    assert metrics['nonlinear_iterations'] > 1
    assert metrics['anisotropy_ratio_max'] > 1.0
    assert metrics['n_contact_faces'] > 0
    assert metrics['contact_temperature_jump_max_abs_K'] > 0.0
    assert arrays['conductivity_reference_diag_W_mK'].shape == (metrics['n_cells'], 3)
    assert np.any(np.ptp(arrays['conductivity_reference_diag_W_mK'], axis=1) > 0)
    assert np.any(arrays['temperature_coefficient_per_K'] != 0)
    assert arrays['contact_pair_index'].shape[1] == metrics['n_contact_faces']

    without_contact = copy.deepcopy(cfg)
    without_contact['advanced_physics']['contact_resistance']['interfaces'] = []
    no_contact_metrics, _ = solve_level(without_contact, chips, width, depth, 'coarse')
    assert metrics['temperature_max_K'] > no_contact_metrics['temperature_max_K']


def test_advanced_graph_contract_and_nonlinear_energy_projection():
    import torch
    from torch_geometric.data import Batch

    cfg, _, chips, width, depth = load_problem(ADVANCED_CONFIG)
    _, arrays = solve_level(cfg, chips, width, depth, 'coarse')
    height = sum(float(layer['thickness_m']) for layer in cfg['layers'])
    data = graph_from_arrays(
        arrays, width, height, depth, 0, 0, 'test', 'coarse', 0,
        advanced_material_features=True)
    assert data.x.shape[1] == 32
    assert data.edge_attr.shape[1] == 7
    assert data.contact_pair_index.shape[1] > 0
    assert data.edge_axis.shape[0] == data.edge_index.shape[1]
    assert data.edge_half_distance_src.shape == data.edge_axis.shape
    assert data.conductivity_reference_diag.shape == (data.num_nodes, 3)
    assert float(contact_jump_loss(data.y, data, 0.0, 1.0)) < 1e-10
    batch = Batch.from_data_list([data, data.clone()])
    assert float(contact_jump_loss(batch.y, batch, 0.0, 1.0)) < 1e-10

    model = PhysicsCorrectedSAGE(data.x.shape[1], 8, 1, 0.0, 8.0, 7.0, 0, 0.65)
    with torch.no_grad():
        theta = model(data)*7.0+8.0
    absolute_temperature = theta + data.t_ambient[0]
    bounds = data.conductivity_factor_bounds.reshape(2)
    factor = 1.0 + data.temperature_coefficient * (
        absolute_temperature-data.conductivity_reference_temperature[0])
    factor = torch.maximum(torch.minimum(factor, bounds[1]), bounds[0])
    k_y = data.conductivity_reference_y*factor
    top_g = data.top_face_area / (
        data.top_half_distance/k_y + 1.0/data.boundary_h[0])
    relative_balance = abs(float(top_g @ theta-data.source_power.sum())) / float(
        data.source_power.sum())
    assert relative_balance < 1e-6

    residual, _, edge_g, _, _ = nonlinear_residual(data.y, data)
    assert float(torch.linalg.vector_norm(residual)/torch.linalg.vector_norm(data.source_power)) < 1e-4
    assert torch.allclose(
        edge_g, torch.from_numpy(arrays['edge_conductance_W_K']).float(),
        rtol=2e-5, atol=1e-7)
    exact_loss = nonlinear_residual_loss(data.y, data, 0.0, 1.0)
    perturbed_loss = nonlinear_residual_loss(data.y+0.5, data, 0.0, 1.0)
    assert float(exact_loss) < float(perturbed_loss)
    exact_relative_loss = nonlinear_relative_residual_loss(data.y, data, 0.0, 1.0)
    perturbed_relative_loss = nonlinear_relative_residual_loss(
        data.y+0.5, data, 0.0, 1.0)
    assert float(exact_relative_loss) < float(perturbed_relative_loss)

    corrected = nonlinear_warm_start_solve(data, data.y.numpy()+0.5)
    assert np.max(np.abs(corrected.temperature_rise_K-data.y.numpy())) < 2e-4
    assert corrected.final_residual_rel_l2 < 1e-7
    assert corrected.energy_balance_rel_error < 1e-8


def test_generated_full_report_passes_when_present():
    path = ROOT / 'data/atplace_case1_six_layer_fvm/convergence_report.json'
    if not path.exists():
        pytest.skip('run atplace_six_layer_fvm.py first')
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['all_checks_passed']
    assert report['ml_gate'] == 'ready_for_graph_conversion'


def test_exact_prediction_has_zero_ml_comparison_error():
    reference = ROOT / 'data/atplace_case1_six_layer_fvm/coarse.npz'
    if not reference.exists():
        pytest.skip('run atplace_six_layer_fvm.py first')
    raw = np.load(reference)
    metrics = evaluate_array(reference, raw['temperature_K']-318.15, 'rise_K')
    assert metrics['volume_weighted_mae_K'] < 1e-12
    assert metrics['energy_balance_rel_error'] < 1e-10


def test_exact_advanced_prediction_uses_temperature_dependent_boundary_operator():
    reference = ROOT / 'data/atplace_case1_contact_material/coarse.npz'
    if not reference.exists():
        pytest.skip('run advanced atplace_six_layer_fvm.py first')
    raw = np.load(reference)
    metrics = evaluate_array(
        reference, raw['temperature_K']-float(raw['ambient_K']), 'rise_K')
    assert metrics['volume_weighted_mae_K'] < 1e-12
    assert metrics['energy_balance_rel_error'] < 1e-10


def test_generated_ml_splits_are_grouped_and_reference_contract_matches():
    data_dir = ROOT / 'data/atplace_case1_ml'
    reference_meta = ROOT / 'data/atplace_case1_ml_reference/metadata.json'
    if not (data_dir / 'metadata.json').exists() or not reference_meta.exists():
        pytest.skip('generate ATPlace ML artifacts first')
    sets = {s: __import__('torch').load(data_dir/f'{s}.pt', weights_only=False)
            for s in ('train', 'val', 'test')}
    ids = {s: {int(d.layout_id) for d in values} for s, values in sets.items()}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    assert all(sum(int(d.layout_id) == lid for d in sets[s]) == 4 for s in ids for lid in ids[s])
    meta = json.loads((data_dir/'metadata.json').read_text(encoding='utf-8'))
    ref = json.loads(reference_meta.read_text(encoding='utf-8'))
    assert meta['node_feature_names'] == ref['node_feature_names']
    assert meta['edge_feature_names'] == ref['edge_feature_names']


def test_energy_projection_enforces_graph_level_balance():
    import torch
    data_path = ROOT / 'data/atplace_case1_ml/train.pt'
    if not data_path.exists():
        pytest.skip('generate ATPlace ML dataset first')
    data = torch.load(data_path, weights_only=False)[0]
    model = PhysicsCorrectedSAGE(data.x.shape[1], 8, 1, 0.0, 8.0, 7.0, 0, 0.65)
    with torch.no_grad():
        theta = model(data)*7.0+8.0
    rejected = (data.top_conductance*theta).sum()
    assert float(abs(rejected-data.source_power.sum())/data.source_power.sum()) < 1e-6


@pytest.mark.parametrize('name', ['graphsage', 'meshgraphnet', 'mgn_transolver'])
def test_all_atplace_graph_models_share_energy_projection(name):
    import torch
    import yaml
    from torch_geometric.data import Data

    cfg = yaml.safe_load((ROOT/'configs/atplace_case1_mgn_transolver.yaml').read_text(encoding='utf-8'))
    metadata = {'node_feature_dim': 28, 'edge_feature_dim': 5,
                'dT_train_mean': 8.0, 'dT_train_std': 4.0}
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    data = Data(
        x=torch.zeros(4, 28), edge_index=edge_index, edge_attr=torch.zeros(6, 5),
        node_volume=torch.ones(4), edge_conductance=torch.ones(6),
        source_power=torch.tensor([2.0, 0.0, 0.0, 0.0]),
        top_conductance=torch.tensor([0.0, 0.0, 0.0, 0.5]),
        thermal_diag=torch.tensor([1.0, 2.0, 2.0, 1.5]),
    )
    model = build_atplace_model(name, metadata, cfg).eval()
    with torch.no_grad():
        theta = model(data)*metadata['dT_train_std']+metadata['dT_train_mean']
    rejected = (data.top_conductance*theta).sum()
    assert theta.shape == (4,)
    assert float(abs(rejected-data.source_power.sum())) < 1e-5


def test_seed_aggregation_includes_contact_jump_when_available():
    metric_names = (
        'volume_weighted_mae_K', 'volume_weighted_mae_p95_K',
        'volume_weighted_rmse_K', 'mean_absolute_peak_error_K',
        'mean_chip_mean_error_K', 'mean_energy_balance_rel_error',
        'layout_mean_mae_K', 'layout_worst_mae_K', 'inference_ms_per_graph',
        'mean_contact_jump_relative_l2')
    suite = {name: 1.0 for name in metric_names}
    seeds = {
        str(seed): {'graphsage': {'run': {}, 'suites': {'layout_only_medium': suite}}}
        for seed in (1, 2, 3)
    }
    aggregate = aggregate_seed_results(seeds)
    assert 'mean_contact_jump_relative_l2' in aggregate['graphsage']['layout_only_medium']
