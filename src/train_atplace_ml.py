"""Train ATPlace Case1 graph models and evaluate layout/mesh generalization."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from atplace_nonlinear_operator import (
    nonlinear_relative_residual_loss, nonlinear_residual,
    nonlinear_residual_loss, require_operator_contract,
)
from models import BaselineGNN, MeshGraphNet, MGNTransolverHybrid


class NodeMLP(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()
        modules = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers-1):
            modules += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, data):
        return self.net(data.x).squeeze(-1)


class EnergyProjectedModel(nn.Module):
    """Apply the same local correction and exact global energy projection to any model."""

    def __init__(self, base, mean, std, steps=0, omega=0.65):
        super().__init__()
        self.base = base
        self.register_buffer('mean', torch.tensor(float(mean)))
        self.register_buffer('std', torch.tensor(float(std)))
        self.steps, self.omega = int(steps), float(omega)

    def forward(self, data):
        pred = self.base(data)
        theta = pred*self.std + self.mean
        src, dst = data.edge_index
        for _ in range(self.steps):
            conductive = scatter(
                data.edge_conductance*(theta[src]-theta[dst]), src,
                dim=0, dim_size=theta.numel(), reduce='sum')
            residual = conductive + data.top_conductance*theta - data.source_power
            theta = theta - self.omega*residual/data.thermal_diag.clamp_min(1e-12)
        # A constant temperature shift leaves every conductive edge flux
        # unchanged.  Project that null mode so each graph rejects exactly its
        # supplied power through the Robin boundary.
        has_batch = getattr(data, 'batch', None) is not None
        graph = data.batch if has_batch else torch.zeros_like(theta, dtype=torch.long)
        n_graphs = int(data.num_graphs) if has_batch else 1
        supplied = scatter(data.source_power, graph, dim=0, dim_size=n_graphs, reduce='sum')
        if all(getattr(data, name, None) is not None for name in (
                'top_face_area', 'top_half_distance', 'conductivity_reference_y',
                'temperature_coefficient', 'boundary_h',
                'conductivity_reference_temperature', 'conductivity_factor_bounds',
                't_ambient')):
            # With k(T), a constant temperature shift also changes the Robin
            # face conductance.  Recompute it during a few fixed-point updates
            # so the projection closes the *physical* nonlinear boundary
            # balance rather than the reference-temperature approximation.
            boundary_h = data.boundary_h.reshape(-1)
            reference_temperature = data.conductivity_reference_temperature.reshape(-1)
            ambient = data.t_ambient.reshape(-1)
            bounds = data.conductivity_factor_bounds.reshape(n_graphs, 2)
            shift = theta.new_zeros(n_graphs)
            for _ in range(6):
                shifted = theta + shift[graph]
                factor = 1.0 + data.temperature_coefficient * (
                    shifted + ambient[graph] - reference_temperature[graph])
                factor = torch.maximum(
                    torch.minimum(factor, bounds[graph, 1]), bounds[graph, 0])
                k_y = data.conductivity_reference_y*factor
                top_conductance = data.top_face_area / (
                    data.top_half_distance/k_y.clamp_min(1e-12) +
                    1.0/boundary_h[graph].clamp_min(1e-12))
                rejected = scatter(
                    top_conductance*shifted, graph, dim=0,
                    dim_size=n_graphs, reduce='sum')
                conductance = scatter(
                    top_conductance, graph, dim=0,
                    dim_size=n_graphs, reduce='sum')
                shift = shift + (supplied-rejected)/conductance.clamp_min(1e-12)
            theta = theta + shift[graph]
        else:
            rejected = scatter(
                data.top_conductance*theta, graph, dim=0,
                dim_size=n_graphs, reduce='sum')
            conductance = scatter(
                data.top_conductance, graph, dim=0,
                dim_size=n_graphs, reduce='sum')
            theta = theta + ((supplied-rejected)/conductance.clamp_min(1e-12))[graph]
        return (theta-self.mean)/self.std


class PhysicsCorrectedSAGE(EnergyProjectedModel):
    """Backward-compatible name for the original energy-projected GraphSAGE."""

    def __init__(self, in_dim, hidden, layers, dropout, mean, std, steps, omega):
        super().__init__(BaselineGNN(in_dim, hidden, layers, dropout, 'sage'),
                         mean, std, steps, omega)


def build_atplace_model(name, metadata, cfg):
    tcfg = cfg['training']
    in_dim = int(metadata['node_feature_dim'])
    edge_dim = int(metadata['edge_feature_dim'])
    hidden = int(tcfg['hidden_dim'])
    layers = int(tcfg['layers'])
    dropout = float(tcfg['dropout'])
    if name in ('graphsage', 'physics_graphsage'):
        base = BaselineGNN(in_dim, hidden, layers, dropout, 'sage')
    elif name == 'meshgraphnet':
        mcfg = cfg['meshgraphnet']
        base = MeshGraphNet(
            in_dim, edge_dim, hidden_dim=int(mcfg.get('hidden_dim', hidden)),
            n_message_passing_steps=int(mcfg['message_passing_steps']),
            dropout=float(mcfg.get('dropout', dropout)),
            aggregation=mcfg.get('aggregation', 'sum'))
    elif name == 'mgn_transolver':
        hcfg = cfg['mgn_transolver']
        base = MGNTransolverHybrid(
            in_dim, edge_dim, hidden_dim=int(hcfg.get('hidden_dim', hidden)),
            n_message_passing_steps=int(hcfg['message_passing_steps']),
            attention_blocks=int(hcfg['attention_blocks']),
            attention_heads=int(hcfg['attention_heads']),
            attention_slices=int(hcfg['attention_slices']),
            dropout=float(hcfg.get('dropout', dropout)),
            attention_mode=hcfg.get('attention_mode', 'full'),
            aggregation=hcfg.get('aggregation', 'sum'),
            global_gate=bool(hcfg.get('global_gate', True)),
            operator_correction_steps=0)
    elif name == 'node_mlp':
        base = NodeMLP(in_dim, hidden, layers, dropout)
    else:
        raise ValueError(name)
    if bool(tcfg.get('energy_projection', name == 'physics_graphsage')):
        return EnergyProjectedModel(
            base, metadata['dT_train_mean'], metadata['dT_train_std'],
            tcfg.get('physics_correction_steps', 0),
            tcfg.get('physics_correction_omega', 0.65))
    return base


def contact_jump_loss(pred, batch, mean, std):
    required = ('contact_pair_index', 'contact_face_area', 'contact_resistance',
                'contact_conductance', 'contact_temperature_jump_target')
    if any(getattr(batch, name, None) is None for name in required):
        raise ValueError('contact-jump loss requires explicit contact-face entities')
    pair = batch.contact_pair_index.long()
    if pair.shape[1] == 0:
        return pred.new_zeros(())
    theta = pred*std+mean
    flow = batch.contact_conductance*(theta[pair[0]]-theta[pair[1]])
    predicted_jump = flow/batch.contact_face_area.clamp_min(1e-20)*batch.contact_resistance
    target = batch.contact_temperature_jump_target
    has_batch = getattr(batch, 'batch', None) is not None
    face_graph = (batch.batch[pair[0]] if has_batch else
                  torch.zeros(pair.shape[1], dtype=torch.long, device=pair.device))
    n_graphs = int(batch.num_graphs) if has_batch else 1
    error = scatter(batch.contact_face_area*(predicted_jump-target).square(), face_graph,
                    dim=0, dim_size=n_graphs, reduce='sum')
    signal = scatter(batch.contact_face_area*target.square(), face_graph,
                     dim=0, dim_size=n_graphs, reduce='sum').clamp_min(1e-16)
    return (error/signal).mean()


def graph_loss(pred, target, batch, peak_weight, contact_jump_weight=0.0,
               operator_residual_weight=0.0,
               operator_relative_residual_weight=0.0,
               mean=None, std=None):
    volume = batch.node_volume
    graph = batch.batch
    n = batch.num_graphs
    mass = scatter(volume, graph, dim=0, dim_size=n, reduce='sum')
    mse = scatter(volume*(pred-target).square(), graph, dim=0, dim_size=n, reduce='sum')/mass
    pred_peak = scatter(pred, graph, dim=0, dim_size=n, reduce='max')
    true_peak = scatter(target, graph, dim=0, dim_size=n, reduce='max')
    loss = mse.mean() + peak_weight*(pred_peak-true_peak).square().mean()
    if contact_jump_weight > 0:
        loss = loss + contact_jump_weight*contact_jump_loss(pred, batch, mean, std)
    if operator_residual_weight > 0:
        loss = loss + operator_residual_weight*nonlinear_residual_loss(
            pred, batch, mean, std)
    if operator_relative_residual_weight > 0:
        loss = loss + operator_relative_residual_weight*nonlinear_relative_residual_loss(
            pred, batch, mean, std)
    return loss


@torch.no_grad()
def validation_loss(model, loader, device, mean, std, peak_weight, contact_jump_weight=0.0,
                    operator_residual_weight=0.0, operator_relative_residual_weight=0.0):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = (batch.y-mean)/std
        loss = graph_loss(
            pred, target, batch, peak_weight, contact_jump_weight,
            operator_residual_weight, operator_relative_residual_weight, mean, std)
        total += float(loss)*batch.num_graphs
        count += batch.num_graphs
    return total/count


@torch.no_grad()
def evaluate(model, loader, device, mean, std):
    model.eval()
    first = next(iter(loader)).to(device)
    for _ in range(3):
        model(first)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        batch = batch.to(device)
        model(batch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-started

    cases = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)*std+mean
        true = batch.y
        operator_residual = None
        try:
            require_operator_contract(batch)
            operator_residual = nonlinear_residual(pred, batch)[0]
        except ValueError:
            pass
        for gid in range(batch.num_graphs):
            mask = batch.batch == gid
            volume = batch.node_volume[mask]
            error = pred[mask]-true[mask]
            mae = float((volume*error.abs()).sum()/volume.sum())
            rmse = float(torch.sqrt((volume*error.square()).sum()/volume.sum()))
            peak = float((pred[mask].max()-true[mask].max()).abs())
            if all(getattr(batch, name, None) is not None for name in (
                    'top_face_area', 'top_half_distance', 'conductivity_reference_y',
                    'temperature_coefficient', 'boundary_h',
                    'conductivity_reference_temperature', 'conductivity_factor_bounds',
                    't_ambient')):
                bounds = batch.conductivity_factor_bounds.reshape(batch.num_graphs, 2)[gid]
                absolute_temperature = pred[mask] + batch.t_ambient.reshape(-1)[gid]
                factor = 1.0 + batch.temperature_coefficient[mask] * (
                    absolute_temperature -
                    batch.conductivity_reference_temperature.reshape(-1)[gid])
                factor = torch.maximum(torch.minimum(factor, bounds[1]), bounds[0])
                k_y = batch.conductivity_reference_y[mask]*factor
                top_conductance = batch.top_face_area[mask] / (
                    batch.top_half_distance[mask]/k_y.clamp_min(1e-12) +
                    1.0/batch.boundary_h.reshape(-1)[gid].clamp_min(1e-12))
                rejected = (top_conductance*pred[mask]).sum()
            else:
                rejected = (batch.top_conductance[mask]*pred[mask]).sum()
            supplied = batch.source_power[mask].sum()
            balance = float((rejected-supplied).abs()/supplied)
            local_chip = batch.chip_id[mask]
            local_chip_errors = []
            for cid in torch.unique(local_chip[local_chip >= 0]).tolist():
                cmask = local_chip == cid
                if cmask.any():
                    w = volume[cmask]
                    local_chip_errors.append(float(abs(
                        (w*pred[mask][cmask]).sum()/w.sum() -
                        (w*true[mask][cmask]).sum()/w.sum())))
            case = {
                'layout_id': int(batch.layout_id[gid]),
                'power_profile_id': int(batch.power_profile_id[gid]),
                'physical_case_id': int(batch.physical_case_id[gid]),
                'volume_weighted_mae_K': mae,
                'volume_weighted_rmse_K': rmse,
                'absolute_peak_error_K': peak,
                'mean_chip_mean_error_K': float(np.mean(local_chip_errors)),
                'energy_balance_rel_error': balance,
            }
            if operator_residual is not None:
                denominator = torch.linalg.vector_norm(batch.source_power[mask]).clamp_min(1e-20)
                case['nonlinear_operator_residual_rel_l2'] = float(
                    torch.linalg.vector_norm(operator_residual[mask])/denominator)
            if getattr(batch, 'contact_pair_index', None) is not None:
                pair = batch.contact_pair_index.long()
                if pair.shape[1] > 0:
                    face_mask = batch.batch[pair[0]] == gid
                    local_pair = pair[:, face_mask]
                    flow = (batch.contact_conductance[face_mask] *
                            (pred[local_pair[0]]-pred[local_pair[1]]))
                    predicted_jump = (flow/batch.contact_face_area[face_mask].clamp_min(1e-20) *
                                      batch.contact_resistance[face_mask])
                    target_jump = batch.contact_temperature_jump_target[face_mask]
                    area = batch.contact_face_area[face_mask]
                    numerator = torch.sqrt((area*(predicted_jump-target_jump).square()).sum())
                    denominator = torch.sqrt((area*target_jump.square()).sum()).clamp_min(1e-12)
                    case['contact_jump_relative_l2'] = float(numerator/denominator)
            cases.append(case)
    maes = np.asarray([x['volume_weighted_mae_K'] for x in cases])
    rmses = np.asarray([x['volume_weighted_rmse_K'] for x in cases])
    peaks = np.asarray([x['absolute_peak_error_K'] for x in cases])
    balances = np.asarray([x['energy_balance_rel_error'] for x in cases])
    chip_errors = np.asarray([x['mean_chip_mean_error_K'] for x in cases])
    layout_mae = {}
    for layout_id in sorted({x['layout_id'] for x in cases}):
        layout_mae[str(layout_id)] = float(np.mean([
            x['volume_weighted_mae_K'] for x in cases if x['layout_id'] == layout_id]))
    summary = {
        'n_cases': len(cases),
        'volume_weighted_mae_K': float(maes.mean()),
        'volume_weighted_mae_p95_K': float(np.quantile(maes, 0.95)),
        'volume_weighted_rmse_K': float(rmses.mean()),
        'mean_absolute_peak_error_K': float(peaks.mean()),
        'mean_chip_mean_error_K': float(chip_errors.mean()),
        'mean_energy_balance_rel_error': float(balances.mean()),
        'layout_mean_mae_K': float(np.mean(list(layout_mae.values()))),
        'layout_worst_mae_K': float(max(layout_mae.values())),
        'mae_by_layout_K': layout_mae,
        'inference_ms_per_graph': 1000*elapsed/len(cases),
        'cases': cases,
    }
    if cases and all('contact_jump_relative_l2' in case for case in cases):
        summary['mean_contact_jump_relative_l2'] = float(np.mean([
            case['contact_jump_relative_l2'] for case in cases]))
    if cases and all('nonlinear_operator_residual_rel_l2' in case for case in cases):
        summary['mean_nonlinear_operator_residual_rel_l2'] = float(np.mean([
            case['nonlinear_operator_residual_rel_l2'] for case in cases]))
    return summary


def train_one(name, train_set, val_set, eval_sets, metadata, cfg, out_dir, device, seed, epochs):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tcfg = cfg['training']
    mean, std = metadata['dT_train_mean'], metadata['dT_train_std']
    model = build_atplace_model(name, metadata, cfg).to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=int(tcfg['batch_size']), shuffle=True,
                              generator=generator)
    val_loader = DataLoader(val_set, batch_size=int(tcfg['batch_size']), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg['learning_rate']),
                                  weight_decay=float(tcfg['weight_decay']))
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)
    best, best_state, wait, history = float('inf'), None, 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            target = (batch.y-mean_t)/std_t
            loss = graph_loss(
                pred, target, batch, float(tcfg['peak_loss_weight']),
                float(tcfg.get('contact_jump_weight', 0.0)),
                float(tcfg.get('operator_residual_weight', 0.0)),
                float(tcfg.get('operator_relative_residual_weight', 0.0)), mean_t, std_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())*batch.num_graphs
            count += batch.num_graphs
        val = validation_loss(
            model, val_loader, device, mean_t, std_t, float(tcfg['peak_loss_weight']),
            float(tcfg.get('contact_jump_weight', 0.0)),
            float(tcfg.get('operator_residual_weight', 0.0)),
            float(tcfg.get('operator_relative_residual_weight', 0.0)))
        history.append({'epoch': epoch, 'train_loss': total/count, 'val_loss': val})
        if val < best-1e-7:
            best, wait = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(name, epoch, f'train={total/count:.5f}', f'val={val:.5f}')
        if wait >= int(tcfg['patience']):
            break
    model.load_state_dict(best_state)
    train_time = time.perf_counter()-started
    metrics = {
        suite: evaluate(model, DataLoader(dataset, batch_size=1, shuffle=False),
                        device, mean_t, std_t)
        for suite, dataset in eval_sets.items()
    }
    run_info = {
        'best_val_loss': best, 'epochs_run': len(history), 'train_time_s': train_time,
        'parameter_count': sum(p.numel() for p in model.parameters()), 'seed': seed,
    }
    model_dir = out_dir/name
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model': best_state, 'metadata': metadata, 'config': cfg,
                'model_name': name, 'seed': seed},
               model_dir/'checkpoint.pt')
    (model_dir/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    (model_dir/'metrics.json').write_text(
        json.dumps({'run': run_info, 'suites': metrics}, indent=2), encoding='utf-8')
    return {'run': run_info, 'suites': metrics}


def aggregate_seed_results(seed_results):
    metric_names = [
        'volume_weighted_mae_K', 'volume_weighted_mae_p95_K',
        'volume_weighted_rmse_K', 'mean_absolute_peak_error_K',
        'mean_chip_mean_error_K', 'mean_energy_balance_rel_error',
        'layout_mean_mae_K', 'layout_worst_mae_K', 'inference_ms_per_graph']
    output = {}
    models = sorted(next(iter(seed_results.values())))
    suites = sorted(next(iter(seed_results.values()))[models[0]]['suites'])
    if all(
            'mean_contact_jump_relative_l2' in
            seed_results[seed][model]['suites'][suite]
            for seed in seed_results for model in models for suite in suites):
        metric_names.append('mean_contact_jump_relative_l2')
    if all(
            'mean_nonlinear_operator_residual_rel_l2' in
            seed_results[seed][model]['suites'][suite]
            for seed in seed_results for model in models for suite in suites):
        metric_names.append('mean_nonlinear_operator_residual_rel_l2')
    for model in models:
        output[model] = {}
        for suite in suites:
            output[model][suite] = {}
            for metric in metric_names:
                values = np.asarray([
                    seed_results[str(seed)][model]['suites'][suite][metric]
                    for seed in sorted(map(int, seed_results))], dtype=float)
                output[model][suite][metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    'values': values.tolist(),
                }
    return output


def main():
    parser = argparse.ArgumentParser(description='ATPlace Case1 grouped-layout ML comparison')
    parser.add_argument('--data', type=Path, default=Path('data/atplace_case1_ml'))
    parser.add_argument('--config', type=Path, default=Path('configs/atplace_case1_ml.yaml'))
    parser.add_argument('--out', type=Path, default=Path('outputs/atplace_case1_ml'))
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--models', nargs='+', default=['graphsage', 'meshgraphnet', 'mgn_transolver'])
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--resume-existing', action='store_true',
                        help='Reuse a seed/model run when checkpoint.pt and metrics.json both exist')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    metadata = json.loads((args.data/'metadata.json').read_text(encoding='utf-8'))
    sets = {s: torch.load(args.data/f'{s}.pt', weights_only=False) for s in ('train', 'val', 'test')}
    train_ids = {int(d.layout_id) for d in sets['train']}
    val_ids = {int(d.layout_id) for d in sets['val']}
    test_ids = {int(d.layout_id) for d in sets['test']}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError('layout leakage across splits')
    evaluation_level = metadata.get('evaluation_mesh_level')
    eval_sets = {'layout_only_medium': sets['test']}
    if evaluation_level:
        eval_sets['mesh_only_fine'] = torch.load(
            args.data/f'eval_mesh_only_{evaluation_level}.pt', weights_only=False)
        eval_sets['layout_mesh_fine'] = torch.load(
            args.data/f'eval_layout_mesh_{evaluation_level}.pt', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs or int(cfg['training']['epochs'])
    seeds = args.seeds or [int(x) for x in cfg.get('seeds', [cfg['seed']])]
    args.out.mkdir(parents=True, exist_ok=True)
    seed_results = {}
    for seed in seeds:
        seed_dir = args.out/f'seed_{seed}'
        seed_results[str(seed)] = {}
        for model in args.models:
            model_dir = seed_dir/model
            metrics_path, checkpoint_path = model_dir/'metrics.json', model_dir/'checkpoint.pt'
            if args.resume_existing and metrics_path.exists() and checkpoint_path.exists():
                seed_results[str(seed)][model] = json.loads(
                    metrics_path.read_text(encoding='utf-8'))
                print(seed, model, 'resumed existing run')
            else:
                seed_results[str(seed)][model] = train_one(
                    model, sets['train'], sets['val'], eval_sets, metadata, cfg,
                    seed_dir, device, seed, epochs)
            print(seed, model, json.dumps(seed_results[str(seed)][model]['run']))
    summary = {
        'device': str(device), 'seeds': seeds,
        'split_layout_ids': {'train': sorted(train_ids), 'val': sorted(val_ids),
                             'test': sorted(test_ids)},
        'suite_definitions': metadata.get('evaluation_protocol', {}),
        'seed_results': seed_results,
        'aggregate': aggregate_seed_results(seed_results),
    }
    (args.out/'benchmark.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
