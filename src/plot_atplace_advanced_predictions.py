"""Plot genuine model predictions for the advanced-material ATPlace benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_atplace_ml import build_atplace_model


MODELS = ('graphsage', 'meshgraphnet', 'mgn_transolver')
LABELS = {'graphsage': 'GraphSAGE', 'meshgraphnet': 'MeshGraphNet',
          'mgn_transolver': 'MGN+TRANS'}


def predict(checkpoint_path: Path, graph, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = build_atplace_model(
        checkpoint['model_name'], checkpoint['metadata'], checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    local_graph = graph.clone().to(device)
    with torch.no_grad():
        normalized = model(local_graph)
        theta = (normalized*float(checkpoint['metadata']['dT_train_std']) +
                 float(checkpoint['metadata']['dT_train_mean']))
    return theta.detach().cpu().numpy()


def weighted_mae(pred, true, volume):
    return float(np.sum(volume*np.abs(pred-true))/np.sum(volume))


def contact_jump(graph, theta):
    pair = graph.contact_pair_index.numpy().astype(int)
    flow = graph.contact_conductance.numpy()*(theta[pair[0]]-theta[pair[1]])
    return (flow/np.maximum(graph.contact_face_area.numpy(), 1e-20) *
            graph.contact_resistance.numpy())


def plot(data_dir: Path, model_dir: Path, seed: int, case_index: int, out_dir: Path):
    metadata = json.loads((data_dir/'metadata.json').read_text(encoding='utf-8'))
    graphs = torch.load(data_dir/'eval_layout_mesh_fine.pt', weights_only=False)
    graph = graphs[case_index]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = {
        name: predict(model_dir/f'seed_{seed}'/name/'checkpoint.pt', graph, device)
        for name in MODELS
    }
    true = graph.y.numpy()
    volume = graph.node_volume.numpy()
    pos = graph.pos.numpy()*1e3
    layer_column = metadata['node_feature_names'].index('layer_4')
    chip_layer = graph.x[:, layer_column].numpy() > 0.5
    layer_y = np.unique(pos[chip_layer, 1])
    selected_y = layer_y[np.argmin(np.abs(layer_y-layer_y.mean()))]
    mask = chip_layer & np.isclose(pos[:, 1], selected_y)
    fields = [('FVM reference', true)] + [(LABELS[name], predictions[name]) for name in MODELS]
    vmin = min(float(values[mask].min()) for _, values in fields)
    vmax = max(float(values[mask].max()) for _, values in fields)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.5), constrained_layout=True)
    field_artist = None
    error_limit = max(float(np.max(np.abs(pred[mask]-true[mask])))
                      for pred in predictions.values())
    for column, (label, values) in enumerate(fields):
        field_artist = axes[0, column].tricontourf(
            pos[mask, 0], pos[mask, 2], values[mask],
            levels=np.linspace(vmin, vmax, 19), cmap='inferno')
        axes[0, column].set_aspect('equal')
        axes[0, column].set_title(label)
        if column == 0:
            axes[0, column].set_ylabel('z [mm]')
        axes[0, column].set_xlabel('x [mm]')
        if column == 0:
            axes[1, column].axis('off')
            axes[1, column].text(
                0.5, 0.56, 'Advanced-material benchmark\nactual network outputs',
                ha='center', va='center', fontsize=14, weight='bold')
            axes[1, column].text(
                0.5, 0.33,
                f'layout={int(graph.layout_id)}  profile={int(graph.power_profile_id)}\n'
                f'nodes={graph.num_nodes:,}  directed edges={graph.num_edges:,}',
                ha='center', va='center', fontsize=10)
            continue
        name = MODELS[column-1]
        error = values-true
        error_artist = axes[1, column].tricontourf(
            pos[mask, 0], pos[mask, 2], error[mask],
            levels=np.linspace(-error_limit, error_limit, 19), cmap='coolwarm')
        axes[1, column].set_aspect('equal')
        axes[1, column].set_xlabel('x [mm]')
        if column == 1:
            axes[1, column].set_ylabel('z [mm]')
        axes[1, column].set_title(
            f'error; MAE={weighted_mae(values, true, volume):.3f} K, '
            f'peak={abs(values.max()-true.max()):.3f} K')
    fig.colorbar(field_artist, ax=axes[0, :], shrink=0.86, label='temperature rise [K]')
    fig.colorbar(error_artist, ax=axes[1, 1:], shrink=0.86, label='prediction error [K]')
    out_dir.mkdir(parents=True, exist_ok=True)
    field_path = out_dir/'model_prediction_comparison.png'
    fig.savefig(field_path, dpi=220, bbox_inches='tight')
    plt.close(fig)

    target_jump = graph.contact_temperature_jump_target.numpy()
    area = graph.contact_face_area.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    low = min(float(target_jump.min()), *(float(contact_jump(graph, p).min())
                                          for p in predictions.values()))
    high = max(float(target_jump.max()), *(float(contact_jump(graph, p).max())
                                           for p in predictions.values()))
    for axis, name in zip(axes, MODELS):
        predicted = contact_jump(graph, predictions[name])
        relative = np.sqrt(np.sum(area*(predicted-target_jump)**2) /
                           max(np.sum(area*target_jump**2), 1e-20))
        take = np.linspace(0, len(target_jump)-1, min(700, len(target_jump))).astype(int)
        axis.scatter(target_jump[take], predicted[take], s=8, alpha=0.45,
                     color='#0072B2', rasterized=True)
        axis.plot([low, high], [low, high], '--', color='#D55E00', linewidth=1.2)
        axis.set_title(f'{LABELS[name]}  relative L2={relative:.3f}')
        axis.set_xlabel('reference contact jump [K]')
        axis.set_aspect('equal', adjustable='box')
    axes[0].set_ylabel('predicted contact jump [K]')
    contact_path = out_dir/'contact_jump_prediction.png'
    fig.savefig(contact_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return field_path, contact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=Path('data/atplace_case1_contact_material_ml'))
    parser.add_argument('--models', type=Path,
                        default=Path('outputs/atplace_case1_contact_material_ml'))
    parser.add_argument('--seed', type=int, default=4108)
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--out-dir', type=Path,
                        default=Path('docs/assets/atplace_contact_material_ml'))
    args = parser.parse_args()
    for path in plot(args.data, args.models, args.seed, args.case_index, args.out_dir):
        print(path)


if __name__ == '__main__':
    main()
