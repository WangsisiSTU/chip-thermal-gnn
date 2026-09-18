"""Plot ATPlace advanced-material ablations, convergence, and field slices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


COLORS = ['#607D8B', '#D55E00', '#0072B2', '#009E73', '#CC79A7']


def plot_benchmark(ablation_path: Path, convergence_path: Path, output: Path):
    ablation = json.loads(ablation_path.read_text(encoding='utf-8'))['variants']
    convergence = json.loads(convergence_path.read_text(encoding='utf-8'))['levels']
    keys = ['linear_isotropic_no_contact', 'contact_only', 'anisotropy_only',
            'temperature_only', 'combined']
    labels = ['Baseline', 'Contact', 'Anisotropy', 'k(T)', 'Combined']
    peak_delta = [ablation[key]['delta_peak_vs_linear_isotropic_no_contact_K'] for key in keys]
    chip_delta = [ablation[key]['delta_mean_chip_vs_linear_isotropic_no_contact_K'] for key in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    x = np.arange(len(keys))
    width = 0.38
    bars1 = axes[0].bar(x-width/2, peak_delta, width, color=COLORS,
                        edgecolor='white', label='Peak temperature')
    axes[0].bar(x+width/2, chip_delta, width, color=COLORS, alpha=0.48,
                edgecolor='white', hatch='//', label='Mean chip temperature')
    axes[0].axhline(0, color='#333333', linewidth=0.8)
    axes[0].set_xticks(x, labels, rotation=18, ha='right')
    axes[0].set_ylabel('Temperature change vs baseline [K]')
    axes[0].set_title('Controlled physics ablation (medium mesh)')
    axes[0].legend(frameon=False, fontsize=9)
    for bar, value in zip(bars1, peak_delta):
        if value > 0:
            axes[0].text(bar.get_x()+bar.get_width()/2, value+0.45, f'{value:.2f}',
                         ha='center', va='bottom', fontsize=8)

    cells = np.asarray([row['n_cells'] for row in convergence])
    peaks = np.asarray([row['temperature_max_K'] for row in convergence])
    errors = np.abs(peaks-peaks[-1])
    balances = np.asarray([row['energy_balance_rel_error'] for row in convergence])
    positive = errors[:-1]
    axes[1].loglog(cells[:-1], positive, 'o-', color='#0072B2', linewidth=2,
                   label='Peak error vs verification')
    axes[1].scatter(cells[-1], max(positive.min()/10, 1e-9), marker='*', s=120,
                    color='#0072B2', label='Verification reference')
    axes[1].set_xlabel('Control volumes')
    axes[1].set_ylabel('Absolute peak difference [K]', color='#0072B2')
    axes[1].tick_params(axis='y', labelcolor='#0072B2')
    axes[1].grid(True, which='both', alpha=0.22)
    balance_axis = axes[1].twinx()
    balance_axis.loglog(cells, balances, 's--', color='#D55E00', linewidth=1.5,
                        label='Energy balance error')
    balance_axis.set_ylabel('Relative energy error', color='#D55E00')
    balance_axis.tick_params(axis='y', labelcolor='#D55E00')
    axes[1].set_title('Five-level convergence and conservation')
    lines = axes[1].get_lines()+balance_axis.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], frameon=False,
                   loc='upper right', fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_fields(reference_path: Path, output: Path):
    raw = np.load(reference_path)
    pos = raw['pos_m']*1e3
    temperature = raw['temperature_K']
    layer = raw['layer_id']
    ambient = float(raw['ambient_K'])

    fig = plt.figure(figsize=(15.0, 4.7), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection='3d')
    stride = max(1, pos.shape[0]//6500)
    sample = np.arange(0, pos.shape[0], stride)
    points = ax3d.scatter(pos[sample, 0], pos[sample, 2], pos[sample, 1],
                          c=temperature[sample]-ambient, s=3, cmap='inferno',
                          alpha=0.72, rasterized=True)
    ax3d.set_xlabel('x [mm]')
    ax3d.set_ylabel('z [mm]')
    ax3d.set_zlabel('stack y [mm]')
    ax3d.set_title('3D temperature-rise cloud')
    fig.colorbar(points, ax=ax3d, shrink=0.72, pad=0.08, label='dT [K]')

    ax_slice = fig.add_subplot(1, 3, 2)
    chip_y = np.unique(pos[layer == 4, 1])
    selected_y = chip_y[np.argmin(np.abs(chip_y-chip_y.mean()))]
    mask = (layer == 4) & np.isclose(pos[:, 1], selected_y)
    contour = ax_slice.tricontourf(pos[mask, 0], pos[mask, 2],
                                   temperature[mask]-ambient, levels=18, cmap='inferno')
    ax_slice.set_aspect('equal')
    ax_slice.set_xlabel('x [mm]')
    ax_slice.set_ylabel('z [mm]')
    ax_slice.set_title(f'Chip-layer slice (y={selected_y:.3f} mm)')
    fig.colorbar(contour, ax=ax_slice, label='dT [K]')

    ax_contact = fig.add_subplot(1, 3, 3)
    pair = raw['contact_pair_index'].astype(int)
    centers = 0.5*(pos[pair[0]]+pos[pair[1]])
    jump = np.abs(raw['contact_temperature_jump_K'])
    nonzero = jump > 0
    scatter = ax_contact.scatter(
        centers[nonzero, 0], centers[nonzero, 2], c=jump[nonzero], s=12,
        cmap='viridis', norm=LogNorm(vmin=max(jump[nonzero].min(), 1e-3),
                                    vmax=jump[nonzero].max()), rasterized=True)
    ax_contact.set_aspect('equal')
    ax_contact.set_xlabel('x [mm]')
    ax_contact.set_ylabel('z [mm]')
    ax_contact.set_title('Contact-interface temperature jump')
    fig.colorbar(scatter, ax=ax_contact, label='|contact jump| [K]')

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ablation', type=Path,
                        default=Path('outputs/atplace_case1_contact_material/physics_ablation.json'))
    parser.add_argument('--convergence', type=Path,
                        default=Path('data/atplace_case1_contact_material/convergence_report.json'))
    parser.add_argument('--reference', type=Path,
                        default=Path('data/atplace_case1_contact_material/verification.npz'))
    parser.add_argument('--out-dir', type=Path,
                        default=Path('docs/assets/atplace_contact_material'))
    args = parser.parse_args()
    plot_benchmark(args.ablation, args.convergence, args.out_dir/'physics_benchmark.png')
    plot_fields(args.reference, args.out_dir/'advanced_temperature_field.png')
    print(args.out_dir/'physics_benchmark.png')
    print(args.out_dir/'advanced_temperature_field.png')


if __name__ == '__main__':
    main()
