"""Plot the ATPlace operator-alignment and nonlinear warm-start comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'Supervised': ROOT / 'outputs/atplace_case1_contact_material_ml/warm_start_layout_mesh_fine.json',
    'Scaled residual': ROOT / 'outputs/atplace_case1_contact_material_operator_aligned/warm_start_layout_mesh_fine.json',
    'True residual': ROOT / 'outputs/atplace_case1_contact_material_operator_aligned_v2/warm_start_layout_mesh_fine.json',
}
OUT = ROOT / 'docs/assets/atplace_contact_material_ml/operator_alignment_comparison.png'


def main():
    payloads = {name: json.loads(path.read_text(encoding='utf-8'))['aggregate']
                for name, path in RUNS.items()}
    labels = list(payloads)
    colors = ['#777777', '#dd8452', '#4c72b0']
    mae = [payloads[k]['network']['volume_weighted_mae_K']['mean'] for k in labels]
    residual = [payloads[k]['network_initial_residual_rel_l2']['mean'] for k in labels]
    iterations = [payloads[k]['warm_linear_iterations']['mean'] for k in labels]
    corrected_micro_K = [
        1e6*payloads[k]['warm_corrected']['volume_weighted_mae_K']['mean'] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), constrained_layout=True)
    x = np.arange(len(labels))
    panels = [
        (mae, 'Raw network MAE [K]', False, '{:.2f}'),
        (residual, r'Initial residual $||r||_2/||q||_2$', True, '{:.1f}'),
        (iterations, 'Warm-start CG iterations', False, '{:.0f}'),
    ]
    for ax, (values, title, log_scale, fmt) in zip(axes, panels):
        bars = ax.bar(x, values, color=colors, width=0.68)
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.25)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], padding=3, fontsize=9)
    fig.suptitle(
        'Unseen layout + fine mesh (12 cases)\n'
        f'Nonlinear correction MAE: {np.mean(corrected_micro_K):.2f} micro-K for all variants',
        fontsize=12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190)
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
