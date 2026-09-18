"""Render the configured ATPlace Case1 six-layer computational geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import yaml


BACKGROUND = '#071826'
FOREGROUND = '#F2F5F7'
MUTED = '#AFC0CB'
LAYER_COLORS = {
    'substrate': '#6FA8DC',
    'c4': '#F6B26B',
    'interposer': '#B4A7D6',
    'ubump': '#FFD966',
    'chip': '#76A5AF',
    'tim': '#E691B8',
}
CHIP_COLORS = {'CPU': '#5CC8FF', 'GPU': '#FF9F43', 'HBM': '#65D6A6'}


def cuboid_faces(x0, x1, z0, z1, y0, y1):
    points = np.asarray([
        [x0, z0, y0], [x1, z0, y0], [x1, z1, y0], [x0, z1, y0],
        [x0, z0, y1], [x1, z0, y1], [x1, z1, y1], [x0, z1, y1],
    ])
    return [points[list(ids)] for ids in (
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))]


def add_box(ax, bounds, color, alpha=0.38, linewidth=1.1, edge=None):
    faces = cuboid_faces(*bounds)
    collection = Poly3DCollection(
        faces, facecolors=color, edgecolors=edge or color,
        linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(collection)


def chip_family(name):
    return next((key for key in CHIP_COLORS if name.startswith(key)), 'CPU')


def plot(config_path: Path, layout_path: Path, output_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    layout = json.loads(layout_path.read_text(encoding='utf-8'))
    width_mm, depth_mm = np.asarray(layout['interposer_size_um'], dtype=float)/1000
    vertical_scale = 30.0

    layer_bounds = []
    current_mm = 0.0
    for layer in config['layers']:
        thickness_mm = 1000*float(layer['thickness_m'])
        lower = current_mm*vertical_scale
        upper = (current_mm+thickness_mm)*vertical_scale
        layer_bounds.append((layer, lower, upper, thickness_mm))
        current_mm += thickness_mm

    plt.rcParams.update({
        'font.family': 'Microsoft YaHei',
        'text.color': FOREGROUND,
        'axes.labelcolor': MUTED,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
    })
    fig = plt.figure(figsize=(14.0, 7.2), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.42, 1.0), wspace=0.10)
    ax = fig.add_subplot(grid[0, 0], projection='3d')
    ax.set_facecolor(BACKGROUND)

    for layer, lower, upper, thickness_mm in layer_bounds:
        name = layer['name']
        alpha = 0.13 if name in ('chip', 'tim') else 0.25
        add_box(ax, (0, width_mm, 0, depth_mm, lower, upper),
                LAYER_COLORS[name], alpha=alpha)

    chip_layer = next(row for row in layer_bounds if row[0]['name'] == 'chip')
    chip_low, chip_high = chip_layer[1], chip_layer[2]
    for chip in layout['chiplets']:
        x0, z0 = np.asarray(chip['origin_um'], dtype=float)/1000
        sx, sz = np.asarray(chip['size_um'], dtype=float)/1000
        family = chip_family(chip['name'])
        add_box(ax, (x0, x0+sx, z0, z0+sz, chip_low, chip_high),
                CHIP_COLORS[family], alpha=0.72, linewidth=1.2, edge='#E9F3F8')

    total_display_height = layer_bounds[-1][2]
    ax.set_xlim(0, width_mm+2)
    ax.set_ylim(0, depth_mm)
    ax.set_zlim(0, total_display_height+1.2)
    ax.set_xlabel('x [mm]', labelpad=8)
    ax.set_ylabel('z [mm]', labelpad=8)
    ax.set_zticks([])
    ax.set_box_aspect((54, 42, 23))
    ax.view_init(elev=24, azim=-56)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(BACKGROUND)
        axis.pane.set_edgecolor('#315064')
        axis._axinfo['axisline']['color'] = MUTED
    label_names = {
        'tim': '⑥ TIM 20 μm', 'chip': '⑤ Chiplet + underfill 150 μm',
        'ubump': '④ μ-bump 10 μm',
        'interposer': '③ Si interposer + TSV 110 μm',
        'c4': '② C4 70 μm', 'substrate': '① Substrate 200 μm',
    }
    fig.text(0.06, 0.885, '六层封装计算模型（纵向 30×）', fontsize=16,
             weight='bold', color=FOREGROUND)
    label_positions = {
        'substrate': (0.06, 0.846), 'c4': (0.20, 0.846),
        'interposer': (0.30, 0.846), 'ubump': (0.06, 0.812),
        'chip': (0.20, 0.812), 'tim': (0.43, 0.812),
    }
    for name, position in label_positions.items():
        fig.text(*position, label_names[name], fontsize=9.3,
                 color=LAYER_COLORS[name], weight='bold')

    layout_ax = fig.add_subplot(grid[0, 1])
    layout_ax.set_facecolor(BACKGROUND)
    layout_ax.add_patch(plt.Rectangle(
        (0, 0), width_mm, depth_mm, facecolor='#102B3A',
        edgecolor='#8AA4B5', linewidth=1.5))
    for chip in layout['chiplets']:
        x0, z0 = np.asarray(chip['origin_um'], dtype=float)/1000
        sx, sz = np.asarray(chip['size_um'], dtype=float)/1000
        family = chip_family(chip['name'])
        color = CHIP_COLORS[family]
        layout_ax.add_patch(plt.Rectangle(
            (x0, z0), sx, sz, facecolor=color, edgecolor='#EAF4F8',
            linewidth=1.2, alpha=0.88))
        layout_ax.text(
            x0+sx/2, z0+sz/2,
            f"{chip['name']}\n{chip['power_W']:.0f} W",
            ha='center', va='center', fontsize=9.2, color=BACKGROUND,
            weight='bold')
    layout_ax.set_xlim(-1, width_mm+1)
    layout_ax.set_ylim(-1, depth_mm+1)
    layout_ax.set_aspect('equal')
    layout_ax.set_xlabel('x [mm]')
    layout_ax.set_ylabel('z [mm]')
    layout_ax.set_title('Chiplet 平面布局 · 780 W', loc='left', pad=14,
                        fontsize=18, weight='bold', color=FOREGROUND)
    layout_ax.spines[['top', 'right']].set_visible(False)
    layout_ax.spines[['left', 'bottom']].set_color('#607D8B')

    fig.suptitle('ATPlace Case1 · 42 × 42 mm 多芯粒六层封装',
                 x=0.06, y=0.98, ha='left', fontsize=23,
                 weight='bold', color=FOREGROUND)
    fig.text(0.06, 0.035,
             '几何与功耗来自当前项目配置；纵向放大 30 倍，仅用于展示 560 μm 总堆叠厚度。',
             fontsize=10.5, color=MUTED)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.16)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path('configs/atplace_case1_contact_material.yaml'))
    parser.add_argument('--layout', type=Path,
                        default=Path('data/atplace_layouts/Case1.json'))
    parser.add_argument('--out', type=Path,
                        default=Path('docs/assets/atplace_contact_material/atplace_six_layer_model.png'))
    args = parser.parse_args()
    plot(args.config, args.layout, args.out)
    print(args.out.resolve())


if __name__ == '__main__':
    main()
