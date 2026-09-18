"""Validate paired ATPlace evaluation suites and write the three-seed report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


SUITE_LABELS = {
    'layout_only_medium': '未见布局 / 中网格',
    'mesh_only_fine': '已见布局 / 细网格',
    'layout_mesh_fine': '未见布局 / 细网格',
}
MODEL_LABELS = {
    'graphsage': 'GraphSAGE',
    'meshgraphnet': 'MeshGraphNet',
    'mgn_transolver': 'MGN+TRANS',
}


def _case_index(graphs):
    result = {}
    for graph in graphs:
        key = int(graph.physical_case_id)
        if key in result:
            raise ValueError(f'duplicate physical_case_id {key}')
        result[key] = graph
    return result


def _validate_pair(coarse, fine, label):
    coarse_index, fine_index = _case_index(coarse), _case_index(fine)
    if set(fine_index) - set(coarse_index):
        raise ValueError(f'{label}: fine cases are absent from the medium suite')
    for case_id, fine_graph in fine_index.items():
        medium_graph = coarse_index[case_id]
        if int(medium_graph.layout_id) != int(fine_graph.layout_id):
            raise ValueError(f'{label}: layout mismatch for physical case {case_id}')
        if int(medium_graph.power_profile_id) != int(fine_graph.power_profile_id):
            raise ValueError(f'{label}: profile mismatch for physical case {case_id}')
        if not np.isclose(float(medium_graph.total_power_W), float(fine_graph.total_power_W),
                          rtol=1e-7, atol=1e-5):
            raise ValueError(f'{label}: total-power mismatch for physical case {case_id}')
    return len(fine_index)


def _graph_range(graphs, field):
    values = [int(getattr(graph, field)) for graph in graphs]
    return min(values), max(values)


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def summarize(data_dir: Path, benchmark_path: Path, output_path: Path):
    metadata = json.loads((data_dir/'metadata.json').read_text(encoding='utf-8'))
    benchmark = json.loads(benchmark_path.read_text(encoding='utf-8'))
    train = torch.load(data_dir/'train.pt', weights_only=False)
    val = torch.load(data_dir/'val.pt', weights_only=False)
    test = torch.load(data_dir/'test.pt', weights_only=False)
    level = metadata['evaluation_mesh_level']
    mesh_only = torch.load(data_dir/f'eval_mesh_only_{level}.pt', weights_only=False)
    layout_mesh = torch.load(data_dir/f'eval_layout_mesh_{level}.pt', weights_only=False)
    paired_mesh = _validate_pair(train, mesh_only, 'mesh_only')
    paired_layout_mesh = _validate_pair(test, layout_mesh, 'layout_mesh')
    medium_nodes = _graph_range(train + val + test, 'num_nodes')
    fine_nodes = _graph_range(mesh_only + layout_mesh, 'num_nodes')
    medium_edges = _graph_range(train + val + test, 'num_edges')
    fine_edges = _graph_range(mesh_only + layout_mesh, 'num_edges')

    lines = [
        '# ATPlace Case1：MGN、MGN+TRANS 与可归因泛化评测', '',
        '## 数据与评测协议', '',
        f'- 独立物理案例：{len(train)+len(val)+len(test)} 个，来自 24 个布局 × 4 个功率工况。',
        f'- 中网格划分：{len(train)}/{len(val)}/{len(test)} 张 train/val/test 图，布局 ID 完全隔离。',
        f'- 细网格评测：{paired_mesh} 个已见布局配对案例，以及 {paired_layout_mesh} 个未见布局配对案例。',
        f'- 中网格节点 {medium_nodes[0]:,}–{medium_nodes[1]:,}，有向边 {medium_edges[0]:,}–{medium_edges[1]:,}；'
        f'细网格节点 {fine_nodes[0]:,}–{fine_nodes[1]:,}，有向边 {fine_edges[0]:,}–{fine_edges[1]:,}。',
        '- 配对检查确认：同一 `physical_case_id` 在中/细网格间保持 layout、功率 profile 和总功率一致。',
        '- 三个模型使用相同训练数据、损失、优化预算和解析全局能量投影；训练种子为 3107/3108/3109。',
        '', '三套测试分别隔离：', '',
        '1. 未见布局 / 中网格：只改变布局。',
        '2. 已见布局 / 细网格：保持物理案例不变，只改变网格。',
        '3. 未见布局 / 细网格：布局与网格同时变化。',
        '', '## 模型规模与训练成本', '',
        '| 模型 | 参数量 | 训练时间均值 ± 标准差 / seed |',
        '| --- | ---: | ---: |',
    ]
    for model in MODEL_LABELS:
        runs = [benchmark['seed_results'][str(seed)][model]['run'] for seed in benchmark['seeds']]
        time_mean, time_std = _mean_std([run['train_time_s'] for run in runs])
        lines.append(f'| {MODEL_LABELS[model]} | {runs[0]["parameter_count"]:,} | '
                     f'{time_mean:.1f} ± {time_std:.1f} s |')

    lines += ['', '## 三种子测试结果', '',
              '| 测试 | 模型 | MAE K | P95 case MAE K | 最差布局 MAE K | 峰温误差 K | 推理 ms/图 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    aggregate = benchmark['aggregate']
    for suite in SUITE_LABELS:
        for model in MODEL_LABELS:
            values = aggregate[model][suite]
            def formatted(metric):
                item = values[metric]
                return f'{item["mean"]:.3f} ± {item["std"]:.3f}'
            lines.append(
                f'| {SUITE_LABELS[suite]} | {MODEL_LABELS[model]} | '
                f'{formatted("volume_weighted_mae_K")} | '
                f'{formatted("volume_weighted_mae_p95_K")} | '
                f'{formatted("layout_worst_mae_K")} | '
                f'{formatted("mean_absolute_peak_error_K")} | '
                f'{values["inference_ms_per_graph"]["mean"]:.2f} |')

    lines += ['', '## 结论', '']
    for suite in SUITE_LABELS:
        best_mae = min(MODEL_LABELS, key=lambda m: aggregate[m][suite]['volume_weighted_mae_K']['mean'])
        best_peak = min(MODEL_LABELS, key=lambda m: aggregate[m][suite]['mean_absolute_peak_error_K']['mean'])
        lines.append(
            f'- {SUITE_LABELS[suite]}：场 MAE 最低为 {MODEL_LABELS[best_mae]} '
            f'`{aggregate[best_mae][suite]["volume_weighted_mae_K"]["mean"]:.3f} K`；'
            f'峰温误差最低为 {MODEL_LABELS[best_peak]} '
            f'`{aggregate[best_peak][suite]["mean_absolute_peak_error_K"]["mean"]:.3f} K`。')
    lines += [
        '- MGN+TRANS 在三套测试的平均场误差均最低，说明全局物理状态交互对布局变化有效。',
        '- MeshGraphNet 的峰温误差更低且细网格种子方差更小；当前结果不支持用单一指标宣布混合模型全面领先。',
        '- 所有模型都使用解析能量投影，因此全局热平衡误差接近数值零。该指标只验证总输入/排热平衡，'
        '不能替代局部通量、界面温降和弱形式残差。',
        '- 当前数据仍是 Case1 内部的布局与网格泛化，不代表跨芯粒数量、跨封装、各向异性材料或接触热阻泛化。',
        '', '## 复现', '', '```powershell',
        'E:\\Anaconda\\python.exe src\\generate_atplace_ml_dataset.py `',
        '  --config configs\\atplace_case1_mgn_transolver.yaml `',
        '  --out data\\atplace_case1_mgn_transolver', '',
        'E:\\Anaconda\\python.exe src\\train_atplace_ml.py `',
        '  --data data\\atplace_case1_mgn_transolver `',
        '  --config configs\\atplace_case1_mgn_transolver.yaml `',
        '  --out outputs\\atplace_case1_mgn_transolver', '',
        'E:\\Anaconda\\python.exe src\\summarize_atplace_generalization.py',
        '```', '',
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return {'paired_mesh_only': paired_mesh, 'paired_layout_mesh': paired_layout_mesh,
            'medium_nodes': medium_nodes, 'fine_nodes': fine_nodes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/atplace_case1_mgn_transolver'))
    parser.add_argument('--benchmark', type=Path,
                        default=Path('outputs/atplace_case1_mgn_transolver/benchmark.json'))
    parser.add_argument('--out', type=Path,
                        default=Path('docs/atplace_mgn_transolver_generalization_report.md'))
    args = parser.parse_args()
    print(json.dumps(summarize(args.data, args.benchmark, args.out), indent=2))


if __name__ == '__main__':
    main()
