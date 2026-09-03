"""图数据构建的形状与结构测试。"""
import numpy as np

from graph_dataset import NormConsts, build_edge_features, build_edge_index, build_node_features


def fake_raw_meta():
    return {
        "materials": {
            "k_die": {"low": 130.0, "high": 155.0},
            "k_tim": {"low": 1.0, "high": 8.0},
            "k_cu": {"low": 360.0, "high": 400.0},
            "k_sub": {"low": 15.0, "high": 60.0},
        },
        "heat_source": {"q_hot": {"low": 5.0e6, "high": 3.0e7}},
        "boundary": {"t_ambient": {"low": 20.0, "high": 45.0}, "h_top": {"low": 500.0, "high": 15000.0}},
        "ood": {"q_hot": {"low": 3.5e7, "high": 6.0e7}, "k_tim": {"low": 0.2, "high": 0.9}, "h_top": {"low": 100.0, "high": 400.0}},
        "geometry": {
            "width": 0.02,
            "substrate_thickness": 0.006,
            "cu_thickness": 0.003,
            "tim_thickness": 0.001,
            "die_thickness": 0.002,
        },
    }


def fake_case():
    return {
        "k_die": 140.0, "k_tim": 3.0, "k_cu": 385.0, "k_sub": 30.0,
        "q_base": 2.0e5, "q_hot": 1.5e7,
        "hotspot_center_frac": 0.5, "hotspot_width_frac": 0.3,
        "t_ambient": 25.0, "h_top": 3000.0,
    }


def test_edge_index_from_triangles():
    # 两个三角形组成的正方形: 节点 0-1-2 与 1-3-2
    tris = np.array([[0, 1], [1, 3], [2, 2]])  # (3, T)
    edge_index = build_edge_index(tris)

    assert edge_index.shape[0] == 2
    # 无向图：每条无向边应包含两个方向
    edges = set(map(tuple, edge_index.T.tolist()))
    for i, j in list(edges):
        assert (j, i) in edges, f"缺少反向边 ({j},{i})"
    # 5 条无向边（4 条外边 + 1 条对角线）=> 10 条有向边
    assert edge_index.shape[1] == 10
    # 无自环
    assert np.all(edge_index[0] != edge_index[1])


def test_node_and_edge_feature_shapes():
    nc = NormConsts(fake_raw_meta())
    n = 6
    points = np.vstack([np.linspace(0, 0.02, n), np.linspace(0, 0.012, n)])  # (2, n)
    material_id = np.array([0, 0, 1, 2, 3, 3])
    q_node = np.array([0, 0, 0, 0, 1e7, 2e7], dtype=float)
    h_node = np.array([0, 0, 0, 0, 0, 3000.0])
    is_top = np.array([False, False, False, False, False, True])
    is_bottom = np.array([True, False, False, False, False, False])

    x = build_node_features(points, material_id, q_node, h_node, is_top, is_bottom, case=fake_case(), nc=nc)
    assert x.shape == (n, 21)
    assert x.dtype == np.float32
    # one-hot 应恰好一个 1
    assert np.allclose(x[:, 2:6].sum(axis=1), 1.0)
    # 全局特征应在所有节点上一致
    assert np.allclose(x[:, 11:], x[0, 11:])

    tris = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
    edge_index = build_edge_index(tris)
    edge_attr = build_edge_features(points, edge_index, material_id, fake_case(), nc)
    assert edge_attr.shape == (edge_index.shape[1], 4)
    assert np.all(edge_attr[:, 2] > 0), "边长应为正"
    assert np.all(edge_attr[:, 3] > 0), "等效导热系数应为正"


def test_processed_dataset_if_exists():
    """若已生成处理后的数据集，验证其形状一致性（未生成时跳过）。"""
    import os

    import pytest
    import torch

    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train.pt")
    if not os.path.exists(path):
        pytest.skip("处理后的数据集尚未生成")
    dataset = torch.load(path, weights_only=False)
    data = dataset[0]
    n, e = data.x.shape[0], data.edge_index.shape[1]
    assert data.x.shape[1] == 21
    assert data.edge_attr.shape == (e, 4)
    assert data.y.shape == (n,)
    assert data.pos.shape == (n, 2)
    assert int(data.edge_index.max()) < n
