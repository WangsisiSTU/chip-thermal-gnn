"""Feature and topology tests for the 3D PyG graph conversion."""
import numpy as np

from graph_dataset_3d import (
    NormConsts3D,
    build_edge_features_3d,
    build_edge_index_from_tets,
    build_node_features_3d,
)


def fake_raw_meta_3d():
    return {
        "dimension": 3,
        "materials": {
            "k_die": {"low": 130.0, "high": 155.0},
            "k_tim": {"low": 1.0, "high": 8.0},
            "k_cu": {"low": 360.0, "high": 400.0},
            "k_sub": {"low": 15.0, "high": 60.0},
        },
        "heat_source": {"q_hot": {"low": 5e6, "high": 3e7}},
        "boundary": {"t_ambient": {"low": 20.0, "high": 45.0}, "h_top": {"low": 500.0, "high": 15000.0}},
        "ood": {"q_hot": {"low": 3.5e7, "high": 6e7}, "k_tim": {"low": 0.2, "high": 0.9}, "h_top": {"low": 100.0, "high": 400.0}},
        "geometry": {"width": 0.02, "depth": 0.016, "substrate_thickness": 0.006, "cu_thickness": 0.003, "tim_thickness": 0.001, "die_thickness": 0.002},
    }


def fake_case_3d():
    return {
        "k_die": 140.0, "k_tim": 3.0, "k_cu": 385.0, "k_sub": 30.0,
        "q_base": 2e5, "q_hot": 1.5e7,
        "hotspot_center_x_frac": 0.5, "hotspot_center_z_frac": 0.5,
        "hotspot_width_x_frac": 0.3, "hotspot_width_z_frac": 0.3,
        "t_ambient": 25.0, "h_top": 3000.0,
    }


def test_edge_index_contains_all_tetrahedron_edges_bidirectionally():
    tets = np.array([[0], [1], [2], [3]])
    edge_index = build_edge_index_from_tets(tets)
    assert edge_index.shape == (2, 12)  # 6 tetrahedron edges in both directions
    edges = set(map(tuple, edge_index.T.tolist()))
    assert all((b, a) in edges for a, b in edges)
    assert all(a != b for a, b in edges)


def test_3d_node_and_edge_feature_shapes():
    nc = NormConsts3D(fake_raw_meta_3d())
    points = np.array([[0.0, 0.02, 0.0, 0.02], [0.0, 0.0, 0.012, 0.012], [0.0, 0.0, 0.016, 0.016]])
    material_id = np.array([0, 1, 2, 3])
    q_node = np.array([0.0, 0.0, 0.0, 1.5e7])
    h_node = np.array([0.0, 0.0, 0.0, 3000.0])
    is_top = np.array([False, False, False, True])
    is_bottom = np.array([True, False, False, False])
    x = build_node_features_3d(points, material_id, q_node, h_node, is_top, is_bottom, fake_case_3d(), nc)
    assert x.shape == (4, 24)
    assert x.dtype == np.float32
    assert np.allclose(x[:, 3:7].sum(axis=1), 1.0)
    assert np.allclose(x[:, 12:], x[0, 12:])

    edge_index = build_edge_index_from_tets(np.array([[0], [1], [2], [3]]))
    edge_attr = build_edge_features_3d(points, edge_index, material_id, fake_case_3d(), nc)
    assert edge_attr.shape == (12, 5)
    assert np.all(edge_attr[:, 3] > 0)
    assert np.all(edge_attr[:, 4] > 0)
