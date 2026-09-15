"""Feature and topology tests for the 3D PyG graph conversion."""
import numpy as np

from graph_dataset_3d import (
    NormConsts3D,
    build_edge_features_3d,
    build_edge_index_from_tets,
    build_node_features_3d,
    cooling_path_resistance_fraction_3d,
    lumped_node_volumes,
)


def test_lumped_volume_tracks_irregular_tetrahedral_node_density():
    points = np.array([[0., 1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 2.]])
    tets = np.array([[0, 0], [1, 1], [2, 2], [3, 4]])
    weights = lumped_node_volumes(points, tets)
    assert weights.shape == (5,)
    assert weights.mean() == np.float32(1.)
    assert weights[4] == 2 * weights[3]


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
    assert x.shape == (4, 26)
    assert x.dtype == np.float32
    assert np.allclose(x[:, 3:7].sum(axis=1), 1.0)
    assert np.allclose(x[:, 12:], x[0, 12:])
    assert np.all((x[:, 24] >= 0.0) & (x[:, 24] <= 1.0))
    assert np.all((x[:, 25] > 0.0) & (x[:, 25] < 1.0))

    edge_index = build_edge_index_from_tets(np.array([[0], [1], [2], [3]]))
    edge_attr = build_edge_features_3d(points, edge_index, material_id, fake_case_3d(), nc)
    assert edge_attr.shape == (12, 5)
    assert np.all(edge_attr[:, 3] > 0)
    assert np.all(edge_attr[:, 4] > 0)


def test_shared_tetrahedron_edges_match_reference():
    rng = np.random.default_rng(42)
    tets = np.column_stack([rng.choice(40, 4, replace=False) for _ in range(100)])
    expected = sorted({(int(a), int(b)) for tet in tets.T for a in tet for b in tet if a != b})
    actual = build_edge_index_from_tets(tets)
    np.testing.assert_array_equal(actual, np.asarray(expected).T)
    np.testing.assert_array_equal(actual, build_edge_index_from_tets(tets[::-1, ::-1]))
    assert actual.dtype == np.int64


def test_empty_tetrahedron_edges():
    edges = build_edge_index_from_tets(np.empty((4, 0), dtype=np.int64))
    assert edges.shape == (2, 0)
    assert edges.dtype == np.int64


def test_invalid_tetrahedron_indices():
    import pytest
    for tets in (np.zeros((3, 1), dtype=int), np.zeros((4, 1)), -np.ones((4, 1), dtype=int)):
        with pytest.raises(ValueError):
            build_edge_index_from_tets(tets)


def test_cooling_path_feature_increases_when_top_convection_worsens():
    nc = NormConsts3D(fake_raw_meta_3d())
    well_cooled = fake_case_3d()
    poorly_cooled = fake_case_3d()
    well_cooled["h_top"] = 12000.0
    poorly_cooled["h_top"] = 150.0

    assert cooling_path_resistance_fraction_3d(poorly_cooled, nc) > cooling_path_resistance_fraction_3d(well_cooled, nc)
