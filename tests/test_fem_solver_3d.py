"""Small-scale smoke and physical-consistency tests for 3D tetrahedral FEM."""
import numpy as np

from fem_solver_3d import (
    CaseParams3D,
    GeometryConfig3D,
    build_mesh_3d,
    material_id_of_points_3d,
    q_field_3d,
    solve_case_3d,
)


def make_case_3d(**overrides):
    values = dict(
        k_die=140.0,
        k_tim=3.0,
        k_cu=385.0,
        k_sub=30.0,
        q_base=2.0e5,
        q_hot=1.5e7,
        hotspot_center_x_frac=0.5,
        hotspot_center_z_frac=0.5,
        hotspot_width_x_frac=0.4,
        hotspot_width_z_frac=0.4,
        t_ambient=25.0,
        h_top=3000.0,
    )
    values.update(overrides)
    return CaseParams3D(**values)


def test_tetrahedral_fem_smoke_case_is_valid():
    # 5 x 13 x 5 is deliberately small and has every layer interface on a y node plane.
    geom = GeometryConfig3D(nx=5, ny=13, nz=5)
    mesh, basis = build_mesh_3d(geom)
    result = solve_case_3d(mesh, basis, geom, make_case_3d())

    assert mesh.p.shape[0] == 3
    assert mesh.t.shape[0] == 4
    assert result.T.shape == (mesh.p.shape[1],)
    assert np.all(np.isfinite(result.T))
    assert np.allclose(result.T[result.is_bottom], 25.0)
    assert result.T.max() > 25.0
    assert result.material_id[int(np.argmax(result.T))] == 3
    assert np.all(result.h_node[~result.is_top] == 0.0)
    assert np.all(result.h_node[result.is_top] == 3000.0)


def test_hotspot_is_localized_in_xz_plane_and_die_only():
    geom = GeometryConfig3D()
    case = make_case_3d(hotspot_center_x_frac=0.5, hotspot_center_z_frac=0.5, hotspot_width_x_frac=0.2, hotspot_width_z_frac=0.2)
    x = np.array([0.010, 0.002, 0.010, 0.010])
    y = np.array([0.011, 0.011, 0.011, 0.009])
    z = np.array([0.008, 0.008, 0.001, 0.008])
    q = q_field_3d(x, y, z, geom, case)
    assert q.tolist() == [case.q_hot, case.q_base, case.q_base, 0.0]


def test_3d_material_layers_follow_y_coordinate():
    geom = GeometryConfig3D()
    y = np.array([geom.y_sub / 2, (geom.y_sub + geom.y_cu) / 2, (geom.y_cu + geom.y_tim) / 2, (geom.y_tim + geom.height) / 2])
    x = np.zeros_like(y)
    z = np.zeros_like(y)
    assert material_id_of_points_3d(x, y, z, geom).tolist() == [0, 1, 2, 3]
