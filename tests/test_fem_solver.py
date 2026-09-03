"""有限元求解器的基本有效性测试。"""
import numpy as np

from fem_solver import CaseParams, GeometryConfig, build_mesh, material_id_of_points, solve_case


def make_case(**overrides):
    base = dict(
        k_die=140.0, k_tim=3.0, k_cu=385.0, k_sub=30.0,
        q_base=2.0e5, q_hot=1.5e7,
        hotspot_center_frac=0.5, hotspot_width_frac=0.3,
        t_ambient=25.0, h_top=3000.0,
    )
    base.update(overrides)
    return CaseParams(**base)


def test_fem_solution_is_valid():
    geom = GeometryConfig(nx=31, ny=19)
    mesh, basis = build_mesh(geom)
    result = solve_case(mesh, basis, geom, make_case())

    n_nodes = mesh.p.shape[1]
    assert result.T.shape == (n_nodes,)
    assert np.all(np.isfinite(result.T)), "温度场包含 NaN/Inf"

    # 有正热源、底部恒温 T_amb：温度不应低于环境温度（数值容差内）
    assert result.T.min() >= 25.0 - 1e-6
    assert result.T.max() > 25.0, "存在热源时最高温度应高于环境温度"

    # 底部 Dirichlet 边界应严格等于环境温度
    assert np.allclose(result.T[result.is_bottom], 25.0)

    # 最高温度应出现在 Die 层（热源所在层）
    hottest_node = int(np.argmax(result.T))
    assert result.material_id[hottest_node] == 3


def test_fem_monotonic_with_power():
    """热源功率密度更大时，峰值温升应更大（线性问题的基本物理一致性）。"""
    geom = GeometryConfig(nx=31, ny=19)
    mesh, basis = build_mesh(geom)
    low = solve_case(mesh, basis, geom, make_case(q_hot=5.0e6))
    high = solve_case(mesh, basis, geom, make_case(q_hot=3.0e7))
    assert high.T.max() > low.T.max()


def test_material_id_layers():
    geom = GeometryConfig()
    x = np.zeros(4)
    y = np.array([
        geom.y_sub / 2,                      # 基板
        (geom.y_sub + geom.y_cu) / 2,        # 铜
        (geom.y_cu + geom.y_tim) / 2,        # TIM
        (geom.y_tim + geom.height) / 2,      # Die
    ])
    assert material_id_of_points(x, y, geom).tolist() == [0, 1, 2, 3]
