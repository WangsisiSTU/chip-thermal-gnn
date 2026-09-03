"""
二维芯片封装稳态导热有限元求解器（基于 scikit-fem）。

几何：矩形截面，自下而上为 基板(Substrate) -> 铜散热层(Cu) -> TIM -> Die(硅)。
方程： -div(k * grad(T)) = q，  k、q 分材料/分区域取值。

边界条件（简化设置，详见 README 中的物理假设说明）：
    - 顶部（Die 上表面）：对流边界  -k dT/dn = h_top * (T - T_ambient)
    - 底部（基板下表面）：固定温度 Dirichlet  T = T_ambient
    - 左右两侧：绝热（自然 Neumann 零通量，无需额外处理）

材料类别编码： 0 = 基板, 1 = 铜, 2 = TIM, 3 = Die(硅)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from skfem import Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm, MeshTri, condense, solve
from skfem.helpers import dot, grad

N_MATERIALS = 4
MATERIAL_NAMES = ["substrate", "cu", "tim", "die"]


@dataclass
class GeometryConfig:
    width: float = 0.020
    substrate_thickness: float = 0.006
    cu_thickness: float = 0.003
    tim_thickness: float = 0.001
    die_thickness: float = 0.002
    nx: int = 61
    ny: int = 37

    @property
    def y_sub(self) -> float:
        return self.substrate_thickness

    @property
    def y_cu(self) -> float:
        return self.substrate_thickness + self.cu_thickness

    @property
    def y_tim(self) -> float:
        return self.y_cu + self.tim_thickness

    @property
    def height(self) -> float:
        return self.y_tim + self.die_thickness

    @classmethod
    def from_dict(cls, d: dict) -> "GeometryConfig":
        return cls(
            width=d["width"],
            substrate_thickness=d["substrate_thickness"],
            cu_thickness=d["cu_thickness"],
            tim_thickness=d["tim_thickness"],
            die_thickness=d["die_thickness"],
            nx=d["nx"],
            ny=d["ny"],
        )


@dataclass
class CaseParams:
    k_die: float
    k_tim: float
    k_cu: float
    k_sub: float
    q_base: float
    q_hot: float
    hotspot_center_frac: float
    hotspot_width_frac: float
    t_ambient: float
    h_top: float
    regime: str = "id"  # "id" (in-distribution) 或 "ood_xxx"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def build_mesh(geom: GeometryConfig):
    """构建规则三角形网格（tensor-product 网格三角化），所有样本共用同一网格。"""
    x = np.linspace(0.0, geom.width, geom.nx)
    y = np.linspace(0.0, geom.height, geom.ny)
    mesh = MeshTri.init_tensor(x, y)
    basis = Basis(mesh, ElementTriP1())
    return mesh, basis


def material_id_of_points(x: np.ndarray, y: np.ndarray, geom: GeometryConfig) -> np.ndarray:
    """给定坐标数组，返回材料类别编码（0=基板,1=铜,2=TIM,3=Die）。"""
    return np.where(
        y <= geom.y_sub + 1e-12,
        0,
        np.where(y <= geom.y_cu + 1e-12, 1, np.where(y <= geom.y_tim + 1e-12, 2, 3)),
    )


def k_field(x: np.ndarray, y: np.ndarray, geom: GeometryConfig, case: CaseParams) -> np.ndarray:
    return np.where(
        y <= geom.y_sub,
        case.k_sub,
        np.where(y <= geom.y_cu, case.k_cu, np.where(y <= geom.y_tim, case.k_tim, case.k_die)),
    )


def hotspot_bounds(geom: GeometryConfig, case: CaseParams):
    hot_lo = (case.hotspot_center_frac - case.hotspot_width_frac / 2.0) * geom.width
    hot_hi = (case.hotspot_center_frac + case.hotspot_width_frac / 2.0) * geom.width
    return hot_lo, hot_hi


def q_field(x: np.ndarray, y: np.ndarray, geom: GeometryConfig, case: CaseParams) -> np.ndarray:
    hot_lo, hot_hi = hotspot_bounds(geom, case)
    in_die = y > (geom.y_tim + 1e-12)
    in_hot = (x >= hot_lo) & (x <= hot_hi)
    return np.where(in_die, np.where(in_hot, case.q_hot, case.q_base), 0.0)


@dataclass
class FemResult:
    points: np.ndarray          # (2, N) 节点坐标 [m]
    tris: np.ndarray            # (3, T) 三角形单元连接关系（节点索引）
    T: np.ndarray                # (N,) 节点温度 [摄氏度]
    material_id: np.ndarray      # (N,) 节点材料类别
    q_node: np.ndarray           # (N,) 节点处热源体密度 [W/m^3]
    is_top: np.ndarray            # (N,) 是否位于顶部对流边界
    is_bottom: np.ndarray         # (N,) 是否位于底部恒温边界
    h_node: np.ndarray            # (N,) 节点对流换热系数（非顶部边界为 0）
    solve_time_s: float
    case: CaseParams


def solve_case(mesh: MeshTri, basis: Basis, geom: GeometryConfig, case: CaseParams) -> FemResult:
    """对给定网格与工况参数求解稳态导热方程，返回节点温度场及相关特征。"""
    H = geom.height

    @BilinearForm
    def stiffness(u, v, w):
        k = k_field(w.x[0], w.x[1], geom, case)
        return k * dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        q = q_field(w.x[0], w.x[1], geom, case)
        return q * v

    @BilinearForm
    def robin(u, v, w):
        return case.h_top * u * v

    @LinearForm
    def robin_rhs(v, w):
        return case.h_top * case.t_ambient * v

    top_facets = mesh.facets_satisfying(lambda p: np.isclose(p[1], H))
    bottom_facets = mesh.facets_satisfying(lambda p: np.isclose(p[1], 0.0))
    fbasis_top = FacetBasis(mesh, basis.elem, facets=top_facets)

    t0 = time.perf_counter()
    A = stiffness.assemble(basis)
    b = load.assemble(basis)
    A_r = robin.assemble(fbasis_top)
    b_r = robin_rhs.assemble(fbasis_top)
    A_total = A + A_r
    b_total = b + b_r

    bottom_dofs = basis.get_dofs(bottom_facets)
    x0 = basis.zeros()
    x0[bottom_dofs] = case.t_ambient

    A_out, b_out, x_out, I = condense(A_total, b_total, x=x0, D=bottom_dofs)
    T_I = solve(A_out, b_out)
    T = x_out.copy()
    T[I] = T_I
    solve_time_s = time.perf_counter() - t0

    px, py = mesh.p[0], mesh.p[1]
    material_id = material_id_of_points(px, py, geom)
    q_node = q_field(px, py, geom, case)

    top_dofs = np.asarray(basis.get_dofs(top_facets))
    bottom_dofs_arr = np.asarray(bottom_dofs)
    is_top = np.zeros(mesh.p.shape[1], dtype=bool)
    is_bottom = np.zeros(mesh.p.shape[1], dtype=bool)
    is_top[top_dofs] = True
    is_bottom[bottom_dofs_arr] = True
    h_node = np.where(is_top, case.h_top, 0.0)

    return FemResult(
        points=mesh.p.copy(),
        tris=mesh.t.copy(),
        T=T,
        material_id=material_id,
        q_node=q_node,
        is_top=is_top,
        is_bottom=is_bottom,
        h_node=h_node,
        solve_time_s=solve_time_s,
        case=case,
    )


def _demo_case(geom: GeometryConfig) -> CaseParams:
    return CaseParams(
        k_die=140.0,
        k_tim=3.0,
        k_cu=385.0,
        k_sub=30.0,
        q_base=2.0e5,
        q_hot=1.5e7,
        hotspot_center_frac=0.5,
        hotspot_width_frac=0.3,
        t_ambient=25.0,
        h_top=3000.0,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="运行一个演示工况的二维稳态导热 FEM 求解")
    parser.add_argument("--nx", type=int, default=61)
    parser.add_argument("--ny", type=int, default=37)
    args = parser.parse_args()

    geom = GeometryConfig(nx=args.nx, ny=args.ny)
    mesh, basis = build_mesh(geom)
    case = _demo_case(geom)
    result = solve_case(mesh, basis, geom, case)

    print(f"节点数: {mesh.p.shape[1]}, 三角单元数: {mesh.t.shape[1]}")
    print(f"求解耗时: {result.solve_time_s * 1000:.3f} ms")
    print(f"温度范围: {result.T.min():.3f} ~ {result.T.max():.3f} 摄氏度")
    print(f"最大温升 dT: {(result.T.max() - case.t_ambient):.3f} K")


if __name__ == "__main__":
    main()
