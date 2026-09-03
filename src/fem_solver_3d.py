"""Three-dimensional steady-state chip-package thermal FEM solver.

The package is a layered rectangular volume with y as the stack direction:
substrate -> Cu -> TIM -> die.  A P1 tetrahedral mesh is used throughout.
The die heat source is a rectangular hot spot on the x-z plane and extends
through the die thickness in y.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

try:  # The normal project path uses scikit-fem; the fallback keeps the smoke test runnable offline.
    from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, LinearForm, MeshTet, condense, solve
    from skfem.helpers import dot, grad

    HAS_SKFEM = True
except ImportError:  # pragma: no cover - exercised only in minimal/offline environments
    HAS_SKFEM = False


N_MATERIALS = 4
MATERIAL_NAMES = ["substrate", "cu", "tim", "die"]


@dataclass
class GeometryConfig3D:
    """Layered package dimensions in metres and tensor-grid resolutions."""

    width: float = 0.020
    depth: float = 0.016
    substrate_thickness: float = 0.006
    cu_thickness: float = 0.003
    tim_thickness: float = 0.001
    die_thickness: float = 0.002
    nx: int = 13
    ny: int = 13
    nz: int = 11

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
    def from_dict(cls, d: dict) -> "GeometryConfig3D":
        return cls(
            width=d["width"],
            depth=d["depth"],
            substrate_thickness=d["substrate_thickness"],
            cu_thickness=d["cu_thickness"],
            tim_thickness=d["tim_thickness"],
            die_thickness=d["die_thickness"],
            nx=d["nx"],
            ny=d["ny"],
            nz=d["nz"],
        )


@dataclass
class CaseParams3D:
    k_die: float
    k_tim: float
    k_cu: float
    k_sub: float
    q_base: float
    q_hot: float
    hotspot_center_x_frac: float
    hotspot_center_z_frac: float
    hotspot_width_x_frac: float
    hotspot_width_z_frac: float
    t_ambient: float
    h_top: float
    regime: str = "id"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FemResult3D:
    points: np.ndarray       # (3, N), coordinates in m
    tets: np.ndarray         # (4, M), tetrahedron connectivity
    T: np.ndarray            # (N,), nodal temperature in Celsius
    material_id: np.ndarray  # (N,), material IDs 0..3
    q_node: np.ndarray       # (N,), volumetric heat source in W/m^3
    is_top: np.ndarray       # (N,), top convection boundary marker
    is_bottom: np.ndarray    # (N,), bottom Dirichlet boundary marker
    h_node: np.ndarray       # (N,), nodal convection coefficient
    solve_time_s: float
    case: CaseParams3D


@dataclass
class _TetraMeshFallback:
    """Minimal mesh interface used only when scikit-fem is unavailable."""

    p: np.ndarray
    t: np.ndarray


def build_mesh_3d(geom: GeometryConfig3D):
    """Build a regular tetrahedral mesh shared by all 3D cases.

    ``MeshTet.init_tensor`` decomposes each structured hexahedral cell into
    tetrahedra, so every node remains aligned with the package layer planes
    when the requested y grid contains their coordinates.
    """
    if min(geom.nx, geom.ny, geom.nz) < 2:
        raise ValueError("nx, ny, and nz must each be at least 2")
    x = np.linspace(0.0, geom.width, geom.nx)
    y = np.linspace(0.0, geom.height, geom.ny)
    z = np.linspace(0.0, geom.depth, geom.nz)
    if HAS_SKFEM:
        mesh = MeshTet.init_tensor(x, y, z)
        return mesh, Basis(mesh, ElementTetP1())
    return _build_tensor_tet_mesh_fallback(x, y, z), None


def _build_tensor_tet_mesh_fallback(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> _TetraMeshFallback:
    """Split each structured hexahedron into six tetrahedra around its 0--7 diagonal."""
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
    ny, nz = len(y), len(z)

    def node(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    tets = []
    local_tets = ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7))
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            for k in range(len(z) - 1):
                corners = (
                    node(i, j, k), node(i + 1, j, k), node(i, j + 1, k), node(i + 1, j + 1, k),
                    node(i, j, k + 1), node(i + 1, j, k + 1), node(i, j + 1, k + 1), node(i + 1, j + 1, k + 1),
                )
                tets.extend(tuple(corners[index] for index in tet) for tet in local_tets)
    return _TetraMeshFallback(p=points, t=np.asarray(tets, dtype=np.int64).T)


def material_id_of_points_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, geom: GeometryConfig3D) -> np.ndarray:
    """Return the material ID at each point; x and z are accepted for a uniform layered API."""
    del x, z
    return np.where(
        y <= geom.y_sub + 1e-12,
        0,
        np.where(y <= geom.y_cu + 1e-12, 1, np.where(y <= geom.y_tim + 1e-12, 2, 3)),
    )


def k_field_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, geom: GeometryConfig3D, case: CaseParams3D) -> np.ndarray:
    del x, z
    return np.where(
        y <= geom.y_sub,
        case.k_sub,
        np.where(y <= geom.y_cu, case.k_cu, np.where(y <= geom.y_tim, case.k_tim, case.k_die)),
    )


def hotspot_bounds_3d(geom: GeometryConfig3D, case: CaseParams3D) -> tuple[float, float, float, float]:
    """Return x/z bounds of the planar rectangular die hot spot."""
    x_lo = (case.hotspot_center_x_frac - case.hotspot_width_x_frac / 2.0) * geom.width
    x_hi = (case.hotspot_center_x_frac + case.hotspot_width_x_frac / 2.0) * geom.width
    z_lo = (case.hotspot_center_z_frac - case.hotspot_width_z_frac / 2.0) * geom.depth
    z_hi = (case.hotspot_center_z_frac + case.hotspot_width_z_frac / 2.0) * geom.depth
    return x_lo, x_hi, z_lo, z_hi


def q_field_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, geom: GeometryConfig3D, case: CaseParams3D) -> np.ndarray:
    """Volumetric heat source, localized on an x-z rectangle inside the die."""
    x_lo, x_hi, z_lo, z_hi = hotspot_bounds_3d(geom, case)
    in_die = y > (geom.y_tim + 1e-12)
    in_hotspot = (x >= x_lo) & (x <= x_hi) & (z >= z_lo) & (z <= z_hi)
    return np.where(in_die, np.where(in_hotspot, case.q_hot, case.q_base), 0.0)


def solve_case_3d(mesh: MeshTet, basis: Basis, geom: GeometryConfig3D, case: CaseParams3D) -> FemResult3D:
    """Solve ``-div(k grad(T)) = q`` with bottom Dirichlet and top Robin BCs."""

    if not HAS_SKFEM:
        return _solve_case_dense_fallback(mesh, geom, case)

    @BilinearForm
    def stiffness(u, v, w):
        return k_field_3d(w.x[0], w.x[1], w.x[2], geom, case) * dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        return q_field_3d(w.x[0], w.x[1], w.x[2], geom, case) * v

    @BilinearForm
    def robin(u, v, w):
        return case.h_top * u * v

    @LinearForm
    def robin_rhs(v, w):
        return case.h_top * case.t_ambient * v

    top_facets = mesh.facets_satisfying(lambda p: np.isclose(p[1], geom.height))
    bottom_facets = mesh.facets_satisfying(lambda p: np.isclose(p[1], 0.0))
    fbasis_top = FacetBasis(mesh, basis.elem, facets=top_facets)

    t0 = time.perf_counter()
    A_total = stiffness.assemble(basis) + robin.assemble(fbasis_top)
    b_total = load.assemble(basis) + robin_rhs.assemble(fbasis_top)

    bottom_dofs = basis.get_dofs(bottom_facets)
    x0 = basis.zeros()
    x0[bottom_dofs] = case.t_ambient
    A_out, b_out, x_out, free_dofs = condense(A_total, b_total, x=x0, D=bottom_dofs)
    T = x_out.copy()
    T[free_dofs] = solve(A_out, b_out)
    solve_time_s = time.perf_counter() - t0

    px, py, pz = mesh.p
    material_id = material_id_of_points_3d(px, py, pz, geom)
    q_node = q_field_3d(px, py, pz, geom, case)
    top_dofs = np.asarray(basis.get_dofs(top_facets))
    bottom_dofs_arr = np.asarray(bottom_dofs)
    is_top = np.zeros(mesh.p.shape[1], dtype=bool)
    is_bottom = np.zeros(mesh.p.shape[1], dtype=bool)
    is_top[top_dofs] = True
    is_bottom[bottom_dofs_arr] = True

    return FemResult3D(
        points=mesh.p.copy(),
        tets=mesh.t.copy(),
        T=T,
        material_id=material_id,
        q_node=q_node,
        is_top=is_top,
        is_bottom=is_bottom,
        h_node=np.where(is_top, case.h_top, 0.0),
        solve_time_s=solve_time_s,
        case=case,
    )


def _solve_case_dense_fallback(mesh: _TetraMeshFallback, geom: GeometryConfig3D, case: CaseParams3D) -> FemResult3D:
    """Tiny, dependency-free P1 tetrahedral FEM implementation for offline smoke tests.

    It assembles a dense matrix and is intentionally restricted to modest meshes.
    Regular dataset production should use scikit-fem's sparse implementation.
    """
    n_nodes = mesh.p.shape[1]
    if n_nodes > 600:
        raise RuntimeError("scikit-fem is required for 3D meshes with more than 600 nodes")

    t0 = time.perf_counter()
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    b = np.zeros(n_nodes, dtype=np.float64)
    for tet in mesh.t.T:
        coords = mesh.p[:, tet].T
        affine = np.column_stack([np.ones(4), coords])
        inv_affine = np.linalg.inv(affine)
        gradients = inv_affine[1:, :].T
        volume = abs(np.linalg.det(coords[1:] - coords[0])) / 6.0
        centroid = coords.mean(axis=0)
        conductivity = float(k_field_3d(centroid[0], centroid[1], centroid[2], geom, case))
        source = float(q_field_3d(centroid[0], centroid[1], centroid[2], geom, case))
        A[np.ix_(tet, tet)] += conductivity * volume * (gradients @ gradients.T)
        b[tet] += source * volume / 4.0

    top_faces = set()
    for tet in mesh.t.T:
        for face in ((tet[0], tet[1], tet[2]), (tet[0], tet[1], tet[3]), (tet[0], tet[2], tet[3]), (tet[1], tet[2], tet[3])):
            if np.all(np.isclose(mesh.p[1, list(face)], geom.height)):
                top_faces.add(tuple(sorted(map(int, face))))
    for face in top_faces:
        nodes = np.asarray(face)
        coords = mesh.p[:, nodes].T
        area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
        A[np.ix_(nodes, nodes)] += case.h_top * area / 12.0 * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
        b[nodes] += case.h_top * case.t_ambient * area / 3.0

    is_bottom = np.isclose(mesh.p[1], 0.0)
    is_top = np.isclose(mesh.p[1], geom.height)
    free = np.flatnonzero(~is_bottom)
    prescribed = np.flatnonzero(is_bottom)
    T = np.full(n_nodes, case.t_ambient, dtype=np.float64)
    T[free] = np.linalg.solve(A[np.ix_(free, free)], b[free] - A[np.ix_(free, prescribed)] @ T[prescribed])
    px, py, pz = mesh.p
    return FemResult3D(
        points=mesh.p.copy(),
        tets=mesh.t.copy(),
        T=T,
        material_id=material_id_of_points_3d(px, py, pz, geom),
        q_node=q_field_3d(px, py, pz, geom, case),
        is_top=is_top,
        is_bottom=is_bottom,
        h_node=np.where(is_top, case.h_top, 0.0),
        solve_time_s=time.perf_counter() - t0,
        case=case,
    )


def _demo_case() -> CaseParams3D:
    return CaseParams3D(
        k_die=140.0,
        k_tim=3.0,
        k_cu=385.0,
        k_sub=30.0,
        q_base=2.0e5,
        q_hot=1.5e7,
        hotspot_center_x_frac=0.5,
        hotspot_center_z_frac=0.5,
        hotspot_width_x_frac=0.35,
        hotspot_width_z_frac=0.35,
        t_ambient=25.0,
        h_top=3000.0,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run a small 3D tetrahedral thermal FEM case.")
    parser.add_argument("--nx", type=int, default=9)
    parser.add_argument("--ny", type=int, default=13)
    parser.add_argument("--nz", type=int, default=7)
    args = parser.parse_args()

    geom = GeometryConfig3D(nx=args.nx, ny=args.ny, nz=args.nz)
    mesh, basis = build_mesh_3d(geom)
    result = solve_case_3d(mesh, basis, geom, _demo_case())
    print(f"nodes={mesh.p.shape[1]}, tetrahedra={mesh.t.shape[1]}")
    print(f"solve_time_ms={result.solve_time_s * 1000:.3f}")
    print(f"temperature_C={result.T.min():.3f}..{result.T.max():.3f}")
    print(f"max_rise_K={result.T.max() - result.case.t_ambient:.3f}")


if __name__ == "__main__":
    main()
