"""Manufactured-solution verification for the 3D P1 tetrahedral FEM path.

It solves -Delta(T) = 3*pi^2*T on the unit cube with T=0 on every face.
The exact solution is T=sin(pi*x)sin(pi*y)sin(pi*z), allowing a mesh-refinement
study independent of the chip-package material and boundary assumptions.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from skfem import Basis, BilinearForm, ElementTetP1, LinearForm, MeshTet, condense, solve
from skfem.helpers import dot, grad


def exact_temperature(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])


def solve_mms_case(n_per_axis: int) -> dict:
    if n_per_axis < 3:
        raise ValueError("n_per_axis must be at least 3")
    axis = np.linspace(0.0, 1.0, n_per_axis)
    mesh = MeshTet.init_tensor(axis, axis, axis)
    basis = Basis(mesh, ElementTetP1())

    @BilinearForm
    def stiffness(u, v, w):
        return dot(grad(u), grad(v))

    @LinearForm
    def source(v, w):
        return 3.0 * np.pi ** 2 * exact_temperature(w.x) * v

    A = stiffness.assemble(basis)
    b = source.assemble(basis)
    boundary_dofs = basis.get_dofs(mesh.boundary_facets())
    x0 = basis.zeros()
    A_free, b_free, x_out, free_dofs = condense(A, b, x=x0, D=boundary_dofs)
    u_h = x_out.copy()
    u_h[free_dofs] = solve(A_free, b_free)
    nodal_error = u_h - exact_temperature(mesh.p)
    return {
        "n_per_axis": n_per_axis,
        "h": 1.0 / (n_per_axis - 1),
        "n_nodes": int(mesh.p.shape[1]),
        "n_tets": int(mesh.t.shape[1]),
        "nodal_rmse": float(np.sqrt(np.mean(nodal_error ** 2))),
        "nodal_max_abs_error": float(np.max(np.abs(nodal_error))),
    }


def run_mms_study(levels: tuple[int, ...] = (5, 9, 13)) -> dict:
    if len(levels) < 2:
        raise ValueError("at least two mesh levels are required")
    rows = [solve_mms_case(n) for n in levels]
    orders = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        orders.append(
            float(
                np.log(coarse["nodal_rmse"] / fine["nodal_rmse"])
                / np.log(coarse["h"] / fine["h"])
            )
        )
    return {"exact_solution": "sin(pi*x) sin(pi*y) sin(pi*z)", "rows": rows, "observed_orders": orders}


def main():
    parser = argparse.ArgumentParser(description="Run the 3D tetrahedral manufactured-solution convergence study.")
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 9, 13])
    parser.add_argument("--out", default="outputs/3d/metrics/mms_3d_convergence.json")
    args = parser.parse_args()

    report = run_mms_study(tuple(args.levels))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for row in report["rows"]:
        print(
            f"n={row['n_per_axis']:2d}, nodes={row['n_nodes']:5d}, tets={row['n_tets']:6d}, "
            f"nodal_RMSE={row['nodal_rmse']:.4e}"
        )
    print("observed_orders=" + ", ".join(f"{order:.3f}" for order in report["observed_orders"]))


if __name__ == "__main__":
    main()
