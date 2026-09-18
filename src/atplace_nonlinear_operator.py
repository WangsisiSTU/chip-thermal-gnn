"""Temperature-consistent ATPlace finite-volume operator and warm-start solve.

The graph stores only geometry and reference material data.  Conductances are
reassembled from the candidate temperature, so no final-solution conductivity
is leaked into training or inference.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg
from torch_geometric.utils import scatter


OPERATOR_FIELDS = (
    'edge_axis', 'edge_half_distance_src', 'edge_half_distance_dst',
    'edge_face_area', 'edge_contact_resistance',
    'conductivity_reference_diag', 'temperature_coefficient',
    'top_face_area', 'top_half_distance', 'boundary_h',
    'conductivity_reference_temperature', 'conductivity_factor_bounds',
    't_ambient', 'source_power',
)


def require_operator_contract(data):
    missing = [name for name in OPERATOR_FIELDS if getattr(data, name, None) is None]
    if missing:
        raise ValueError(f'nonlinear operator contract is missing fields: {missing}')


def _node_graph(data, n_nodes: int, device):
    batch = getattr(data, 'batch', None)
    if batch is None:
        return torch.zeros(n_nodes, dtype=torch.long, device=device), 1
    return batch, int(data.num_graphs)


def dynamic_conductances(theta: torch.Tensor, data):
    """Reassemble directed internal-face and node-aligned Robin conductances."""
    require_operator_contract(data)
    graph, n_graphs = _node_graph(data, theta.numel(), theta.device)
    ambient = data.t_ambient.reshape(-1)
    reference_temperature = data.conductivity_reference_temperature.reshape(-1)
    bounds = data.conductivity_factor_bounds.reshape(n_graphs, 2)
    absolute_temperature = theta + ambient[graph]
    factor = 1.0 + data.temperature_coefficient * (
        absolute_temperature-reference_temperature[graph])
    factor = torch.maximum(torch.minimum(factor, bounds[graph, 1]), bounds[graph, 0])
    conductivity = data.conductivity_reference_diag*factor[:, None]

    src, dst = data.edge_index.long()
    axis = data.edge_axis.long()
    k_src = conductivity[src, axis]
    k_dst = conductivity[dst, axis]
    resistance_area = (
        data.edge_half_distance_src/k_src.clamp_min(1e-12) +
        data.edge_contact_resistance +
        data.edge_half_distance_dst/k_dst.clamp_min(1e-12))
    edge_g = data.edge_face_area/resistance_area.clamp_min(1e-20)

    k_y = conductivity[:, 1]
    top_g = data.top_face_area / (
        data.top_half_distance/k_y.clamp_min(1e-12) +
        1.0/data.boundary_h.reshape(-1)[graph].clamp_min(1e-12))
    return edge_g, top_g, conductivity


def nonlinear_residual(theta: torch.Tensor, data):
    """Return the physical residual, diagonal, and dynamic conductances."""
    edge_g, top_g, conductivity = dynamic_conductances(theta, data)
    src, dst = data.edge_index.long()
    conductive = scatter(
        edge_g*(theta[src]-theta[dst]), src, dim=0,
        dim_size=theta.numel(), reduce='sum')
    diagonal = scatter(edge_g, src, dim=0, dim_size=theta.numel(), reduce='sum') + top_g
    residual = conductive + top_g*theta-data.source_power
    return residual, diagonal, edge_g, top_g, conductivity


def nonlinear_residual_loss(prediction, data, mean, std):
    """Volume-weighted, dimensionless diagonal-scaled operator residual."""
    theta = prediction*std+mean
    residual, diagonal, _, _, _ = nonlinear_residual(theta, data)
    equivalent_temperature_error = residual/diagonal.clamp_min(1e-12)
    graph, n_graphs = _node_graph(data, theta.numel(), theta.device)
    mass = scatter(data.node_volume, graph, dim=0, dim_size=n_graphs, reduce='sum')
    mse = scatter(
        data.node_volume*equivalent_temperature_error.square(), graph,
        dim=0, dim_size=n_graphs, reduce='sum')/mass.clamp_min(1e-20)
    scale = torch.as_tensor(std, device=theta.device, dtype=theta.dtype).square().clamp_min(1e-12)
    return (mse/scale).mean()


def nonlinear_relative_residual_loss(prediction, data, mean, std):
    """Log-compressed graphwise ||r||²/||source||² for solver warm starts."""
    theta = prediction*std+mean
    residual = nonlinear_residual(theta, data)[0]
    graph, n_graphs = _node_graph(data, theta.numel(), theta.device)
    numerator = scatter(
        residual.square(), graph, dim=0, dim_size=n_graphs, reduce='sum')
    denominator = scatter(
        data.source_power.square(), graph, dim=0, dim_size=n_graphs,
        reduce='sum').clamp_min(1e-20)
    return torch.log1p(numerator/denominator).mean()


@dataclass
class WarmStartResult:
    temperature_rise_K: np.ndarray
    nonlinear_iterations: int
    linear_iterations: int
    initial_residual_rel_l2: float
    final_residual_rel_l2: float
    energy_balance_rel_error: float
    max_update_K: float
    solve_time_s: float


def _as_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _numpy_operator_data(data):
    require_operator_contract(data)
    if getattr(data, 'batch', None) is not None and int(data.num_graphs) != 1:
        raise ValueError('warm-start solve accepts exactly one graph')
    return {
        'edge_index': _as_numpy(data.edge_index).astype(np.int64),
        'edge_axis': _as_numpy(data.edge_axis).astype(np.int64),
        'distance_src': _as_numpy(data.edge_half_distance_src).astype(float),
        'distance_dst': _as_numpy(data.edge_half_distance_dst).astype(float),
        'edge_area': _as_numpy(data.edge_face_area).astype(float),
        'edge_rtc': _as_numpy(data.edge_contact_resistance).astype(float),
        'k_reference': _as_numpy(data.conductivity_reference_diag).astype(float),
        'alpha': _as_numpy(data.temperature_coefficient).astype(float),
        'top_area': _as_numpy(data.top_face_area).astype(float),
        'top_half': _as_numpy(data.top_half_distance).astype(float),
        'boundary_h': float(_as_numpy(data.boundary_h).reshape(-1)[0]),
        'reference_temperature': float(
            _as_numpy(data.conductivity_reference_temperature).reshape(-1)[0]),
        'bounds': _as_numpy(data.conductivity_factor_bounds).reshape(-1, 2)[0].astype(float),
        'ambient': float(_as_numpy(data.t_ambient).reshape(-1)[0]),
        'source': _as_numpy(data.source_power).astype(float),
    }


def _assemble_numpy(theta, contract):
    factor = np.clip(
        1.0+contract['alpha']*(theta+contract['ambient']-contract['reference_temperature']),
        contract['bounds'][0], contract['bounds'][1])
    conductivity = contract['k_reference']*factor[:, None]
    src, dst = contract['edge_index']
    axis = contract['edge_axis']
    edge_g = contract['edge_area'] / (
        contract['distance_src']/conductivity[src, axis] + contract['edge_rtc'] +
        contract['distance_dst']/conductivity[dst, axis])
    top_g = contract['top_area'] / (
        contract['top_half']/conductivity[:, 1] + 1.0/contract['boundary_h'])
    diagonal = np.bincount(src, weights=edge_g, minlength=theta.size)+top_g
    index = np.arange(theta.size)
    operator = coo_matrix(
        (np.concatenate((-edge_g, diagonal)),
         (np.concatenate((src, index)), np.concatenate((dst, index)))),
        shape=(theta.size, theta.size)).tocsr()
    return operator, diagonal, edge_g, top_g


def _relative_residual(theta, contract):
    operator, _, _, top_g = _assemble_numpy(theta, contract)
    residual = operator@theta-contract['source']
    denominator = max(float(np.linalg.norm(contract['source'])), 1e-30)
    energy = abs(float(top_g@theta)-float(contract['source'].sum())) / max(
        float(contract['source'].sum()), 1e-30)
    return float(np.linalg.norm(residual)/denominator), energy


def nonlinear_warm_start_solve(
        data, initial_temperature_rise, *, nonlinear_tolerance_K: float = 1e-7,
        nonlinear_max_iterations: int = 80, relaxation: float = 0.70,
        linear_rtol: float = 1e-10, linear_max_iterations: int | None = None):
    """Converge the exact graph FVM operator from a network or cold initial guess."""
    if nonlinear_tolerance_K <= 0 or nonlinear_max_iterations < 1:
        raise ValueError('invalid nonlinear solve tolerances')
    if not 0 < relaxation <= 1:
        raise ValueError('relaxation must be in (0, 1]')
    contract = _numpy_operator_data(data)
    theta = np.asarray(initial_temperature_rise, dtype=float).reshape(-1).copy()
    if theta.size != contract['source'].size or not np.all(np.isfinite(theta)):
        raise ValueError('initial temperature rise has the wrong shape or non-finite values')
    initial_residual, _ = _relative_residual(theta, contract)
    total_linear_iterations = 0
    max_update = float('inf')
    started = time.perf_counter()

    def linear_solve(operator, diagonal, x0):
        nonlocal total_linear_iterations
        count = 0

        def callback(_):
            nonlocal count
            count += 1

        preconditioner = LinearOperator(
            operator.shape, matvec=lambda value: value/np.maximum(diagonal, 1e-30))
        solution, info = cg(
            operator, contract['source'], x0=x0, rtol=linear_rtol, atol=0.0,
            maxiter=linear_max_iterations, M=preconditioner, callback=callback)
        total_linear_iterations += count
        if info != 0:
            raise RuntimeError(f'preconditioned CG failed to converge; info={info}')
        return solution

    for nonlinear_iterations in range(1, nonlinear_max_iterations+1):
        operator, diagonal, _, _ = _assemble_numpy(theta, contract)
        candidate = linear_solve(operator, diagonal, theta)
        max_update = float(np.max(np.abs(candidate-theta)))
        if max_update <= nonlinear_tolerance_K:
            theta = candidate
            break
        theta = (1.0-relaxation)*theta+relaxation*candidate
    else:
        raise RuntimeError(
            f'nonlinear warm-start solve failed in {nonlinear_max_iterations} iterations; '
            f'last update={max_update:.3e} K')

    # Match the reference path: assemble once more at the converged material
    # state and solve the resulting linear system to the requested tolerance.
    operator, diagonal, _, _ = _assemble_numpy(theta, contract)
    theta = linear_solve(operator, diagonal, theta)
    final_residual, energy = _relative_residual(theta, contract)
    return WarmStartResult(
        temperature_rise_K=theta,
        nonlinear_iterations=nonlinear_iterations,
        linear_iterations=total_linear_iterations,
        initial_residual_rel_l2=initial_residual,
        final_residual_rel_l2=final_residual,
        energy_balance_rel_error=energy,
        max_update_K=max_update,
        solve_time_s=time.perf_counter()-started,
    )
