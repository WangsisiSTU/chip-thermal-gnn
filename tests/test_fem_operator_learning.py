import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from fem_solver_3d import assemble_thermal_operator_3d, build_mesh_3d, solve_case_3d
from graph_dataset_3d import (
    build_edge_index_from_tets,
    discrete_operator_features_3d,
    material_interface_geometry_3d,
    tetra_p1_geometry_3d,
)
from test_fem_solver_3d import GeometryConfig3D, make_case_3d
from train import fem_energy_residual_loss_fn, interface_flux_loss_fn
from models import apply_fem_operator_correction


def _operator_data():
    geom = GeometryConfig3D(nx=5, ny=13, nz=5)
    mesh, basis = build_mesh_3d(geom)
    case = make_case_3d()
    result = solve_case_3d(mesh, basis, geom, case)
    conductive, robin, source = assemble_thermal_operator_3d(mesh, basis, geom, case)
    edge_index = build_edge_index_from_tets(mesh.t)
    _, _, op_index, op_value, op_diag = discrete_operator_features_3d(
        conductive, robin, source, result.is_bottom, edge_index, result.material_id
    )
    grad, _, _ = tetra_p1_geometry_3d(mesh.p, mesh.t, geom.y_tim)
    interface = material_interface_geometry_3d(mesh.p, mesh.t, geom)
    material_k = np.asarray([case.k_sub, case.k_cu, case.k_tim, case.k_die])
    data = Data(
        y=torch.from_numpy((result.T - case.t_ambient).astype(np.float32)),
        num_nodes=mesh.p.shape[1],
        fem_operator_index=torch.from_numpy(op_index),
        fem_operator_value=torch.from_numpy(op_value),
        fem_operator_diag=torch.from_numpy(op_diag),
        fem_source_load=torch.from_numpy(source.astype(np.float32)),
        fem_free_node=torch.from_numpy(~result.is_bottom),
        interface_node_index=torch.from_numpy(interface["node_index"]),
        interface_normal=torch.from_numpy(interface["normal"]),
        interface_area=torch.from_numpy(interface["area"]),
        interface_grad_phi_left=torch.from_numpy(grad[interface["tet_left"]]),
        interface_grad_phi_right=torch.from_numpy(grad[interface["tet_right"]]),
        interface_k_left=torch.from_numpy(material_k[interface["material_left"]].astype(np.float32)),
        interface_k_right=torch.from_numpy(material_k[interface["material_right"]].astype(np.float32)),
    )
    return geom, mesh, conductive, robin, source, data


def test_discrete_operator_reproduces_fem_energy_balance_and_robin_area():
    geom, mesh, conductive, robin, source, data = _operator_data()
    total = conductive + robin
    dT = data.y.numpy().astype(np.float64)
    residual = np.asarray(total @ dT - source).reshape(-1)
    assert np.max(np.abs(residual[data.fem_free_node.numpy()])) < 2e-6
    assert float(robin.sum()) == pytest.approx(
        data.interface_k_left.new_tensor(make_case_3d().h_top * geom.width * geom.depth).item(), rel=2e-6
    )
    assert data.fem_operator_index.shape[1] >= mesh.p.shape[1]


def test_energy_and_interface_flux_losses_are_batch_safe_and_zero_on_fem_target():
    _, _, _, _, _, data = _operator_data()
    batch = Batch.from_data_list([data, data])
    n = data.num_nodes
    assert batch.fem_operator_index[:, data.fem_operator_index.shape[1]].min().item() >= n
    assert batch.interface_node_index[:, data.interface_node_index.shape[1]].min().item() >= n
    energy = fem_energy_residual_loss_fn(batch.y, batch.y, batch, 0.0, 1.0)
    flux = interface_flux_loss_fn(batch.y, batch.y, batch)
    assert energy.item() < 1e-8
    assert flux.item() == pytest.approx(0.0, abs=1e-10)
    prediction = batch.y.clone()
    prediction[batch.interface_node_index[0, 0]] += 0.25
    assert fem_energy_residual_loss_fn(prediction, batch.y, batch, 0.0, 1.0).item() > energy.item()
    assert interface_flux_loss_fn(prediction, batch.y, batch).item() > 0.0


def test_damped_operator_correction_reduces_energy_residual():
    _, _, _, _, _, data = _operator_data()
    batch = Batch.from_data_list([data])
    raw = torch.zeros_like(batch.y)
    before = fem_energy_residual_loss_fn(raw, batch.y, batch, 0.0, 1.0)
    corrected = apply_fem_operator_correction(raw, batch, torch.tensor(0.0), torch.tensor(1.0), 4, 0.65)
    after = fem_energy_residual_loss_fn(corrected, batch.y, batch, 0.0, 1.0)
    assert after.item() < before.item()
    assert torch.all(corrected[~batch.fem_free_node.bool()] == 0.0)
