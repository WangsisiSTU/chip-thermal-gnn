"""Convert raw 3D tetrahedral FEM samples to PyTorch Geometric graphs.

This is deliberately separate from ``graph_dataset.py`` so the existing 2D
feature contract (21 node and 4 edge features) remains stable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import torch
    from torch_geometric.data import Data

    HAS_PYG = True
except ImportError:  # pragma: no cover - allows geometry/feature checks in a minimal environment
    HAS_PYG = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem_solver_3d import (
    HAS_SKFEM, GeometryConfig3D, CaseParams3D,
    _TetraMeshFallback, assemble_thermal_operator_3d, lumped_mass_3d,
    material_id_of_points_3d, source_load_3d,
)
if HAS_SKFEM:
    from fem_solver_3d import Basis, ElementTetP1, MeshTet


def load_metadata(raw_dir: str) -> dict:
    with open(os.path.join(raw_dir, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_mesh_3d(raw_dir: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = np.load(os.path.join(raw_dir, "mesh.npz"))
    return mesh["points"], mesh["tets"]


def build_edge_index_from_tets(tets: np.ndarray) -> np.ndarray:
    """Build directed, de-duplicated graph edges from tetrahedron faces/edges."""
    tets = np.asarray(tets)
    if tets.ndim != 2 or tets.shape[0] != 4:
        raise ValueError("tets must have shape (4, n_tets)")
    if not np.issubdtype(tets.dtype, np.integer) or np.any(tets < 0):
        raise ValueError("tets must contain non-negative integer node indices")
    pairs = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    edges = tets[pairs].transpose(0, 2, 1).reshape(-1, 2)
    directed = np.ascontiguousarray(np.concatenate((edges, edges[:, ::-1])), dtype=np.int64)
    # Sort columns directly, then remove adjacent duplicates without Python objects.
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    keep = np.ones(len(directed), dtype=bool)
    keep[1:] = np.any(directed[1:] != directed[:-1], axis=1)
    return directed[keep].T


def lumped_node_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """P1 mass-lumped integration weights; normalize to mean one."""
    mass = lumped_mass_3d(points, tets)
    return (mass / mass.mean()).astype(np.float32)


def tetra_p1_geometry_3d(points: np.ndarray, tets: np.ndarray, die_start_y: float):
    """Within-element P1 gradients; never differentiate across a material face."""
    vertices = points[:, tets].transpose(2, 1, 0)
    affine = np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=-1)
    inverse = np.linalg.inv(affine)
    grad_phi = inverse[:, 1:, :].transpose(0, 2, 1).astype(np.float32)
    volume = (np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 6.0).astype(np.float32)
    die = vertices[:, :, 1].mean(axis=1) > die_start_y + 1e-12
    return grad_phi, volume, die


def material_interface_geometry_3d(points: np.ndarray, tets: np.ndarray, geom: GeometryConfig3D):
    """Return paired P1 elements on conforming internal material interfaces."""
    center = points[:, tets].mean(axis=1)
    tet_material = material_id_of_points_3d(center[0], center[1], center[2], geom)
    local_faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    owners: dict[tuple[int, int, int], int] = {}
    pairs = []
    face_nodes = []
    for tet_id, tet in enumerate(tets.T):
        for local in local_faces:
            face = tuple(sorted(int(tet[index]) for index in local))
            other = owners.pop(face, None)
            if other is None:
                owners[face] = tet_id
            elif tet_material[other] != tet_material[tet_id]:
                pairs.append((other, tet_id))
                face_nodes.append(face)
    if not pairs:
        return {
            "node_index": np.empty((8, 0), dtype=np.int64),
            "normal": np.empty((0, 3), dtype=np.float32),
            "area": np.empty(0, dtype=np.float32),
            "material_left": np.empty(0, dtype=np.int64),
            "material_right": np.empty(0, dtype=np.int64),
            "tet_left": np.empty(0, dtype=np.int64),
            "tet_right": np.empty(0, dtype=np.int64),
        }
    pair_array = np.asarray(pairs, dtype=np.int64)
    faces = np.asarray(face_nodes, dtype=np.int64)
    coords = points[:, faces].transpose(1, 2, 0)
    cross = np.cross(coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    if np.any(norm <= 0):
        raise ValueError("material interface contains a degenerate triangular face")
    return {
        "node_index": np.concatenate((tets[:, pair_array[:, 0]], tets[:, pair_array[:, 1]]), axis=0),
        "normal": (cross / norm[:, None]).astype(np.float32),
        "area": (0.5 * norm).astype(np.float32),
        "material_left": tet_material[pair_array[:, 0]].astype(np.int64),
        "material_right": tet_material[pair_array[:, 1]].astype(np.int64),
        "tet_left": pair_array[:, 0],
        "tet_right": pair_array[:, 1],
    }


def discrete_operator_features_3d(conductive, robin, source_load: np.ndarray,
                                  is_bottom: np.ndarray, edge_index: np.ndarray,
                                  material_id: np.ndarray):
    """Build row-normalized FEM features plus the exact sparse energy operator."""
    total = conductive + robin
    if hasattr(total, "tocsr"):
        conductive = conductive.tocsr()
        robin = robin.tocsr()
        total = total.tocsr()
        total.eliminate_zeros()
        diag = np.asarray(total.diagonal()).reshape(-1)
        cond_diag = np.asarray(conductive.diagonal()).reshape(-1)
        robin_diag = np.asarray(robin.diagonal()).reshape(-1)
        src, dst = edge_index
        cond_edge = np.asarray(conductive[src, dst]).reshape(-1)
        robin_edge = np.asarray(robin[src, dst]).reshape(-1)
        coo = total.tocoo()
        operator_index = np.vstack((coo.row, coo.col)).astype(np.int64)
        operator_value = np.asarray(coo.data, dtype=np.float32)
    else:
        total = np.asarray(total)
        conductive = np.asarray(conductive)
        robin = np.asarray(robin)
        diag = np.diag(total)
        cond_diag = np.diag(conductive)
        robin_diag = np.diag(robin)
        src, dst = edge_index
        cond_edge = conductive[src, dst]
        robin_edge = robin[src, dst]
        row, col = np.nonzero(total)
        operator_index = np.vstack((row, col)).astype(np.int64)
        operator_value = total[row, col].astype(np.float32)
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        raise ValueError("FEM operator must have a finite positive diagonal")
    node_features = np.column_stack((
        source_load / diag,
        cond_diag / diag,
        robin_diag / diag,
        (~np.asarray(is_bottom, dtype=bool)).astype(np.float64),
    )).astype(np.float32)
    edge_features = np.column_stack((
        cond_edge / diag[src],
        robin_edge / diag[src],
        (material_id[src] != material_id[dst]).astype(np.float64),
    )).astype(np.float32)
    return node_features, edge_features, operator_index, operator_value, diag.astype(np.float32)



def k_by_material(material_id: np.ndarray, case: dict) -> np.ndarray:
    return np.asarray([case["k_sub"], case["k_cu"], case["k_tim"], case["k_die"]], dtype=np.float64)[material_id]


class NormConsts3D:
    """Feature normalization constants derived from declared sampling limits."""

    def __init__(self, raw_meta: dict):
        mat, hs, bc, ood = raw_meta["materials"], raw_meta["heat_source"], raw_meta["boundary"], raw_meta["ood"]
        self.q_ref = max(hs["q_hot"]["high"], ood["q_hot"]["high"])
        self.t_ref = bc["t_ambient"]["high"]
        self.h_ref = bc["h_top"]["high"]
        self.k_ref = max(mat["k_cu"]["high"], 400.0)
        self.inv_k_ref = 1.0 / min(mat["k_tim"]["low"], ood["k_tim"]["low"])
        self.inv_h_ref = 1.0 / min(bc["h_top"]["low"], ood["h_top"]["low"])
        coverage_h = raw_meta.get("coverage", {}).get("poor_cooling", {}).get("h_top")
        h_lows = [bc["h_top"]["low"], ood["h_top"]["low"]]
        h_highs = [bc["h_top"]["high"], ood["h_top"]["high"]]
        if coverage_h is not None:
            h_lows.append(coverage_h["low"])
            h_highs.append(coverage_h["high"])
        self.log_h_low = float(np.log(min(h_lows)))
        self.log_h_span = float(np.log(max(h_highs)) - self.log_h_low)
        if self.log_h_span <= 0:
            raise ValueError("h_top sampling range must be positive and non-degenerate")
        geom = raw_meta["geometry"]
        self.width = geom["width"]
        self.depth = geom["depth"]
        self.height = sum(geom[key] for key in ("substrate_thickness", "cu_thickness", "tim_thickness", "die_thickness"))
        self.diag = float(np.sqrt(self.width ** 2 + self.height ** 2 + self.depth ** 2))
        self.substrate_thickness = geom["substrate_thickness"]
        self.cu_thickness = geom["cu_thickness"]
        self.tim_thickness = geom["tim_thickness"]
        self.die_thickness = geom["die_thickness"]

    def to_dict(self) -> dict:
        return {
            "q_ref": self.q_ref,
            "t_ref": self.t_ref,
            "h_ref": self.h_ref,
            "k_ref": self.k_ref,
            "inv_k_ref": self.inv_k_ref,
            "inv_h_ref": self.inv_h_ref,
            "log_h_low": self.log_h_low,
            "log_h_span": self.log_h_span,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "diag": self.diag,
        }


def cooling_path_resistance_fraction_3d(case: dict, nc: NormConsts3D) -> float:
    """Approximate how much of the die-centre thermal path is top-side resistance."""
    top_resistance = nc.die_thickness / (2.0 * case["k_die"]) + 1.0 / case["h_top"]
    bottom_resistance = (
        nc.die_thickness / (2.0 * case["k_die"])
        + nc.tim_thickness / case["k_tim"]
        + nc.cu_thickness / case["k_cu"]
        + nc.substrate_thickness / case["k_sub"]
    )
    return float(top_resistance / (top_resistance + bottom_resistance))

def build_node_features_3d(
    points: np.ndarray,
    material_id: np.ndarray,
    q_node: np.ndarray,
    h_node: np.ndarray,
    is_top: np.ndarray,
    is_bottom: np.ndarray,
    case: dict,
    nc: NormConsts3D,
) -> np.ndarray:
    """Return 26 node features: coordinates, local fields, and global physics."""
    n_nodes = points.shape[1]
    one_hot = np.zeros((n_nodes, 4), dtype=np.float64)
    one_hot[np.arange(n_nodes), material_id] = 1.0
    global_values = np.asarray(
        [
            case["q_base"] / nc.q_ref,
            case["q_hot"] / nc.q_ref,
            case["hotspot_center_x_frac"],
            case["hotspot_center_z_frac"],
            case["hotspot_width_x_frac"],
            case["hotspot_width_z_frac"],
            case["k_die"] / nc.k_ref,
            case["k_cu"] / nc.k_ref,
            case["k_sub"] / nc.k_ref,
            case["k_tim"] / nc.k_ref,
            (1.0 / case["k_tim"]) / nc.inv_k_ref,
            (1.0 / case["h_top"]) / nc.inv_h_ref,
            (np.log(case["h_top"]) - nc.log_h_low) / nc.log_h_span,
            cooling_path_resistance_fraction_3d(case, nc),
        ],
        dtype=np.float64,
    )
    features = np.column_stack(
        [
            points[0] / nc.width,
            points[1] / nc.height,
            points[2] / nc.depth,
            one_hot,
            q_node / nc.q_ref,
            np.full(n_nodes, case["t_ambient"] / nc.t_ref),
            h_node / nc.h_ref,
            is_top.astype(np.float64),
            is_bottom.astype(np.float64),
            np.tile(global_values, (n_nodes, 1)),
        ]
    )
    return features.astype(np.float32)


def build_edge_features_3d(
    points: np.ndarray,
    edge_index: np.ndarray,
    material_id: np.ndarray,
    case: dict,
    nc: NormConsts3D,
) -> np.ndarray:
    """Return dx/dy/dz, normalized length, and harmonic-mean conductivity."""
    src, dst = edge_index
    delta = points[:, dst] - points[:, src]
    k_node = k_by_material(material_id, case)
    k_eq = 2.0 * k_node[src] * k_node[dst] / (k_node[src] + k_node[dst])
    return np.column_stack(
        [
            delta[0] / nc.width,
            delta[1] / nc.height,
            delta[2] / nc.depth,
            np.linalg.norm(delta, axis=0) / nc.diag,
            k_eq / nc.k_ref,
        ]
    ).astype(np.float32)


NODE_FEATURE_NAMES_3D = [
    "x_norm", "y_norm", "z_norm",
    "mat_substrate", "mat_cu", "mat_tim", "mat_die",
    "q_norm", "t_ambient_norm", "h_norm", "is_top", "is_bottom",
    "g_q_base_norm", "g_q_hot_norm",
    "g_hotspot_center_x_frac", "g_hotspot_center_z_frac",
    "g_hotspot_width_x_frac", "g_hotspot_width_z_frac",
    "g_k_die_norm", "g_k_cu_norm", "g_k_sub_norm", "g_k_tim_norm",
    "g_inv_k_tim_norm", "g_inv_h_top_norm",
    "g_log_h_top_norm", "g_top_path_resistance_fraction",
]
EDGE_FEATURE_NAMES_3D = ["dx_norm", "dy_norm", "dz_norm", "edge_len_norm", "k_eq_norm"]
OPERATOR_NODE_FEATURE_NAMES_3D = [
    "fem_source_over_diag_K", "fem_conductive_diag_fraction",
    "fem_robin_diag_fraction", "fem_free_node",
]
OPERATOR_EDGE_FEATURE_NAMES_3D = [
    "fem_conductive_row_weight", "fem_robin_row_weight", "cross_material_interface",
]


def build_dataset_3d(raw_dir: str, out_dir: str) -> dict:
    """Build ``train.pt``, ``val.pt`` and ``test.pt`` containing 3D PyG Data objects."""
    if not HAS_PYG:
        raise RuntimeError("PyTorch and torch_geometric are required to write processed 3D PyG datasets")
    os.makedirs(out_dir, exist_ok=True)
    raw_meta = load_metadata(raw_dir)
    if raw_meta.get("dimension") not in (None, 3):
        raise ValueError("Raw metadata does not describe a 3D dataset")
    points, tets = load_mesh_3d(raw_dir)
    nc = NormConsts3D(raw_meta)
    geom = GeometryConfig3D.from_dict(raw_meta["geometry"])
    fem_mesh = MeshTet(points, tets) if HAS_SKFEM else _TetraMeshFallback(points, tets)
    fem_basis = Basis(fem_mesh, ElementTetP1()) if HAS_SKFEM else None
    physical_node_volume = lumped_mass_3d(points, tets)
    tet_grad_phi, tet_volume, tet_die = tetra_p1_geometry_3d(points, tets, geom.y_tim)
    interface = material_interface_geometry_3d(points, tets, geom)
    edge_index_np = build_edge_index_from_tets(tets)
    edge_index_t = torch.from_numpy(edge_index_np)
    node_volume_t = torch.from_numpy(lumped_node_volumes(points, tets))
    tet_index_t = torch.from_numpy(tets.astype(np.int64))
    tet_grad_t = torch.from_numpy(tet_grad_phi)
    tet_volume_t = torch.from_numpy(tet_volume)
    tet_die_t = torch.from_numpy(tet_die)
    interface_node_index_t = torch.from_numpy(interface["node_index"])
    interface_normal_t = torch.from_numpy(interface["normal"])
    interface_area_t = torch.from_numpy(interface["area"])
    interface_grad_left_t = torch.from_numpy(tet_grad_phi[interface["tet_left"]])
    interface_grad_right_t = torch.from_numpy(tet_grad_phi[interface["tet_right"]])
    split_data = {name: [] for name in ("train", "val", "test")}
    split_info = {name: [] for name in ("train", "val", "test")}
    train_dT = []

    for record in raw_meta["samples"]:
        raw = np.load(os.path.join(raw_dir, record["file"]))
        case = record["case"]
        case_params = CaseParams3D(**case)
        conductive, robin, source_load = assemble_thermal_operator_3d(fem_mesh, fem_basis, geom, case_params)
        q_projected = source_load / physical_node_volume
        x_base = build_node_features_3d(
            points, raw["material_id"], q_projected, raw["h_node"], raw["is_top"], raw["is_bottom"], case, nc
        )
        edge_base = build_edge_features_3d(points, edge_index_np, raw["material_id"], case, nc)
        op_node, op_edge, op_index, op_value, op_diag = discrete_operator_features_3d(
            conductive, robin, source_load, raw["is_bottom"], edge_index_np, raw["material_id"]
        )
        x = np.column_stack((x_base, op_node)).astype(np.float32)
        edge_attr = np.column_stack((edge_base, op_edge)).astype(np.float32)
        dT = (raw["T"] - case["t_ambient"]).astype(np.float32)
        material_k = np.asarray([case["k_sub"], case["k_cu"], case["k_tim"], case["k_die"]])
        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index_t,
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.from_numpy(dT),
            pos=torch.from_numpy(points.T.astype(np.float32)),
            node_volume=node_volume_t,
            tet_index=tet_index_t,
            tet_grad_phi=tet_grad_t,
            tet_volume=tet_volume_t,
            tet_die=tet_die_t,
            fem_operator_index=torch.from_numpy(op_index),
            fem_operator_value=torch.from_numpy(op_value),
            fem_operator_diag=torch.from_numpy(op_diag),
            fem_source_load=torch.from_numpy(source_load.astype(np.float32)),
            fem_free_node=torch.from_numpy((~raw["is_bottom"].astype(bool))),
            interface_node_index=interface_node_index_t,
            interface_normal=interface_normal_t,
            interface_area=interface_area_t,
            interface_grad_phi_left=interface_grad_left_t,
            interface_grad_phi_right=interface_grad_right_t,
            interface_k_left=torch.from_numpy(material_k[interface["material_left"]].astype(np.float32)),
            interface_k_right=torch.from_numpy(material_k[interface["material_right"]].astype(np.float32)),
        )
        data.t_ambient = torch.tensor([case["t_ambient"]], dtype=torch.float32)
        data.dT_max_true = torch.tensor([float(dT.max())], dtype=torch.float32)
        data.solve_time_s = torch.tensor([record["solve_time_s"]], dtype=torch.float32)
        data.mesh_dimension = torch.tensor([3], dtype=torch.int64)
        data.source_power_W = torch.tensor([float(source_load.sum())], dtype=torch.float32)
        split = record["split"]
        split_data[split].append(data)
        split_info[split].append({"file": record["file"], "regime": record["regime"], "case": case})
        if split == "train":
            train_dT.append(dT)

    if not train_dT:
        raise ValueError("The raw dataset has no training samples")
    dT_train = np.concatenate(train_dT)
    metadata = {
        "dimension": 3,
        "element_type": "tetrahedron_p1",
        "n_nodes": int(points.shape[1]),
        "n_tets": int(tets.shape[1]),
        "n_edges": int(edge_index_np.shape[1]),
        "split_counts": {name: len(items) for name, items in split_data.items()},
        "node_feature_dim": len(NODE_FEATURE_NAMES_3D) + len(OPERATOR_NODE_FEATURE_NAMES_3D),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES_3D) + len(OPERATOR_EDGE_FEATURE_NAMES_3D),
        "node_feature_names": NODE_FEATURE_NAMES_3D + OPERATOR_NODE_FEATURE_NAMES_3D,
        "edge_feature_names": EDGE_FEATURE_NAMES_3D + OPERATOR_EDGE_FEATURE_NAMES_3D,
        "node_volume_weighting": True,
        "heat_source_projection": "fem_load_lumped_v1",
        "tetra_gradient_geometry": "p1_physical_m_v1",
        "discrete_operator": "p1_conductive_plus_robin_sparse_v1",
        "material_interface_flux": "paired_p1_tetra_faces_v1",
        "n_material_interface_faces": int(interface["area"].size),
        "norm_consts": nc.to_dict(),
        "dT_train_mean": float(dT_train.mean()),
        "dT_train_std": float(dT_train.std()),
        "source_raw_metadata": os.path.join(raw_dir, "metadata.json"),
    }
    for split in split_data:
        torch.save(split_data[split], os.path.join(out_dir, f"{split}.pt"))
        with open(os.path.join(out_dir, f"{split}_info.json"), "w", encoding="utf-8") as f:
            json.dump(split_info[split], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Convert 3D tetrahedral FEM samples to PyG graph data.")
    parser.add_argument("--raw_dir", default="data/raw_3d")
    parser.add_argument("--out_dir", default="data/processed_3d")
    args = parser.parse_args()
    metadata = build_dataset_3d(args.raw_dir, args.out_dir)
    print(
        f"3D PyG dataset complete: node_features={metadata['node_feature_dim']}, "
        f"edge_features={metadata['edge_feature_dim']}, edges={metadata['n_edges']}"
    )


if __name__ == "__main__":
    main()
