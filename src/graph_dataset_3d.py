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


def load_metadata(raw_dir: str) -> dict:
    with open(os.path.join(raw_dir, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_mesh_3d(raw_dir: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = np.load(os.path.join(raw_dir, "mesh.npz"))
    return mesh["points"], mesh["tets"]


def build_edge_index_from_tets(tets: np.ndarray) -> np.ndarray:
    """Build directed, de-duplicated graph edges from tetrahedron faces/edges."""
    undirected_edges = set()
    for tet in tets.T:
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            a, b = int(tet[i]), int(tet[j])
            undirected_edges.add((min(a, b), max(a, b)))
    directed_edges = [(a, b) for a, b in undirected_edges] + [(b, a) for a, b in undirected_edges]
    return np.asarray(sorted(directed_edges), dtype=np.int64).T


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
        geom = raw_meta["geometry"]
        self.width = geom["width"]
        self.depth = geom["depth"]
        self.height = sum(geom[key] for key in ("substrate_thickness", "cu_thickness", "tim_thickness", "die_thickness"))
        self.diag = float(np.sqrt(self.width ** 2 + self.height ** 2 + self.depth ** 2))

    def to_dict(self) -> dict:
        return {
            "q_ref": self.q_ref,
            "t_ref": self.t_ref,
            "h_ref": self.h_ref,
            "k_ref": self.k_ref,
            "inv_k_ref": self.inv_k_ref,
            "inv_h_ref": self.inv_h_ref,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "diag": self.diag,
        }


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
    """Return 24 node features: 3D coordinates, local fields and global case data."""
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
]
EDGE_FEATURE_NAMES_3D = ["dx_norm", "dy_norm", "dz_norm", "edge_len_norm", "k_eq_norm"]


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
    edge_index_np = build_edge_index_from_tets(tets)
    edge_index_t = torch.from_numpy(edge_index_np)
    split_data = {name: [] for name in ("train", "val", "test")}
    split_info = {name: [] for name in ("train", "val", "test")}
    train_dT = []

    for record in raw_meta["samples"]:
        raw = np.load(os.path.join(raw_dir, record["file"]))
        case = record["case"]
        x = build_node_features_3d(
            points, raw["material_id"], raw["q_node"], raw["h_node"], raw["is_top"], raw["is_bottom"], case, nc
        )
        edge_attr = build_edge_features_3d(points, edge_index_np, raw["material_id"], case, nc)
        dT = (raw["T"] - case["t_ambient"]).astype(np.float32)
        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index_t,
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.from_numpy(dT),
            pos=torch.from_numpy(points.T.astype(np.float32)),
        )
        data.t_ambient = torch.tensor([case["t_ambient"]], dtype=torch.float32)
        data.dT_max_true = torch.tensor([float(dT.max())], dtype=torch.float32)
        data.solve_time_s = torch.tensor([record["solve_time_s"]], dtype=torch.float32)
        data.mesh_dimension = torch.tensor([3], dtype=torch.int64)
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
        "node_feature_dim": len(NODE_FEATURE_NAMES_3D),
        "edge_feature_dim": len(EDGE_FEATURE_NAMES_3D),
        "node_feature_names": NODE_FEATURE_NAMES_3D,
        "edge_feature_names": EDGE_FEATURE_NAMES_3D,
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
