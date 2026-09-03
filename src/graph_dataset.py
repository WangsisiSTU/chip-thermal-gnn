"""
将有限元网格样本转换为 PyTorch Geometric 的图数据（Data 对象），并保存为
train/val/test 三个 .pt 文件，同时生成记录归一化常数与特征说明的 metadata.json。

节点特征 x（共 21 维）：
    局部特征：
    [0:2]  归一化坐标 (x/W, y/H)
    [2:6]  材料 one-hot（基板/铜/TIM/Die）
    [6]    归一化局部热源体密度 q_node / q_ref
    [7]    归一化环境温度 t_ambient / T_ref（全图广播）
    [8]    归一化顶部对流换热系数 h_node / h_ref（非顶部边界节点为 0）
    [9]    是否为顶部对流边界节点
    [10]   是否为底部恒温边界节点
    全局工况特征（广播到所有节点；稳态椭圆问题解全局耦合，而消息传递轮数有限，
    若不广播，远离热源/边界的节点无法感知工况参数）：
    [11]   q_base / q_ref
    [12]   q_hot / q_ref
    [13]   热点中心位置比例 hotspot_center_frac
    [14]   热点宽度比例 hotspot_width_frac
    [15]   k_die / k_ref
    [16]   k_cu / k_ref
    [17]   k_sub / k_ref
    [18]   k_tim / k_ref
    [19]   归一化 TIM 热阻 (1/k_tim) / inv_k_ref（对温升近似线性的物理特征）
    [20]   归一化对流热阻 (1/h_top) / inv_h_ref

边特征 edge_attr（共 4 维）：
    [0:2]  归一化相对坐标 (dx/W, dy/H)
    [2]    归一化边长（按对角线归一化）
    [3]    两端点材料等效导热系数（调和平均）/ k_ref

标签 y：节点温升 dT = T - T_ambient [K]（未归一化，训练时在 train.py 中做标准化）
pos：节点物理坐标 [m]，用于可视化
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem_solver import GeometryConfig


def load_metadata(raw_dir: str) -> dict:
    with open(os.path.join(raw_dir, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_mesh(raw_dir: str):
    d = np.load(os.path.join(raw_dir, "mesh.npz"))
    return d["points"], d["tris"]


def build_edge_index(tris: np.ndarray) -> np.ndarray:
    """从三角形单元生成无向边（每条边保留两个方向），去除重复边。"""
    edges = set()
    for a, b, c in tris.T:
        for i, j in [(a, b), (b, c), (c, a)]:
            edges.add((int(i), int(j)))
            edges.add((int(j), int(i)))
    edge_index = np.array(sorted(edges), dtype=np.int64).T  # (2, E)
    return edge_index


def k_by_material(material_id: np.ndarray, case: dict) -> np.ndarray:
    k_map = np.array([case["k_sub"], case["k_cu"], case["k_tim"], case["k_die"]])
    return k_map[material_id]


class NormConsts:
    """基于配置中已知的物理采样范围（含 OOD）确定的归一化常数，与训练集统计量无关，避免数据泄漏。"""

    def __init__(self, raw_meta: dict):
        mat, hs, bc, ood = raw_meta["materials"], raw_meta["heat_source"], raw_meta["boundary"], raw_meta["ood"]
        self.q_ref = max(hs["q_hot"]["high"], ood["q_hot"]["high"])
        self.t_ref = bc["t_ambient"]["high"]
        self.h_ref = bc["h_top"]["high"]
        self.k_ref = max(mat["k_cu"]["high"], 400.0)
        # 热阻类特征的归一化参考值：取 k_tim 与 h_top 全范围（含 OOD）的最小值对应的最大热阻
        self.inv_k_ref = 1.0 / min(mat["k_tim"]["low"], ood["k_tim"]["low"])
        self.inv_h_ref = 1.0 / min(bc["h_top"]["low"], ood["h_top"]["low"])
        geom = raw_meta["geometry"]
        self.width = geom["width"]
        self.height = (
            geom["substrate_thickness"] + geom["cu_thickness"] + geom["tim_thickness"] + geom["die_thickness"]
        )
        self.diag = float(np.hypot(self.width, self.height))

    def to_dict(self):
        return {
            "q_ref": self.q_ref,
            "t_ref": self.t_ref,
            "h_ref": self.h_ref,
            "k_ref": self.k_ref,
            "inv_k_ref": self.inv_k_ref,
            "inv_h_ref": self.inv_h_ref,
            "width": self.width,
            "height": self.height,
            "diag": self.diag,
        }


def build_node_features(points, material_id, q_node, h_node, is_top, is_bottom, case: dict, nc: NormConsts):
    x_norm = points[0] / nc.width
    y_norm = points[1] / nc.height
    n = points.shape[1]
    one_hot = np.zeros((n, 4), dtype=np.float64)
    one_hot[np.arange(n), material_id] = 1.0
    q_feat = q_node / nc.q_ref
    t_feat = np.full(n, case["t_ambient"] / nc.t_ref)
    h_feat = h_node / nc.h_ref

    # 全局工况特征，广播到所有节点
    global_vals = np.array(
        [
            case["q_base"] / nc.q_ref,
            case["q_hot"] / nc.q_ref,
            case["hotspot_center_frac"],
            case["hotspot_width_frac"],
            case["k_die"] / nc.k_ref,
            case["k_cu"] / nc.k_ref,
            case["k_sub"] / nc.k_ref,
            case["k_tim"] / nc.k_ref,
            (1.0 / case["k_tim"]) / nc.inv_k_ref,
            (1.0 / case["h_top"]) / nc.inv_h_ref,
        ]
    )
    global_feats = np.tile(global_vals, (n, 1))

    feats = np.column_stack(
        [
            x_norm, y_norm, one_hot, q_feat, t_feat, h_feat,
            is_top.astype(np.float64), is_bottom.astype(np.float64),
            global_feats,
        ]
    )
    return feats.astype(np.float32)


def build_edge_features(points, edge_index, material_id, case, nc: NormConsts):
    src, dst = edge_index[0], edge_index[1]
    dx = (points[0, dst] - points[0, src]) / nc.width
    dy = (points[1, dst] - points[1, src]) / nc.height
    length = np.hypot(points[0, dst] - points[0, src], points[1, dst] - points[1, src]) / nc.diag
    k_node = k_by_material(material_id, case)
    k_src, k_dst = k_node[src], k_node[dst]
    k_eq = 2 * k_src * k_dst / (k_src + k_dst)
    k_eq_norm = k_eq / nc.k_ref
    feats = np.column_stack([dx, dy, length, k_eq_norm])
    return feats.astype(np.float32)


def build_dataset(raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    raw_meta = load_metadata(raw_dir)
    points, tris = load_mesh(raw_dir)
    nc = NormConsts(raw_meta)

    edge_index_np = build_edge_index(tris)
    edge_index_t = torch.from_numpy(edge_index_np)

    split_data = {"train": [], "val": [], "test": []}
    split_info = {"train": [], "val": [], "test": []}

    dT_train_values = []

    for rec in raw_meta["samples"]:
        raw = np.load(os.path.join(raw_dir, rec["file"]))
        T = raw["T"]
        material_id = raw["material_id"]
        q_node = raw["q_node"]
        is_top = raw["is_top"]
        is_bottom = raw["is_bottom"]
        h_node = raw["h_node"]
        case = rec["case"]
        t_amb = case["t_ambient"]

        x = build_node_features(points, material_id, q_node, h_node, is_top, is_bottom, case, nc)
        edge_attr = build_edge_features(points, edge_index_np, material_id, case, nc)
        dT = (T - t_amb).astype(np.float32)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index_t,
            edge_attr=torch.from_numpy(edge_attr),
            y=torch.from_numpy(dT),
            pos=torch.from_numpy(points.T.astype(np.float32)),
        )
        data.t_ambient = torch.tensor([t_amb], dtype=torch.float32)
        data.dT_max_true = torch.tensor([float(dT.max())], dtype=torch.float32)
        data.solve_time_s = torch.tensor([rec["solve_time_s"]], dtype=torch.float32)

        split = rec["split"]
        split_data[split].append(data)
        split_info[split].append({"file": rec["file"], "regime": rec["regime"], "case": case})

        if split == "train":
            dT_train_values.append(dT)

    dT_all_train = np.concatenate(dT_train_values)
    dT_mean, dT_std = float(dT_all_train.mean()), float(dT_all_train.std())

    for split in ["train", "val", "test"]:
        torch.save(split_data[split], os.path.join(out_dir, f"{split}.pt"))
        with open(os.path.join(out_dir, f"{split}_info.json"), "w", encoding="utf-8") as f:
            json.dump(split_info[split], f, ensure_ascii=False, indent=2)

    node_feature_names = [
        "x_norm", "y_norm", "mat_substrate", "mat_cu", "mat_tim", "mat_die",
        "q_norm", "t_ambient_norm", "h_norm", "is_top", "is_bottom",
        "g_q_base_norm", "g_q_hot_norm", "g_hotspot_center_frac", "g_hotspot_width_frac",
        "g_k_die_norm", "g_k_cu_norm", "g_k_sub_norm", "g_k_tim_norm",
        "g_inv_k_tim_norm", "g_inv_h_top_norm",
    ]
    edge_feature_names = ["dx_norm", "dy_norm", "edge_len_norm", "k_eq_norm"]

    metadata = {
        "seed": raw_meta["seed"],
        "n_nodes": raw_meta["n_nodes"],
        "n_tris": raw_meta["n_tris"],
        "n_edges": int(edge_index_np.shape[1]),
        "split_counts": {k: len(v) for k, v in split_data.items()},
        "node_feature_dim": len(node_feature_names),
        "edge_feature_dim": len(edge_feature_names),
        "node_feature_names": node_feature_names,
        "edge_feature_names": edge_feature_names,
        "norm_consts": nc.to_dict(),
        "dT_train_mean": dT_mean,
        "dT_train_std": dT_std,
        "source_raw_metadata": "data/raw/metadata.json",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"图数据集构建完成: train={len(split_data['train'])}, val={len(split_data['val'])}, test={len(split_data['test'])}")
    print(f"节点特征维度: {len(node_feature_names)}, 边特征维度: {len(edge_feature_names)}, 边数: {edge_index_np.shape[1]}")
    print(f"训练集 dT 均值/标准差: {dT_mean:.4f} / {dT_std:.4f} K")


def main():
    parser = argparse.ArgumentParser(description="将 FEM 原始样本转换为 PyG 图数据集")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
