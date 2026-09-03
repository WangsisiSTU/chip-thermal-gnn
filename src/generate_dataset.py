"""
数据集生成脚本：随机采样芯片封装工况参数，调用有限元求解器求解稳态温度场，
并将网格、材料、热源、边界特征与温度场标签保存为原始样本文件（.npz），
同时生成记录采样范围/数据划分/随机种子的 metadata.json。

用法：
    python src/generate_dataset.py --config configs/data_config.yaml --out_dir data/raw
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem_solver import CaseParams, GeometryConfig, build_mesh, solve_case


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _uniform(rng: np.random.Generator, rng_dict: dict) -> float:
    return rng.uniform(rng_dict["low"], rng_dict["high"])


def sample_id_case(rng: np.random.Generator, cfg: dict) -> CaseParams:
    mat = cfg["materials"]
    hs = cfg["heat_source"]
    bc = cfg["boundary"]
    return CaseParams(
        k_die=_uniform(rng, mat["k_die"]),
        k_tim=_uniform(rng, mat["k_tim"]),
        k_cu=_uniform(rng, mat["k_cu"]),
        k_sub=_uniform(rng, mat["k_sub"]),
        q_base=_uniform(rng, hs["q_base"]),
        q_hot=_uniform(rng, hs["q_hot"]),
        hotspot_center_frac=_uniform(rng, hs["hotspot_center_frac"]),
        hotspot_width_frac=_uniform(rng, hs["hotspot_width_frac"]),
        t_ambient=_uniform(rng, bc["t_ambient"]),
        h_top=_uniform(rng, bc["h_top"]),
        regime="id",
    )


def sample_ood_case(rng: np.random.Generator, cfg: dict, kind: str) -> CaseParams:
    """采样分布外（OOD）工况：某一个关键因子超出训练集范围，其余因子仍在正常范围内采样。"""
    mat = cfg["materials"]
    hs = cfg["heat_source"]
    bc = cfg["boundary"]
    ood = cfg["ood"]

    case = sample_id_case(rng, cfg)
    if kind == "high_power":
        case.q_hot = _uniform(rng, ood["q_hot"])
    elif kind == "poor_tim":
        case.k_tim = _uniform(rng, ood["k_tim"])
    elif kind == "poor_cooling":
        case.h_top = _uniform(rng, ood["h_top"])
    else:
        raise ValueError(f"未知 OOD 类型: {kind}")
    case.regime = f"ood_{kind}"
    return case


def build_case_list(cfg: dict, rng: np.random.Generator):
    split = cfg["split"]
    n_train, n_val, n_test = split["n_train"], split["n_val"], split["n_test"]
    n_ood = cfg["ood"]["n_ood_test_samples"]
    n_test_id = n_test - n_ood
    assert n_test_id >= 0

    cases = []
    for _ in range(n_train):
        cases.append(("train", sample_id_case(rng, cfg)))
    for _ in range(n_val):
        cases.append(("val", sample_id_case(rng, cfg)))
    for _ in range(n_test_id):
        cases.append(("test", sample_id_case(rng, cfg)))

    ood_kinds = ["high_power", "poor_tim", "poor_cooling"]
    counts = [n_ood // 3 + (1 if i < n_ood % 3 else 0) for i in range(3)]
    for kind, cnt in zip(ood_kinds, counts):
        for _ in range(cnt):
            cases.append(("test", sample_ood_case(rng, cfg, kind)))

    return cases


def main():
    parser = argparse.ArgumentParser(description="生成芯片封装二维稳态热场 FEM 数据集")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--out_dir", type=str, default="data/raw")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)

    geom = GeometryConfig.from_dict(cfg["geometry"])
    mesh, basis = build_mesh(geom)
    n_nodes, n_tris = mesh.p.shape[1], mesh.t.shape[1]
    print(f"网格: {n_nodes} 节点, {n_tris} 三角单元 (共用于所有样本)")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "mesh.npz"),
        points=mesh.p,
        tris=mesh.t,
    )

    cases = build_case_list(cfg, rng)
    print(f"共生成 {len(cases)} 个工况样本")

    records = []
    solve_times = []
    dT_maxes = []
    t0_all = time.perf_counter()
    for idx, (split_name, case) in enumerate(tqdm(cases, desc="FEM 求解")):
        result = solve_case(mesh, basis, geom, case)
        fname = f"sample_{idx:04d}.npz"
        np.savez(
            os.path.join(args.out_dir, fname),
            T=result.T.astype(np.float64),
            material_id=result.material_id.astype(np.int64),
            q_node=result.q_node.astype(np.float64),
            is_top=result.is_top,
            is_bottom=result.is_bottom,
            h_node=result.h_node.astype(np.float64),
        )
        solve_times.append(result.solve_time_s)
        dT_max = float(result.T.max() - case.t_ambient)
        dT_maxes.append(dT_max)
        records.append(
            {
                "file": fname,
                "split": split_name,
                "regime": case.regime,
                "case": case.to_dict(),
                "solve_time_s": result.solve_time_s,
                "T_min": float(result.T.min()),
                "T_max": float(result.T.max()),
                "dT_max": dT_max,
            }
        )
    total_time = time.perf_counter() - t0_all

    metadata = {
        "seed": seed,
        "n_nodes": n_nodes,
        "n_tris": n_tris,
        "geometry": cfg["geometry"],
        "materials": cfg["materials"],
        "heat_source": cfg["heat_source"],
        "boundary": cfg["boundary"],
        "ood": cfg["ood"],
        "split_counts": {
            "train": sum(1 for r in records if r["split"] == "train"),
            "val": sum(1 for r in records if r["split"] == "val"),
            "test": sum(1 for r in records if r["split"] == "test"),
        },
        "total_generation_time_s": total_time,
        "solve_time_stats_s": {
            "mean": float(np.mean(solve_times)),
            "min": float(np.min(solve_times)),
            "max": float(np.max(solve_times)),
        },
        "dT_max_stats_K": {
            "mean": float(np.mean(dT_maxes)),
            "min": float(np.min(dT_maxes)),
            "max": float(np.max(dT_maxes)),
        },
        "samples": records,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"数据生成完成，共耗时 {total_time:.2f} s")
    print(f"平均单样本 FEM 求解耗时: {np.mean(solve_times) * 1000:.3f} ms")
    print(f"最大温升 dT 范围: {np.min(dT_maxes):.2f} ~ {np.max(dT_maxes):.2f} K")
    print(f"划分统计: {metadata['split_counts']}")


if __name__ == "__main__":
    main()
