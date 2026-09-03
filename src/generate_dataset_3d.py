"""Generate a three-dimensional tetrahedral FEM dataset for chip thermal GNNs.

The 2D generator remains unchanged.  This entry point writes an independent
dataset directory, normally ``data/raw_3d``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fem_solver_3d import CaseParams3D, GeometryConfig3D, build_mesh_3d, solve_case_3d


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _uniform(rng: np.random.Generator, interval: dict) -> float:
    return float(rng.uniform(interval["low"], interval["high"]))


def sample_id_case_3d(rng: np.random.Generator, cfg: dict) -> CaseParams3D:
    mat, hs, bc = cfg["materials"], cfg["heat_source"], cfg["boundary"]
    return CaseParams3D(
        k_die=_uniform(rng, mat["k_die"]),
        k_tim=_uniform(rng, mat["k_tim"]),
        k_cu=_uniform(rng, mat["k_cu"]),
        k_sub=_uniform(rng, mat["k_sub"]),
        q_base=_uniform(rng, hs["q_base"]),
        q_hot=_uniform(rng, hs["q_hot"]),
        hotspot_center_x_frac=_uniform(rng, hs["hotspot_center_x_frac"]),
        hotspot_center_z_frac=_uniform(rng, hs["hotspot_center_z_frac"]),
        hotspot_width_x_frac=_uniform(rng, hs["hotspot_width_x_frac"]),
        hotspot_width_z_frac=_uniform(rng, hs["hotspot_width_z_frac"]),
        t_ambient=_uniform(rng, bc["t_ambient"]),
        h_top=_uniform(rng, bc["h_top"]),
        regime="id",
    )


def sample_ood_case_3d(rng: np.random.Generator, cfg: dict, kind: str) -> CaseParams3D:
    case = sample_id_case_3d(rng, cfg)
    ood = cfg["ood"]
    if kind == "high_power":
        case.q_hot = _uniform(rng, ood["q_hot"])
    elif kind == "poor_tim":
        case.k_tim = _uniform(rng, ood["k_tim"])
    elif kind == "poor_cooling":
        case.h_top = _uniform(rng, ood["h_top"])
    else:
        raise ValueError(f"Unknown OOD case type: {kind}")
    case.regime = f"ood_{kind}"
    return case


def build_case_list_3d(cfg: dict, rng: np.random.Generator):
    split = cfg["split"]
    n_ood = cfg["ood"]["n_ood_test_samples"]
    n_test_id = split["n_test"] - n_ood
    if n_test_id < 0:
        raise ValueError("n_ood_test_samples cannot exceed split.n_test")

    cases = []
    for split_name, count in (("train", split["n_train"]), ("val", split["n_val"]), ("test", n_test_id)):
        cases.extend((split_name, sample_id_case_3d(rng, cfg)) for _ in range(count))

    ood_kinds = ["high_power", "poor_tim", "poor_cooling"]
    counts = [n_ood // 3 + (i < n_ood % 3) for i in range(len(ood_kinds))]
    for kind, count in zip(ood_kinds, counts):
        cases.extend(("test", sample_ood_case_3d(rng, cfg, kind)) for _ in range(count))
    return cases


def generate_dataset_3d(cfg: dict, out_dir: str) -> dict:
    """Generate raw 3D FEM samples and return their metadata."""
    rng = np.random.default_rng(cfg["seed"])
    geom = GeometryConfig3D.from_dict(cfg["geometry"])
    mesh, basis = build_mesh_3d(geom)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "mesh.npz"), points=mesh.p, tets=mesh.t)

    records, solve_times, dT_maxes = [], [], []
    cases = build_case_list_3d(cfg, rng)
    started = time.perf_counter()
    for index, (split_name, case) in enumerate(tqdm(cases, desc="3D tetrahedral FEM")):
        result = solve_case_3d(mesh, basis, geom, case)
        filename = f"sample_{index:04d}.npz"
        np.savez(
            os.path.join(out_dir, filename),
            T=result.T.astype(np.float64),
            material_id=result.material_id.astype(np.int64),
            q_node=result.q_node.astype(np.float64),
            is_top=result.is_top,
            is_bottom=result.is_bottom,
            h_node=result.h_node.astype(np.float64),
        )
        dT_max = float(result.T.max() - case.t_ambient)
        solve_times.append(result.solve_time_s)
        dT_maxes.append(dT_max)
        records.append(
            {
                "file": filename,
                "split": split_name,
                "regime": case.regime,
                "case": case.to_dict(),
                "solve_time_s": result.solve_time_s,
                "T_min": float(result.T.min()),
                "T_max": float(result.T.max()),
                "dT_max": dT_max,
            }
        )

    metadata = {
        "dimension": 3,
        "element_type": "tetrahedron_p1",
        "seed": cfg["seed"],
        "n_nodes": int(mesh.p.shape[1]),
        "n_tets": int(mesh.t.shape[1]),
        "geometry": cfg["geometry"],
        "materials": cfg["materials"],
        "heat_source": cfg["heat_source"],
        "boundary": cfg["boundary"],
        "ood": cfg["ood"],
        "split_counts": {name: sum(r["split"] == name for r in records) for name in ("train", "val", "test")},
        "total_generation_time_s": time.perf_counter() - started,
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
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate 3D tetrahedral chip thermal FEM data.")
    parser.add_argument("--config", default="configs/data_config_3d.yaml")
    parser.add_argument("--out_dir", default="data/raw_3d")
    args = parser.parse_args()

    metadata = generate_dataset_3d(load_config(args.config), args.out_dir)
    print(
        f"3D dataset complete: {metadata['n_nodes']} nodes, {metadata['n_tets']} tetrahedra, "
        f"splits={metadata['split_counts']}"
    )
    print(f"mean FEM solve time: {metadata['solve_time_stats_s']['mean'] * 1000:.3f} ms")


if __name__ == "__main__":
    main()
