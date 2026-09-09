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

SAMPLING_STREAMS_3D = (
    "train_id", "val_id", "test_id",
    "covered_poor_cooling_train", "covered_poor_cooling_val", "covered_poor_cooling_test",
    "ood_high_power", "ood_poor_tim", "ood_poor_cooling",
)

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


def _sampling_rngs_3d(seed: int) -> dict[str, np.random.Generator]:
    """Use one stable random stream per split/regime.

    Changing a training-split count must never change validation or test cases.
    """
    return {
        name: np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))
        for index, name in enumerate(SAMPLING_STREAMS_3D)
    }


def _coverage_counts_3d(cfg: dict) -> dict[str, int]:
    coverage = cfg.get("coverage", {}).get("poor_cooling", {})
    return {split: int(coverage.get(split, 0)) for split in ("train", "val", "test")}


def _ood_counts_3d(cfg: dict) -> dict[str, int]:
    ood = cfg["ood"]
    kinds = ("high_power", "poor_tim", "poor_cooling")
    explicit = ood.get("test_counts")
    if explicit is not None:
        unknown = set(explicit) - set(kinds)
        if unknown:
            raise ValueError(f"Unknown OOD test-count regimes: {sorted(unknown)}")
        counts = {kind: int(explicit.get(kind, 0)) for kind in kinds}
        if any(count < 0 for count in counts.values()):
            raise ValueError("OOD test counts must be non-negative")
        return counts
    n_ood = int(ood["n_ood_test_samples"])
    if n_ood < 0:
        raise ValueError("n_ood_test_samples must be non-negative")
    return {kind: n_ood // len(kinds) + (index < n_ood % len(kinds)) for index, kind in enumerate(kinds)}


def sample_covered_poor_cooling_case_3d(rng: np.random.Generator, cfg: dict) -> CaseParams3D:
    """Sample a low-convection case intentionally included in model coverage."""
    coverage = cfg.get("coverage", {}).get("poor_cooling", {})
    if "h_top" not in coverage:
        raise ValueError("coverage.poor_cooling.h_top is required when its count is positive")
    case = sample_id_case_3d(rng, cfg)
    case.h_top = _uniform(rng, coverage["h_top"])
    case.regime = "covered_poor_cooling"
    return case


def _sampling_rngs_3d(seed: int) -> dict[str, np.random.Generator]:
    """Use one stable random stream per split/regime.

    Changing a training-split count must never change validation or test cases.
    """
    return {
        name: np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))
        for index, name in enumerate(SAMPLING_STREAMS_3D)
    }


def build_case_list_3d(cfg: dict):
    split = cfg["split"]
    coverage_counts = _coverage_counts_3d(cfg)
    ood_counts = _ood_counts_3d(cfg)
    n_id = {
        "train": int(split["n_train"]) - coverage_counts["train"],
        "val": int(split["n_val"]) - coverage_counts["val"],
        "test": int(split["n_test"]) - coverage_counts["test"] - sum(ood_counts.values()),
    }
    if any(count < 0 for count in n_id.values()):
        raise ValueError("Coverage and OOD counts cannot exceed their split sizes")

    rngs = _sampling_rngs_3d(cfg["seed"])
    cases = []
    for split_name in ("train", "val", "test"):
        rng = rngs[f"{split_name}_id"]
        cases.extend((split_name, sample_id_case_3d(rng, cfg)) for _ in range(n_id[split_name]))

        coverage_count = coverage_counts[split_name]
        if coverage_count:
            coverage_rng = rngs[f"covered_poor_cooling_{split_name}"]
            cases.extend(
                (split_name, sample_covered_poor_cooling_case_3d(coverage_rng, cfg))
                for _ in range(coverage_count)
            )

    for kind, count in ood_counts.items():
        rng = rngs[f"ood_{kind}"]
        cases.extend(("test", sample_ood_case_3d(rng, cfg, kind)) for _ in range(count))
    return cases


def generate_dataset_3d(cfg: dict, out_dir: str) -> dict:
    """Generate raw 3D FEM samples and return their metadata."""
    geom = GeometryConfig3D.from_dict(cfg["geometry"])
    mesh, basis = build_mesh_3d(geom)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "mesh.npz"), points=mesh.p, tets=mesh.t)

    records, solve_times, dT_maxes = [], [], []
    cases = build_case_list_3d(cfg)
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
        "sampling_streams": {"strategy": "independent_seedsequence", "names": list(SAMPLING_STREAMS_3D)},
        "n_nodes": int(mesh.p.shape[1]),
        "n_tets": int(mesh.t.shape[1]),
        "geometry": cfg["geometry"],
        "materials": cfg["materials"],
        "heat_source": cfg["heat_source"],
        "boundary": cfg["boundary"],
        "ood": cfg["ood"],
        "coverage": cfg.get("coverage", {}),
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
