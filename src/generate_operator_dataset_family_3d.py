"""Generate and graph-build a paired family of 3D FEM operator datasets."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

# scipy/scikit-fem and PyTorch may load separate OpenMP runtimes on Windows.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

from generate_dataset_3d import generate_dataset_3d, load_config
from graph_dataset_3d import build_dataset_3d


def main():
    parser = argparse.ArgumentParser(description="Generate paired FEM-operator mesh-family datasets")
    parser.add_argument("--data_config", default="configs/data_config_3d_operator_learning.yaml")
    parser.add_argument("--mesh_family", default="configs/mesh_family_3d_operator.yaml")
    parser.add_argument("--raw_prefix", default="data/raw_3d_operator_")
    parser.add_argument("--processed_prefix", default="data/processed_3d_operator_")
    parser.add_argument("--meshes", nargs="*", default=None, help="optional subset of manifest names")
    args = parser.parse_args()

    base = load_config(args.data_config)
    family = yaml.safe_load(Path(args.mesh_family).read_text(encoding="utf-8"))["meshes"]
    selected = list(family) if args.meshes is None else args.meshes
    unknown = set(selected) - set(family)
    if unknown:
        raise ValueError(f"unknown mesh variants: {sorted(unknown)}")
    for name in selected:
        spec = family[name]
        config = copy.deepcopy(base)
        config["geometry"].update(spec["geometry"])
        raw_dir = f"{args.raw_prefix}{name}"
        processed_dir = f"{args.processed_prefix}{name}"
        raw_meta = generate_dataset_3d(config, raw_dir)
        processed_meta = build_dataset_3d(raw_dir, processed_dir)
        print(name, spec.get("role", "unspecified"), raw_meta["split_counts"],
              processed_meta["n_nodes"], processed_meta["n_tets"])


if __name__ == "__main__":
    main()
