"""Schema adapter and split-quality checks for Multi-topology-Dataset."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


FIELD_SCHEMA: dict[str, tuple[tuple[int, ...] | str, str]] = {
    "temperature": ((256, 256), "float32"),
    "power_map": ((256, 256), "float32"),
    "instance_map": ((256, 256), "uint16"),
    "ceramic_mask": ((256, 256), "uint8"),
    "baseplate_mask": ((256, 256), "uint8"),
    "copper_mask": ((256, 256), "uint8"),
    "h_normalized": ((1,), "float32"),
    "max_temps": ("N", "float32"),
    "row_ids": ("N", "uint16"),
}

DENSE_FEATURE_CHANNELS = (
    "power_map",
    "chip_mask",
    "ceramic_mask",
    "baseplate_mask",
    "copper_mask",
    "h_normalized",
)

PATH_RE = re.compile(
    r"^(?P<split>TrainData|ValData|TestData)/(?P<topology>[^/]+)/"
    r"(?P<mask_id>\d{3})_Pigbt(?P<Pigbt>\d+)_Pfwd(?P<Pfwd>\d+)_h(?P<h_value>\d+)/data\.h5$"
)


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Multi-topology loading requires h5py>=3.10") from exc
    return h5py


def parse_sample_path(path: str | Path) -> dict[str, Any]:
    normalized = str(path).replace("\\", "/")
    match = PATH_RE.match(normalized)
    if match is None:
        raise ValueError(f"path does not match the released layout: {normalized}")
    parsed: dict[str, Any] = match.groupdict()
    for key in ("Pigbt", "Pfwd", "h_value"):
        parsed[key] = int(parsed[key])
    return parsed


def load_sample(path: str | Path, validate: bool = True) -> dict[str, Any]:
    """Load one sample while preserving categorical instance ids separately."""

    h5py = _require_h5py()
    path = Path(path)
    with h5py.File(path, "r") as source:
        missing = sorted(set(FIELD_SCHEMA) - set(source.keys()))
        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")
        fields = {name: source[name][()] for name in FIELD_SCHEMA}
        attrs = {name: source.attrs[name] for name in source.attrs}
    if validate:
        validate_sample(fields, attrs)
    chip_mask = (fields["instance_map"] > 0).astype(np.float32)
    h_plane = np.full((256, 256), float(fields["h_normalized"][0]), dtype=np.float32)
    features = np.stack(
        [
            fields["power_map"],
            chip_mask,
            fields["ceramic_mask"],
            fields["baseplate_mask"],
            fields["copper_mask"],
            h_plane,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    return {
        "features": features,
        "feature_channels": DENSE_FEATURE_CHANNELS,
        "target": fields["temperature"],
        "instance_map": fields["instance_map"],
        "row_ids": fields["row_ids"],
        "chip_max_temperatures": fields["max_temps"],
        "attributes": attrs,
        "raw_fields": fields,
    }


def validate_sample(fields: Mapping[str, np.ndarray], attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate schema, finiteness, masks, chip ids, and cooling consistency."""

    errors: list[str] = []
    for name, (shape, dtype_name) in FIELD_SCHEMA.items():
        if name not in fields:
            errors.append(f"missing field {name}")
            continue
        value = np.asarray(fields[name])
        if shape != "N" and tuple(value.shape) != shape:
            errors.append(f"{name} shape={value.shape}, expected={shape}")
        if value.dtype != np.dtype(dtype_name):
            errors.append(f"{name} dtype={value.dtype}, expected={dtype_name}")
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            errors.append(f"{name} contains non-finite values")
    if "row_ids" in fields and "max_temps" in fields:
        if len(fields["row_ids"]) != len(fields["max_temps"]):
            errors.append("row_ids and max_temps lengths differ")
    for name in ("ceramic_mask", "baseplate_mask", "copper_mask"):
        if name in fields and not set(np.unique(fields[name])).issubset({0, 1}):
            errors.append(f"{name} is not binary")
    if "instance_map" in fields and "row_ids" in fields:
        observed = set(np.unique(fields["instance_map"]).tolist()) - {0}
        declared = set(np.asarray(fields["row_ids"]).tolist())
        if observed != declared:
            errors.append(f"instance ids {sorted(observed)} != row_ids {sorted(declared)}")
    h_data = float(np.asarray(fields.get("h_normalized", [np.nan]))[0])
    if "h_value" in attrs and not np.isclose(h_data, float(attrs["h_value"]) / 10000.0):
        errors.append("h_normalized does not equal h_value/10000")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "shape": [256, 256],
        "instances": int(len(fields["row_ids"])),
        "temperature_min": float(np.min(fields["temperature"])),
        "temperature_max": float(np.max(fields["temperature"])),
        "h_normalized": h_data,
    }


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_representative_rows(
    rows: Iterable[Mapping[str, str]], per_topology: int = 1, split: str = "TestData"
) -> list[dict[str, str]]:
    """Select deterministic, small samples without downloading the full 4.67 GB tree."""

    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted((dict(r) for r in rows if r["split"] == split), key=lambda r: r["path"]):
        if len(selected[row["topology"]]) < per_topology:
            selected[row["topology"]].append(row)
    return [row for topology in sorted(selected) for row in selected[topology]]


def validate_stratified_split(
    rows: list[Mapping[str, str]], summary_path: str | Path | None = None
) -> dict[str, Any]:
    """Audit released stratification and expose its non-layout-disjoint limitation."""

    required = {
        "split", "topology", "mask_id", "Pigbt", "Pfwd", "h_value",
        "h_normalized", "n_instances", "bytes", "path",
    }
    if not rows:
        raise ValueError("manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"manifest is missing columns: {sorted(missing)}")

    duplicate_paths = [path for path, count in Counter(r["path"] for r in rows).items() if count > 1]
    path_mismatches: list[str] = []
    h_mismatches: list[str] = []
    counts = Counter((r["topology"], r["split"]) for r in rows)
    by_topology: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        by_topology[row["topology"]].append(row)
        try:
            parsed = parse_sample_path(row["path"])
        except ValueError:
            path_mismatches.append(row["path"])
            continue
        for key in ("split", "topology", "mask_id"):
            if str(parsed[key]) != str(row[key]):
                path_mismatches.append(row["path"])
                break
        if not np.isclose(float(row["h_normalized"]), int(row["h_value"]) / 10000.0):
            h_mismatches.append(row["path"])

    split_names = ("TrainData", "ValData", "TestData")
    topology_report: dict[str, Any] = {}
    all_layout_disjoint = True
    for topology, group in sorted(by_topology.items()):
        layout_sets = {
            split: {r["mask_id"] for r in group if r["split"] == split}
            for split in split_names
        }
        overlap = {
            "train_val": len(layout_sets["TrainData"] & layout_sets["ValData"]),
            "train_test": len(layout_sets["TrainData"] & layout_sets["TestData"]),
            "val_test": len(layout_sets["ValData"] & layout_sets["TestData"]),
        }
        all_layout_disjoint &= not any(overlap.values())
        conditions = {}
        for key in ("Pigbt", "Pfwd", "h_value"):
            conditions[key] = {
                split: sorted({int(r[key]) for r in group if r["split"] == split})
                for split in split_names
            }
        topology_report[topology] = {
            "counts": {split: counts[(topology, split)] for split in split_names},
            "layout_overlap": overlap,
            "condition_values": conditions,
        }

    summary_errors: list[str] = []
    if summary_path is not None:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        if int(summary["total_samples"]) != len(rows):
            summary_errors.append("total_samples differs from manifest")
        split_counts = Counter(r["split"] for r in rows)
        for split, expected in summary["splits"].items():
            if split_counts[split] != int(expected):
                summary_errors.append(f"split count differs for {split}")
        topology_counts = Counter(r["topology"] for r in rows)
        for topology, expected in summary["topologies"].items():
            if topology_counts[topology] != int(expected):
                summary_errors.append(f"topology count differs for {topology}")

    fatal_errors = duplicate_paths + path_mismatches + h_mismatches + summary_errors
    return {
        "valid": not fatal_errors,
        "rows": len(rows),
        "duplicate_paths": len(duplicate_paths),
        "path_label_mismatches": len(path_mismatches),
        "h_normalization_mismatches": len(h_mismatches),
        "summary_errors": summary_errors,
        "topology_split": topology_report,
        "layout_disjoint": all_layout_disjoint,
        "warning": (
            "Official split is operating-condition stratified but reuses every layout id "
            "across train/validation/test; it is not a structural-OOD split."
            if not all_layout_disjoint else None
        ),
    }

