"""Adapters for the released IC-ThermBench S3/S4/S5 HDF5 tensors.

The benchmark stores MATLAB v7.3 files as HDF5.  This module intentionally keeps
the official axis order and split semantics visible instead of silently reshaping
or shuffling samples.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCOPE_CHANNELS: dict[str, tuple[str, ...]] = {
    "level3": ("chiplet_power", "grid_x", "grid_y", "local_thermal_k"),
    "level4": (
        "chiplet_power",
        "grid_x",
        "grid_y",
        "local_thermal_k",
        "ambient_K",
        "h_w_m2k",
        "r_convec_k_per_w",
    ),
    "level5": (
        "chiplet_power",
        "grid_x",
        "grid_y",
        "local_thermal_k",
        "ambient_K",
        "h_w_m2k",
        "r_convec_k_per_w",
    ),
}

OFFICIAL_METRICS = {
    "rmse": "per-sample/channel RMSE, then averaged",
    "mean_absolute_error": "per-sample/channel MAE, then averaged",
    "r2": "pooled R2 over all test pixels, averaged across channels",
    "r2_per_sample": "per-sample R2, retained as a diagnostic",
    "max_absolute_error": "worst pixel error per sample/channel, then averaged",
    "max_temperature_error": "absolute predicted-vs-true peak temperature error",
    "topk50_temperature_difference": "MAE at the 50 hottest true pixels",
}


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("IC-ThermBench loading requires h5py>=3.10") from exc
    return h5py


def _find_single_mat(folder: Path, keyword: str) -> Path:
    matches = sorted(p for p in folder.glob("*.mat") if keyword in p.name.lower())
    if len(matches) != 1:
        raise ValueError(
            f"expected one *{keyword}*.mat file in {folder}, found {len(matches)}"
        )
    return matches[0]


def find_mat_pair(folder: str | Path) -> tuple[Path, Path]:
    """Return the unique input/output MATLAB-HDF5 pair in ``folder``."""

    folder = Path(folder)
    return _find_single_mat(folder, "input"), _find_single_mat(folder, "output")


def inspect_mat_pair(folder: str | Path, scope: str) -> dict[str, Any]:
    """Validate shapes/dtypes without materializing the multi-GB tensors."""

    if scope not in SCOPE_CHANNELS:
        raise ValueError(f"unsupported scope {scope!r}; choose from {sorted(SCOPE_CHANNELS)}")
    h5py = _require_h5py()
    input_path, output_path = find_mat_pair(folder)
    with h5py.File(input_path, "r") as input_file, h5py.File(output_path, "r") as output_file:
        if "data" not in input_file or "data" not in output_file:
            raise KeyError("both input.mat and output.mat must contain a 'data' dataset")
        input_data = input_file["data"]
        output_data = output_file["data"]
        if input_data.ndim != 5 or output_data.ndim != 4:
            raise ValueError(
                "expected raw axes input=[B,P,Z,Y,X], output=[B,Z,Y,X], got "
                f"{input_data.shape} and {output_data.shape}"
            )
        if input_data.shape[0] != output_data.shape[0]:
            raise ValueError("input/output sample counts differ")
        if tuple(input_data.shape[2:]) != tuple(output_data.shape[1:]):
            raise ValueError("input/output spatial shapes differ")
        expected_channels = len(SCOPE_CHANNELS[scope])
        if input_data.shape[1] != expected_channels:
            raise ValueError(
                f"{scope} requires {expected_channels} channels, got {input_data.shape[1]}"
            )
        if input_data.dtype != np.dtype("float32") or output_data.dtype != np.dtype("float32"):
            raise TypeError("released IC-ThermBench tensors must be float32")
        return {
            "scope": scope,
            "channels": list(SCOPE_CHANNELS[scope]),
            "samples": int(input_data.shape[0]),
            "raw_input_shape": list(input_data.shape),
            "raw_output_shape": list(output_data.shape),
            "adapter_input_shape": [
                int(input_data.shape[0]),
                int(input_data.shape[4]),
                int(input_data.shape[3]),
                int(input_data.shape[2]),
                int(input_data.shape[1]),
            ],
            "adapter_output_shape": [
                int(output_data.shape[0]),
                int(output_data.shape[3]),
                int(output_data.shape[2]),
                int(output_data.shape[1]),
            ],
        }


class ICThermBenchH5Dataset:
    """Lazy sample-level reader that avoids loading a 4.6 GB archive into RAM."""

    def __init__(self, folder: str | Path, scope: str, indices: Sequence[int] | None = None):
        self.folder = Path(folder)
        self.scope = scope
        self.contract = inspect_mat_pair(self.folder, scope)
        self.input_path, self.output_path = find_mat_pair(self.folder)
        total = int(self.contract["samples"])
        self.indices = np.arange(total, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= total):
            raise IndexError("dataset indices must be a one-dimensional in-range sequence")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        h5py = _require_h5py()
        source_index = int(self.indices[item])
        with h5py.File(self.input_path, "r") as input_file:
            # [P,Z,Y,X] -> [X,Y,Z,P]
            features = np.asarray(input_file["data"][source_index], dtype=np.float32).transpose(3, 2, 1, 0)
        with h5py.File(self.output_path, "r") as output_file:
            # [Z,Y,X] -> [X,Y,Z]
            target = np.asarray(output_file["data"][source_index], dtype=np.float32).transpose(2, 1, 0)
        return features, target


def official_s3_s4_indices(
    total: int, train_ratio: float = 0.8, num_trajectories: int = -1
) -> dict[str, np.ndarray]:
    """Reproduce the official non-shuffled split: 72%/8%/20% by default."""

    if total < 3 or not 0.0 < train_ratio < 1.0:
        raise ValueError("total must be >=3 and train_ratio must lie in (0,1)")
    trainval_total = max(2, min(int(total * train_ratio), total - 1))
    used = trainval_total
    if num_trajectories is not None and num_trajectories > 0:
        used = min(int(num_trajectories), trainval_total)
    if used < 2:
        raise ValueError("at least two train+validation samples are required")
    n_train = max(1, min(int(used * 0.9), used - 1))
    return {
        "train": np.arange(0, n_train, dtype=np.int64),
        "val": np.arange(n_train, used, dtype=np.int64),
        "test": np.arange(trainval_total, total, dtype=np.int64),
    }


def load_s5_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise TypeError("S5 manifest must be a JSON list of objects")
    return records


def s5_fewshot_indices(
    manifest: Sequence[Mapping[str, Any]], shots: int, pool_per_case: int = 500
) -> dict[str, np.ndarray]:
    """Reproduce S5 adaptation/holdout grouping without hard-coding case labels."""

    if not 0 <= int(shots) <= pool_per_case:
        raise ValueError(f"shots must be in [0,{pool_per_case}]")
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(manifest):
        if "case" not in record:
            raise KeyError(f"manifest record {index} has no 'case' field")
        grouped.setdefault(str(record["case"]), []).append(index)
    if len(grouped) != 5:
        raise ValueError(f"S5 must contain five cases, got {len(grouped)}")
    adaptation: list[int] = []
    holdout: list[int] = []
    for case, indices in grouped.items():
        if len(indices) != 2 * pool_per_case:
            raise ValueError(
                f"case {case!r} has {len(indices)} rows; expected {2 * pool_per_case}"
            )
        adaptation.extend(indices[: int(shots)])
        holdout.extend(indices[pool_per_case:])
    return {
        "adaptation": np.asarray(adaptation, dtype=np.int64),
        "holdout": np.asarray(holdout, dtype=np.int64),
        "zero_shot": np.arange(len(manifest), dtype=np.int64),
    }


def load_official_metrics(ic_root: str | Path) -> Callable[..., dict[str, float]]:
    """Load the benchmark's one canonical metric function; do not fork its formulas."""

    metrics_path = Path(ic_root) / "utils" / "metrics.py"
    spec = importlib.util.spec_from_file_location("ic_thermbench_official_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official metrics from {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compute_additional_test_metrics

