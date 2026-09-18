"""公共工具函数：随机种子、配置加载、设备选择、数据集加载与标准化辅助。"""
from __future__ import annotations

import json
import os
import random
import warnings

import numpy as np
import torch
import yaml

from models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_split(data_dir: str, split: str):
    path = os.path.join(data_dir, f"{split}.pt")
    return torch.load(path, weights_only=False)


def load_processed_metadata(data_dir: str) -> dict:
    return load_json(os.path.join(data_dir, "metadata.json"))


def build_data_contract(metadata: dict) -> dict:
    """Return the feature and normalization contract required by a checkpoint."""
    required = (
        "node_feature_dim",
        "edge_feature_dim",
        "node_feature_names",
        "edge_feature_names",
        "norm_consts",
        "dT_train_mean",
        "dT_train_std",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Processed metadata is missing contract fields: {missing}")
    return {key: metadata[key] for key in required} | {
        "dimension": metadata.get("dimension"),
        "node_volume_weighting": metadata.get("node_volume_weighting", False),
        "heat_source_projection": metadata.get("heat_source_projection", "point_sampled"),
        "tetra_gradient_geometry": metadata.get("tetra_gradient_geometry", "none"),
        "discrete_operator": metadata.get("discrete_operator", "none"),
        "material_interface_flux": metadata.get("material_interface_flux", "none"),
    }


def attach_data_contract(checkpoint: dict, metadata: dict) -> dict:
    """Return a checkpoint with a verified contract attached.

    Migration is allowed only when dimensions and stored output normalization
    match the supplied training metadata.  This prevents a current dataset
    contract from being stamped onto an unrelated legacy checkpoint.
    """
    expected = (checkpoint.get("node_in_dim"), checkpoint.get("edge_in_dim"))
    actual = (metadata.get("node_feature_dim"), metadata.get("edge_feature_dim"))
    if expected != actual:
        raise ValueError(f"Checkpoint expects node/edge dimensions {expected}, dataset has {actual}")
    for checkpoint_key, metadata_key in (("dT_mean", "dT_train_mean"),
                                         ("dT_std", "dT_train_std")):
        if checkpoint_key not in checkpoint:
            raise ValueError(f"Legacy checkpoint is missing {checkpoint_key}")
        if not np.isclose(float(checkpoint[checkpoint_key]), float(metadata[metadata_key]),
                          rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"Checkpoint {checkpoint_key} does not match training metadata {metadata_key}")
    migrated = dict(checkpoint)
    migrated["data_contract"] = build_data_contract(metadata)
    migrated["data_contract_migration"] = {
        "version": 1,
        "method": "verified_dimensions_and_output_normalization",
    }
    return migrated


def validate_data_contract(checkpoint: dict, metadata: dict, *, allow_output_stats_shift: bool = False) -> None:
    """Reject evaluation data whose feature semantics differ from training data."""
    trained = checkpoint.get("data_contract")
    if trained is None:
        expected = (checkpoint["node_in_dim"], checkpoint["edge_in_dim"])
        actual = (metadata["node_feature_dim"], metadata["edge_feature_dim"])
        if actual != expected:
            raise ValueError(f"Checkpoint expects node/edge dimensions {expected}, dataset has {actual}")
        warnings.warn(
            "Checkpoint has no data contract; only input dimensions can be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    actual = build_data_contract(metadata)
    allowed = {"dT_train_mean", "dT_train_std"} if allow_output_stats_shift else set()
    mismatches = [key for key in trained if key not in allowed and trained[key] != actual.get(key)]
    # Older checkpoints predate these fields and must not silently consume
    # FEM-projected source features with different physical semantics.
    for key, legacy in (("heat_source_projection", "point_sampled"), ("tetra_gradient_geometry", "none"),
                        ("discrete_operator", "none"), ("material_interface_flux", "none")):
        if trained.get(key, legacy) != actual[key] and key not in mismatches:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"Dataset is incompatible with checkpoint; mismatched contract fields: {mismatches}")
