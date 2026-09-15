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
    }


def validate_data_contract(checkpoint: dict, metadata: dict) -> None:
    """Reject evaluation data whose feature semantics differ from training data."""
    trained = checkpoint.get("data_contract")
    if trained is None:
        warnings.warn(
            "Checkpoint has no data contract; only input dimensions can be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        expected = (checkpoint["node_in_dim"], checkpoint["edge_in_dim"])
        actual = (metadata["node_feature_dim"], metadata["edge_feature_dim"])
        if actual != expected:
            raise ValueError(f"Checkpoint expects node/edge dimensions {expected}, dataset has {actual}")
        return
    actual = build_data_contract(metadata)
    mismatches = [key for key in trained if trained[key] != actual.get(key)]
    if mismatches:
        raise ValueError(f"Dataset is incompatible with checkpoint; mismatched contract fields: {mismatches}")
