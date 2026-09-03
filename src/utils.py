"""公共工具函数：随机种子、配置加载、设备选择、数据集加载与标准化辅助。"""
from __future__ import annotations

import json
import os
import random

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
