"""
模型定义：
    1. MeshGraphNet 风格的 Encode-Process-Decode 消息传递网络（主模型）。
    2. 简洁基线模型：4 层 GCNConv 或 GraphSAGE。

两种模型均以节点/边特征为输入，输出每个节点的（标准化）温升预测值。
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import scatter


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layernorm: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self.layernorm = nn.LayerNorm(out_dim) if layernorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        out = self.net(x)
        if self.layernorm is not None:
            out = self.layernorm(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ProcessorLayer(nn.Module):
    """一轮消息传递：先更新边隐变量，再聚合到目标节点更新节点隐变量；均带残差连接。"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, layernorm=True, dropout=dropout)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, layernorm=True, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        src, dst = edge_index[0], edge_index[1]
        edge_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_out = edge_attr + self.edge_mlp(edge_in)

        agg = scatter(edge_out, dst, dim=0, dim_size=x.size(0), reduce="sum")
        node_in = torch.cat([x, agg], dim=-1)
        node_out = x + self.node_mlp(node_in)
        return node_out, edge_out


class MeshGraphNet(nn.Module):
    """Encode-Process-Decode 消息传递网络，用于预测节点温升 dT。"""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        n_message_passing_steps: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim, layernorm=True, dropout=0.0)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, layernorm=True, dropout=0.0)
        self.processor = nn.ModuleList(
            [ProcessorLayer(hidden_dim, dropout=dropout) for _ in range(n_message_passing_steps)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for layer in self.processor:
            x, e = layer(x, data.edge_index, e)
        out = self.decoder(x).squeeze(-1)
        return out


class BaselineGNN(nn.Module):
    """简洁基线：4 层 GCNConv 或 GraphSAGE（不使用边特征，仅利用图结构与节点特征）。"""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        conv_type: Literal["gcn", "sage"] = "sage",
    ):
        super().__init__()
        conv_cls = GCNConv if conv_type == "gcn" else SAGEConv
        dims = [node_in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([conv_cls(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv, act in zip(self.convs, self.acts):
            x = act(conv(x, edge_index))
            x = self.dropout(x)
        out = self.out_proj(x).squeeze(-1)
        return out


def build_model(name: str, node_in_dim: int, edge_in_dim: int, cfg: dict) -> nn.Module:
    if name == "meshgraphnet":
        mcfg = cfg["model"]
        return MeshGraphNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=mcfg["hidden_dim"],
            n_message_passing_steps=mcfg["n_message_passing_steps"],
            dropout=mcfg["dropout"],
        )
    elif name == "baseline":
        bcfg = cfg["baseline_model"]
        return BaselineGNN(
            node_in_dim=node_in_dim,
            hidden_dim=bcfg["hidden_dim"],
            n_layers=bcfg["n_layers"],
            dropout=bcfg["dropout"],
            conv_type=bcfg["type"],
        )
    else:
        raise ValueError(f"未知模型名称: {name}")
