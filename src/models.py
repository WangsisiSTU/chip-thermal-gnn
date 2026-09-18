"""
模型定义：
    1. MeshGraphNet 风格的 Encode-Process-Decode 消息传递网络（主模型）。
    2. MeshGraphNet + lightweight physics-state attention hybrid.
    3. 简洁基线模型：4 层 GCNConv 或 GraphSAGE。

两种模型均以节点/边特征为输入，输出每个节点的（标准化）温升预测值。
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import scatter


def apply_fem_operator_correction(pred_norm: torch.Tensor, data, mean: torch.Tensor,
                                  std: torch.Tensor, steps: int, omega: float) -> torch.Tensor:
    """Apply differentiable damped-Jacobi steps to ``(K+R)dT=f``."""
    if steps <= 0:
        return pred_norm
    required = ("fem_operator_index", "fem_operator_value", "fem_operator_diag",
                "fem_source_load", "fem_free_node")
    if any(getattr(data, key, None) is None for key in required):
        raise ValueError("operator correction requires the exact FEM sparse operator")
    if not 0.0 < omega <= 1.0:
        raise ValueError("operator correction omega must be in (0, 1]")
    row, col = data.fem_operator_index.long()
    dT = pred_norm * std + mean
    free = data.fem_free_node.bool()
    for _ in range(steps):
        residual = scatter(data.fem_operator_value * dT[col], row, dim=0,
                           dim_size=dT.numel(), reduce="sum") - data.fem_source_load
        candidate = dT - omega * residual / data.fem_operator_diag.clamp_min(1e-12)
        dT = torch.where(free, candidate, torch.zeros_like(candidate))
    return (dT - mean) / std


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

    def __init__(self, hidden_dim: int, dropout: float = 0.1, aggregation: str = "sum"):
        super().__init__()
        if aggregation not in ("sum", "mean"):
            raise ValueError("aggregation must be sum or mean")
        self.aggregation = aggregation
        self.edge_mlp = MLP(3 * hidden_dim, hidden_dim, hidden_dim, layernorm=True, dropout=dropout)
        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, layernorm=True, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        src, dst = edge_index[0], edge_index[1]
        edge_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        edge_out = edge_attr + self.edge_mlp(edge_in)

        agg = scatter(edge_out, dst, dim=0, dim_size=x.size(0), reduce=self.aggregation)
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
        aggregation: str = "sum",
    ):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim, layernorm=True, dropout=0.0)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, layernorm=True, dropout=0.0)
        self.processor = nn.ModuleList(
            [ProcessorLayer(hidden_dim, dropout=dropout, aggregation=aggregation) for _ in range(n_message_passing_steps)]
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


class PhysicsStateAttention(nn.Module):
    """Per-graph slice/attend/deslice on irregular meshes, without node padding.

    The learnable slices approximate physical states; attention is computed on
    the small state set rather than on all mesh nodes. Graph IDs keep samples
    independent even when their node counts differ within a PyG batch.
    """

    def __init__(self, hidden_dim: int, heads: int, slices: int, dropout: float = 0.0, mode: str = "full"):
        super().__init__()
        if heads < 1 or slices < 1 or hidden_dim % heads:
            raise ValueError("heads and slices must be positive and hidden_dim divisible by heads")
        if mode not in ("full", "slice_only", "adaptive", "adaptive_gumbel", "pool"):
            raise ValueError("unknown physics-state attention mode")
        self.mode = mode
        self.heads = heads
        self.slices = slices
        self.head_dim = hidden_dim // heads
        self.state_values = nn.Linear(hidden_dim, hidden_dim)
        self.slice_logits = nn.Linear(hidden_dim, heads * slices) if mode != "pool" else None
        self.temperature = nn.Parameter(torch.tensor(0.5))
        self.temperature_adjust = nn.Linear(hidden_dim, heads) if mode in ("adaptive", "adaptive_gumbel") else None
        if self.temperature_adjust is not None:
            nn.init.zeros_(self.temperature_adjust.weight)
            nn.init.zeros_(self.temperature_adjust.bias)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim) if mode in ("full", "adaptive", "adaptive_gumbel") else None
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph_id: torch.Tensor, node_weight: torch.Tensor | None = None):
        n_nodes, hidden_dim = x.shape
        n_graphs = int(graph_id.max().item()) + 1
        if self.mode == "pool":
            weights = x.new_full((n_nodes, self.heads, self.slices), 1.0 / self.slices)
        else:
            logits = self.slice_logits(x).view(n_nodes, self.heads, self.slices)
            if self.temperature_adjust is None:
                temperature = F.softplus(self.temperature) + 1e-4
            else:
                temperature = F.softplus(self.temperature + self.temperature_adjust(x).unsqueeze(-1)) + 1e-4
            if self.mode == "adaptive_gumbel" and self.training:
                uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
                logits = logits - torch.log(-torch.log(uniform))
            weights = F.softmax(logits / temperature, dim=-1)
        if node_weight is not None:
            if node_weight.numel() != n_nodes or torch.any(node_weight <= 0):
                raise ValueError("node_weight must be positive and have one value per node")
            weights_for_states = weights * node_weight.reshape(-1, 1, 1)
        else:
            weights_for_states = weights
        values = self.state_values(x).view(n_nodes, self.heads, self.head_dim)
        state_sum = scatter(
            weights_for_states.unsqueeze(-1) * values.unsqueeze(2), graph_id,
            dim=0, dim_size=n_graphs, reduce="sum",
        )
        state_mass = scatter(weights_for_states, graph_id, dim=0, dim_size=n_graphs, reduce="sum")
        states = state_sum / state_mass.clamp_min(1e-8).unsqueeze(-1)
        states = states.permute(0, 2, 1, 3).reshape(n_graphs, self.slices, hidden_dim)
        if self.qkv is not None:
            qkv = self.qkv(states).reshape(n_graphs, self.slices, 3, self.heads, self.head_dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            state_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
            state_out = state_out.transpose(1, 2).reshape(n_graphs, self.slices, hidden_dim)
        else:
            state_out = states
        state_out = state_out.view(n_graphs, self.slices, self.heads, self.head_dim).permute(0, 2, 1, 3)
        node_out = (weights.unsqueeze(-1) * state_out[graph_id]).sum(dim=2)
        return self.dropout(self.output(node_out.reshape(n_nodes, hidden_dim)))


class PhysicsStateBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, slices: int, dropout: float,
                 mode: str = "full", global_gate: bool = False):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsStateAttention(hidden_dim, heads, slices, dropout, mode=mode)
        self.global_gate = nn.Parameter(torch.tensor(-2.0)) if global_gate else None
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, graph_id: torch.Tensor, node_weight: torch.Tensor | None = None):
        global_update = self.attn(self.attn_norm(x), graph_id, node_weight)
        x = x + (torch.sigmoid(self.global_gate) * global_update if self.global_gate is not None else global_update)
        return x + self.ffn(self.ffn_norm(x))


class MGNTransolverHybrid(nn.Module):
    """Local mesh messages followed by global physics-state interactions."""

    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 128,
                 n_message_passing_steps: int = 3, attention_blocks: int = 2,
                 attention_heads: int = 4, attention_slices: int = 32, dropout: float = 0.03,
                 attention_mode: str = "full", aggregation: str = "sum", global_gate: bool = False,
                 output_mean: float = 0.0, output_std: float = 1.0,
                 operator_correction_steps: int = 0, operator_correction_omega: float = 0.65):
        super().__init__()
        if n_message_passing_steps < 1 or attention_blocks < 1:
            raise ValueError("hybrid requires at least one mesh and attention block")
        self.node_encoder = MLP(node_in_dim, hidden_dim, hidden_dim, dropout=0.0)
        self.edge_encoder = MLP(edge_in_dim, hidden_dim, hidden_dim, dropout=0.0)
        self.processor = nn.ModuleList([
            ProcessorLayer(hidden_dim, dropout, aggregation=aggregation) for _ in range(n_message_passing_steps)
        ])
        self.global_blocks = nn.ModuleList([
            PhysicsStateBlock(hidden_dim, attention_heads, attention_slices, dropout,
                              mode=attention_mode, global_gate=global_gate)
            for _ in range(attention_blocks)
        ])
        self.decoder = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                                     nn.GELU(), nn.Linear(hidden_dim, 1))
        if output_std <= 0:
            raise ValueError("output_std must be positive")
        self.register_buffer("output_mean", torch.tensor(float(output_mean)), persistent=False)
        self.register_buffer("output_std", torch.tensor(float(output_std)), persistent=False)
        self.operator_correction_steps = int(operator_correction_steps)
        self.operator_correction_omega = float(operator_correction_omega)

    def forward(self, data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for layer in self.processor:
            x, e = layer(x, data.edge_index, e)
        graph_id = getattr(data, "batch", None)
        if graph_id is None:
            graph_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        node_weight = getattr(data, "node_volume", None)
        for block in self.global_blocks:
            x = block(x, graph_id, node_weight)
        prediction = self.decoder(x).squeeze(-1)
        return apply_fem_operator_correction(
            prediction, data, self.output_mean, self.output_std,
            self.operator_correction_steps, self.operator_correction_omega,
        )


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
            aggregation=mcfg.get("aggregation", "sum"),
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
    elif name in ("mgn_transolver", "mgn_global_pool", "mgn_transolver_slice_only",
                  "mgn_transolver_adaptive", "mgn_transolver_adaptive_gumbel"):
        hcfg = cfg["hybrid_model"]
        modes = {"mgn_transolver": "full", "mgn_global_pool": "pool",
                 "mgn_transolver_slice_only": "slice_only", "mgn_transolver_adaptive": "adaptive",
                 "mgn_transolver_adaptive_gumbel": "adaptive_gumbel"}
        return MGNTransolverHybrid(
            node_in_dim=node_in_dim, edge_in_dim=edge_in_dim,
            hidden_dim=hcfg["hidden_dim"],
            n_message_passing_steps=hcfg["n_message_passing_steps"],
            attention_blocks=hcfg["attention_blocks"],
            attention_heads=hcfg["attention_heads"],
            attention_slices=hcfg["attention_slices"],
            dropout=hcfg["dropout"],
            attention_mode=modes[name],
            aggregation=hcfg.get("aggregation", "sum"),
            global_gate=hcfg.get("global_gate", False),
            output_mean=cfg.get("output_normalization", {}).get("mean", 0.0),
            output_std=cfg.get("output_normalization", {}).get("std", 1.0),
            operator_correction_steps=hcfg.get("operator_correction_steps", 0),
            operator_correction_omega=hcfg.get("operator_correction_omega", 0.65),
        )
    else:
        raise ValueError(f"未知模型名称: {name}")
