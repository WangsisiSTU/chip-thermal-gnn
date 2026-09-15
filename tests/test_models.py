"""模型前向传播输出形状与可训练性测试。"""
import torch
from torch_geometric.data import Batch, Data

from models import BaselineGNN, MGNTransolverHybrid, MeshGraphNet


def make_toy_graph(n_nodes=20, node_dim=11, edge_dim=4):
    torch.manual_seed(0)
    x = torch.randn(n_nodes, node_dim)
    # 构造一个环状图（保证每个节点都有边），双向
    src = torch.arange(n_nodes)
    dst = (src + 1) % n_nodes
    edge_index = torch.cat(
        [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
    )
    edge_attr = torch.randn(edge_index.shape[1], edge_dim)
    y = torch.randn(n_nodes)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def test_meshgraphnet_forward_and_backward():
    data = make_toy_graph()
    model = MeshGraphNet(node_in_dim=11, edge_in_dim=4, hidden_dim=32, n_message_passing_steps=3, dropout=0.0)
    out = model(data)
    assert out.shape == (data.x.shape[0],)
    assert torch.all(torch.isfinite(out))

    loss = torch.nn.functional.mse_loss(out, data.y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "反向传播未产生梯度"


def test_baseline_forward():
    data = make_toy_graph()
    for conv_type in ["sage", "gcn"]:
        model = BaselineGNN(node_in_dim=11, hidden_dim=32, n_layers=4, dropout=0.0, conv_type=conv_type)
        out = model(data)
        assert out.shape == (data.x.shape[0],)
        assert torch.all(torch.isfinite(out))


def test_hybrid_keeps_variable_mesh_graphs_independent():
    small = make_toy_graph(n_nodes=13)
    large = make_toy_graph(n_nodes=21)
    small.node_volume = torch.linspace(0.5, 1.5, small.num_nodes)
    large.node_volume = torch.linspace(0.6, 1.4, large.num_nodes)
    model = MGNTransolverHybrid(11, 4, hidden_dim=32, n_message_passing_steps=2,
                                attention_blocks=2, attention_heads=4, attention_slices=8, dropout=0.0)
    model.eval()
    batch = Batch.from_data_list([small, large])
    batched = model(batch)
    assert batched.shape == (34,)
    assert torch.allclose(batched[:13], model(small), atol=1e-5)
    assert torch.allclose(batched[13:], model(large), atol=1e-5)
    batched.square().mean().backward()
    assert model.global_blocks[0].attn.slice_logits.weight.grad is not None
    assert model.processor[0].edge_mlp.net[0].weight.grad is not None
