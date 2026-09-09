import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from train import peak_loss_fn, run_epoch
from evaluate import checkpoint_normalization, evaluate_model
from utils import build_data_contract, validate_data_contract


class IdentityModel(torch.nn.Module):
    def forward(self, batch):
        return batch.x[:, 0]


@pytest.mark.parametrize("loss_type", ["zscore_mse", "per_sample_rel"])
def test_validation_loss_is_independent_of_partial_batch(loss_type):
    samples = [Data(x=torch.full((2, 1), value), y=torch.tensor([0., 1.]))
               for value in (0., 1., 9.)]
    results = [run_epoch(IdentityModel(), DataLoader(samples, batch_size=size),
                         torch.device("cpu"), 0., 1., 0.2, loss_type)
               for size in (1, 2, 3)]
    for result in results[1:]:
        assert result == pytest.approx(results[0], rel=1e-6)


def test_empty_epoch_has_clear_error():
    with pytest.raises(ValueError, match="empty dataset"):
        run_epoch(IdentityModel(), [], torch.device("cpu"), 0., 1., 0., "zscore_mse")


def test_checkpoint_normalization_decodes_training_units():
    mean, std = checkpoint_normalization({"dT_mean": 12., "dT_std": 3.}, torch.device("cpu"))
    assert (torch.tensor([2.]) * std + mean).item() == 18.


@pytest.mark.parametrize("std", [0., -1., float("nan"), float("inf")])
def test_invalid_checkpoint_scale_is_rejected(std):
    with pytest.raises(ValueError):
        checkpoint_normalization({"dT_mean": 0., "dT_std": std}, torch.device("cpu"))


def _metadata_contract():
    return {
        "dimension": 3,
        "node_feature_dim": 24,
        "edge_feature_dim": 5,
        "node_feature_names": ["x", "y"],
        "edge_feature_names": ["dx"],
        "norm_consts": {"q_ref": 1.0},
        "dT_train_mean": 2.0,
        "dT_train_std": 0.5,
    }


def test_data_contract_rejects_semantic_feature_mismatch():
    metadata = _metadata_contract()
    checkpoint = {
        "node_in_dim": 24,
        "edge_in_dim": 5,
        "data_contract": build_data_contract(metadata),
    }
    validate_data_contract(checkpoint, metadata)

    metadata["node_feature_names"] = ["y", "x"]
    with pytest.raises(ValueError, match="node_feature_names"):
        validate_data_contract(checkpoint, metadata)


def test_legacy_checkpoint_checks_dimensions_and_warns():
    metadata = _metadata_contract()
    checkpoint = {"node_in_dim": 24, "edge_in_dim": 5}
    with pytest.warns(RuntimeWarning, match="no data contract"):
        validate_data_contract(checkpoint, metadata)

    metadata["edge_feature_dim"] = 6
    with pytest.raises(ValueError, match="expects node/edge dimensions"):
        validate_data_contract(checkpoint, metadata)

def test_global_peak_loss_penalizes_spurious_hotspots():
    prediction = torch.tensor([2.0, 10.0])
    truth = torch.tensor([3.0, 1.0])
    assert peak_loss_fn(prediction, truth, 1, "true_peak_node").item() == pytest.approx(1.0)
    assert peak_loss_fn(prediction, truth, 1, "global_max").item() == pytest.approx(49.0)


def test_unknown_peak_loss_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown peak loss type"):
        peak_loss_fn(torch.zeros(2), torch.zeros(2), 1, "invalid")

def test_evaluation_uses_warmed_single_and_mini_batch_timing():
    samples = []
    for value in (1.0, 2.0):
        sample = Data(x=torch.tensor([[value], [value + 1.0]]), y=torch.tensor([value, value + 1.0]))
        sample.solve_time_s = torch.tensor([0.01])
        samples.append(sample)
    info = [{"file": "one.npz", "regime": "id"}, {"file": "two.npz", "regime": "id"}]

    per_sample, summary = evaluate_model(
        IdentityModel(), samples, info, torch.tensor(0.0), torch.tensor(1.0), torch.device("cpu"),
        timing_repeats=2, benchmark_batch_size=2,
    )
    assert len(per_sample) == 2
    assert summary["overall_mae_K"] == pytest.approx(0.0)
    assert summary["timing_repeats"] == 2
    assert summary["benchmark_batch_size"] == 2
    assert summary["mean_inference_time_ms"] >= 0.0
    assert summary["batched_inference_time_per_sample_ms"] >= 0.0