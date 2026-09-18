import h5py
import numpy as np
import pytest

from ic_thermbench_adapter import (
    ICThermBenchH5Dataset,
    inspect_mat_pair,
    load_official_metrics,
    official_s3_s4_indices,
    s5_fewshot_indices,
)


def test_lazy_axis_mapping_and_contract(tmp_path):
    raw_x = np.arange(4 * 4 * 2 * 3 * 5, dtype=np.float32).reshape(4, 4, 2, 3, 5)
    raw_y = np.arange(4 * 2 * 3 * 5, dtype=np.float32).reshape(4, 2, 3, 5)
    with h5py.File(tmp_path / "input.mat", "w") as handle:
        handle.create_dataset("data", data=raw_x)
    with h5py.File(tmp_path / "output.mat", "w") as handle:
        handle.create_dataset("data", data=raw_y)

    contract = inspect_mat_pair(tmp_path, "level3")
    assert contract["raw_input_shape"] == [4, 4, 2, 3, 5]
    assert contract["adapter_input_shape"] == [4, 5, 3, 2, 4]

    dataset = ICThermBenchH5Dataset(tmp_path, "level3", indices=[2])
    features, target = dataset[0]
    assert features.shape == (5, 3, 2, 4)
    assert target.shape == (5, 3, 2)
    np.testing.assert_array_equal(features, raw_x[2].transpose(3, 2, 1, 0))
    np.testing.assert_array_equal(target, raw_y[2].transpose(2, 1, 0))


def test_official_s3_s4_split_sizes_and_fixed_test():
    split = official_s3_s4_indices(15_000)
    assert {name: len(indices) for name, indices in split.items()} == {
        "train": 10_800,
        "val": 1_200,
        "test": 3_000,
    }
    assert split["test"][0] == 12_000
    assert split["test"][-1] == 14_999


def test_s5_case_grouped_protocol():
    manifest = [
        {"case": f"case-{case}", "source_file": f"{case}-{sample}.npz"}
        for case in range(5)
        for sample in range(1000)
    ]
    split = s5_fewshot_indices(manifest, shots=10)
    assert len(split["adaptation"]) == 50
    assert len(split["holdout"]) == 2500
    assert len(split["zero_shot"]) == 5000
    assert split["adaptation"][:10].tolist() == list(range(10))
    assert split["holdout"][:3].tolist() == [500, 501, 502]


def test_metrics_are_loaded_from_official_source():
    root = __import__("pathlib").Path(__file__).parents[1] / "external" / "IC-ThermBench"
    if not (root / "utils" / "metrics.py").is_file():
        pytest.skip("official IC-ThermBench source is an optional external artifact")
    metric_fn = load_official_metrics(root)
    target = np.arange(128, dtype=np.float32).reshape(2, 8, 8)
    pred = target + 2.0
    metrics = metric_fn(pred, target, prefix="test", topk=50)
    assert metrics["test/rmse"] == 2.0
    assert metrics["test/mean_absolute_error"] == 2.0
    assert metrics["test/max_absolute_error"] == 2.0
    assert metrics["test/max_temperature_error"] == 2.0
    assert metrics["test/topk50_temperature_difference"] == 2.0
