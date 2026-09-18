import h5py
import numpy as np

from multitopology_adapter import (
    DENSE_FEATURE_CHANNELS,
    load_sample,
    parse_sample_path,
    validate_stratified_split,
)


def _write_sample(path):
    shape = (256, 256)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[10:20, 10:20] = 1
    instance_map[30:40, 30:40] = 2
    with h5py.File(path, "w") as handle:
        handle.create_dataset("temperature", data=np.full(shape, 50, dtype=np.float32))
        handle.create_dataset("power_map", data=np.ones(shape, dtype=np.float32))
        handle.create_dataset("instance_map", data=instance_map)
        for key in ("ceramic_mask", "baseplate_mask", "copper_mask"):
            handle.create_dataset(key, data=np.ones(shape, dtype=np.uint8))
        handle.create_dataset("h_normalized", data=np.array([0.6], dtype=np.float32))
        handle.create_dataset("max_temps", data=np.array([50, 50], dtype=np.float32))
        handle.create_dataset("row_ids", data=np.array([1, 2], dtype=np.uint16))
        handle.attrs["h_value"] = 6000
        handle.attrs["version"] = "Mask_HB_V1"


def test_hdf5_mapping_keeps_instance_ids_categorical(tmp_path):
    path = tmp_path / "data.h5"
    _write_sample(path)
    sample = load_sample(path)
    assert sample["features"].shape == (256, 256, 6)
    assert sample["feature_channels"] == DENSE_FEATURE_CHANNELS
    assert sample["target"].shape == (256, 256)
    assert sample["instance_map"].dtype == np.uint16
    assert set(np.unique(sample["features"][..., 1])) == {0.0, 1.0}


def test_path_parser():
    parsed = parse_sample_path(
        "TestData/Mask_HB_V3-0/000_Pigbt150_Pfwd100_h8000/data.h5"
    )
    assert parsed["topology"] == "Mask_HB_V3-0"
    assert parsed["Pigbt"] == 150
    assert parsed["h_value"] == 8000


def test_split_audit_flags_layout_reuse():
    rows = []
    for split in ("TrainData", "ValData", "TestData"):
        rows.append(
            {
                "split": split,
                "topology": "Mask_HB_V1",
                "mask_id": "000",
                "Pigbt": "150",
                "Pfwd": "100",
                "h_value": "6000",
                "h_normalized": "0.6",
                "n_instances": "2",
                "bytes": "1",
                "path": f"{split}/Mask_HB_V1/000_Pigbt150_Pfwd100_h6000/data.h5",
            }
        )
    report = validate_stratified_split(rows)
    assert report["valid"]
    assert not report["layout_disjoint"]
    assert report["topology_split"]["Mask_HB_V1"]["layout_overlap"]["train_test"] == 1
