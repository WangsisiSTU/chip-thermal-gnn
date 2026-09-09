"""Sampling-isolation tests for three-dimensional dataset generation."""
from copy import deepcopy

from generate_dataset_3d import build_case_list_3d


def sampling_config():
    return {
        "seed": 2024,
        "materials": {
            "k_die": {"low": 130.0, "high": 155.0},
            "k_tim": {"low": 1.0, "high": 8.0},
            "k_cu": {"low": 360.0, "high": 400.0},
            "k_sub": {"low": 15.0, "high": 60.0},
        },
        "heat_source": {
            "q_base": {"low": 1e5, "high": 5e5},
            "q_hot": {"low": 5e6, "high": 3e7},
            "hotspot_center_x_frac": {"low": 0.25, "high": 0.75},
            "hotspot_center_z_frac": {"low": 0.25, "high": 0.75},
            "hotspot_width_x_frac": {"low": 0.15, "high": 0.4},
            "hotspot_width_z_frac": {"low": 0.15, "high": 0.4},
        },
        "boundary": {"t_ambient": {"low": 20.0, "high": 45.0}, "h_top": {"low": 500.0, "high": 15000.0}},
        "ood": {
            "n_ood_test_samples": 6,
            "q_hot": {"low": 3.5e7, "high": 6e7},
            "k_tim": {"low": 0.2, "high": 0.9},
            "h_top": {"low": 100.0, "high": 400.0},
        },
        "split": {"n_train": 3, "n_val": 2, "n_test": 9},
    }


def serialise(cases, split=None, regime=None):
    return [case.to_dict() for case_split, case in cases if (split is None or case_split == split) and (regime is None or case.regime == regime)]


def test_training_split_growth_does_not_change_validation_or_test_cases():
    base = sampling_config()
    expanded = deepcopy(base)
    expanded["split"]["n_train"] = 11

    base_cases = build_case_list_3d(base)
    expanded_cases = build_case_list_3d(expanded)

    assert serialise(base_cases, "train") == serialise(expanded_cases, "train")[:3]
    assert serialise(base_cases, "val") == serialise(expanded_cases, "val")
    assert serialise(base_cases, "test") == serialise(expanded_cases, "test")


def test_id_test_growth_does_not_change_ood_cases():
    base = sampling_config()
    expanded = deepcopy(base)
    expanded["split"]["n_test"] = 17

    base_cases = build_case_list_3d(base)
    expanded_cases = build_case_list_3d(expanded)
    for regime in ("ood_high_power", "ood_poor_tim", "ood_poor_cooling"):
        assert serialise(base_cases, regime=regime) == serialise(expanded_cases, regime=regime)

def test_covered_poor_cooling_has_explicit_split_counts_and_labels():
    cfg = sampling_config()
    cfg["split"] = {"n_train": 5, "n_val": 3, "n_test": 6}
    cfg["coverage"] = {
        "poor_cooling": {
            "train": 2,
            "val": 1,
            "test": 1,
            "h_top": {"low": 100.0, "high": 400.0},
        }
    }
    cfg["ood"]["test_counts"] = {"high_power": 1, "poor_tim": 1, "poor_cooling": 1}

    cases = build_case_list_3d(cfg)
    expected_counts = {"train": cfg["split"]["n_train"], "val": cfg["split"]["n_val"], "test": cfg["split"]["n_test"]}
    assert {split: sum(case_split == split for case_split, _ in cases) for split in expected_counts} == expected_counts
    covered = [case for _, case in cases if case.regime == "covered_poor_cooling"]
    assert len(covered) == 4
    assert all(100.0 <= case.h_top <= 400.0 for case in covered)