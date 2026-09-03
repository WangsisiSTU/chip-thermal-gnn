"""Regression test for the independent 3D manufactured-solution verification."""
from verify_mms_3d import run_mms_study


def test_mms_tetrahedral_refinement_reduces_error():
    report = run_mms_study((5, 9, 13))
    errors = [row["nodal_rmse"] for row in report["rows"]]
    assert errors[0] > errors[1] > errors[2]
    assert min(report["observed_orders"]) > 1.2
