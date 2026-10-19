import numpy as np
import pandas as pd
import pytest

from pipeline.anonymize import anonymize
from pipeline.bias import outcome_disparity, representation_report
from pipeline.clean import clean
from pipeline.transform import integrate, transform
from pipeline.validate import validate


@pytest.fixture(scope="module")
def staged(raw_df):
    import os
    os.environ["ANON_SALT"] = "test-salt"
    anon, _ = anonymize(clean(raw_df))
    return anon


@pytest.fixture(scope="module")
def curated(staged):
    return integrate(transform(staged))


def test_transform_builds_binary_target_and_drops_unresolved(staged):
    out = transform(staged)
    assert set(out["default"].unique()) == {0, 1}
    assert len(out) < len(staged)  # Current / grace-period loans have no label
    assert "loan_status" not in out.columns


def test_transform_features_are_finite(staged):
    out = transform(staged)
    feats = out[["loan_to_income", "installment_to_income", "credit_history_years", "log_annual_inc"]]
    assert np.isfinite(feats.to_numpy()).all()
    assert (out["credit_history_years"] >= 0).all()


def test_integrate_keeps_row_count_and_fills_region(staged):
    t = transform(staged)
    out = integrate(t)
    assert len(out) == len(t)
    assert out["region"].notna().all()


def test_validation_passes_on_curated(curated):
    result = validate(curated, out_path=False)
    assert result["success"] and result["failed"] == 0


def test_validation_fails_on_corrupted_data(curated):
    bad = curated.copy()
    bad.loc[bad.index[:50], "dti"] = 500
    bad.loc[bad.index[50:80], "grade"] = "Z"
    with pytest.raises(ValueError):
        validate(bad, out_path=False)


def test_representation_report_flags_small_and_shifted_groups():
    before = pd.DataFrame({"g": ["a"] * 90 + ["b"] * 10})
    after = pd.DataFrame({"g": ["a"] * 98 + ["b"] * 2})
    rep = representation_report(before, after, attrs=["g"])
    b = rep[rep.group == "b"].iloc[0]
    assert b.under_represented and b.shifted


def test_outcome_disparity_flags_low_rate_ratio():
    df = pd.DataFrame({"g": ["x"] * 200 + ["y"] * 200, "default": [1] * 20 + [0] * 180 + [1] * 60 + [0] * 140})
    out = outcome_disparity(df, attrs=["g"], min_n=100)
    assert out["review"].all() and out["attr_rate_ratio"].iloc[0] == pytest.approx(0.33, abs=0.01)
