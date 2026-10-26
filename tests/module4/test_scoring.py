"""Scoring tests: split integrity, leakage guard, calibration, tiers, monotone rule, metrics and fairness maths."""
import numpy as np
import pandas as pd
import pytest

from src.module4 import config, data, evaluate as ev, fairness
from src.module4.model import feature_frame


def test_split_is_chronological_and_disjoint(frames):
    tr, va, te = frames
    assert tr.issue_year.max() <= config.TRAIN_LAST_YEAR
    assert set(va.issue_year) == {config.VAL_YEAR} and set(te.issue_year) == {config.TEST_YEAR}
    assert not set(tr.index) & set(va.index) and not set(va.index) & set(te.index)


def test_features_exclude_leaky_or_identifier_columns():
    for bad in config.EXCLUDED + [config.TARGET]:
        assert bad not in config.FEATURES
    assert len(config.NUMERIC) == 18 and len(config.CATEGORICAL) == 4


def test_income_groups_are_quartiles(frames):
    tr, _, _ = frames
    sizes = tr["income_quartile"].value_counts(normalize=True)
    assert len(sizes) == 4 and sizes.between(0.2, 0.3).all()


def test_calibrated_probabilities_are_valid_and_monotone(fitted, frames):
    X = feature_frame(frames[2])
    raw, cal = fitted.raw(X), fitted.predict_proba(X)
    assert ((cal >= 0) & (cal <= 1)).all()
    order = np.argsort(raw)
    assert (np.diff(cal[order]) >= -1e-12).all()


def test_tiers_follow_the_20_30_50_rule_on_validation(fitted, frames):
    va = frames[1]
    t = fitted.tier(fitted.raw(feature_frame(va)))
    share = pd.Series(t).value_counts(normalize=True)
    assert abs(share["high"] - 0.20) < 0.03 and abs(share["medium"] - 0.30) < 0.03


def test_lightgbm_risk_does_not_fall_when_dti_rises(fitted, frames):
    X = feature_frame(frames[2]).head(200)
    hi = X.copy(); hi["dti"] = hi["dti"] + 10
    assert (fitted.raw(hi) >= fitted.raw(X) - 1e-9).all()


def test_ks_and_capture_on_perfect_and_random_scores():
    y = np.array([0] * 80 + [1] * 20)
    assert ev.ks_stat(y, y.astype(float)) == pytest.approx(1.0)
    assert ev.capture_top(y, y.astype(float)) == pytest.approx(1.0)


def test_fairness_metrics_on_known_data():
    y = np.array([1, 0] * 300 + [1, 0] * 300)
    g = np.array(["A"] * 600 + ["B"] * 600)
    flag = np.array([1, 0] * 300 + [0, 0] * 300)           # B never flagged
    s = fairness.summarize(y, flag, g)
    assert s["approval_ratio"] == pytest.approx(0.5)       # 0.5 vs 1.0
    assert s["tpr_gap_pp"] == pytest.approx(100.0) and s["fpr_gap_pp"] == pytest.approx(0.0)
    assert s["passes"] is False


def test_small_groups_are_excluded_from_fairness():
    y = np.array([0, 1] * 300 + [0, 1] * 10)
    g = np.array(["big"] * 600 + ["tiny"] * 20)
    s = fairness.summarize(y, np.zeros(len(y), int), g, min_n=200)
    assert s["excluded_groups"] == ["tiny"]


def test_threshold_optimizer_reduces_a_large_gap(fitted, frames):
    tr, va, te = frames
    g = ["income_quartile"]
    Xva, Xte = feature_frame(va).reset_index(drop=True), feature_frame(te).reset_index(drop=True)
    yva, yte = va[config.TARGET].reset_index(drop=True), te[config.TARGET].reset_index(drop=True)
    gva, gte = va[g].reset_index(drop=True), te[g].reset_index(drop=True)
    before = fairness.audit(fitted, Xte, yte, gte, min_n=50)["income_quartile"]
    after = fairness.mitigate(fitted, Xva, yva, gva, Xte, yte, gte, min_n=50)["income_quartile"]
    assert after["tpr_gap_pp"] < before["tpr_gap_pp"]


def test_amortised_payment_matches_known_value():
    assert float(data.amortised_payment(10000, 12.0, 36)) == pytest.approx(332.14, abs=0.01)


def test_employment_group_uses_the_missing_flag_from_module3(frames):
    tr, _, _ = frames
    assert "Not reported" in set(tr["employment_length_group"])
    flagged = tr[tr["emp_length_missing"] == 1]
    assert (flagged["employment_length_group"] == "Not reported").all()
