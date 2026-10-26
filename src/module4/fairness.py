"""Fairness metrics (demographic parity, equalized odds, disparate impact) and ThresholdOptimizer mitigation."""
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, precision_score

from . import config
from .model import ScoreEstimator


def group_table(y, flag, groups, min_n=config.MIN_GROUP_N) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "flag": np.asarray(flag), "g": np.asarray(groups)})
    rows = []
    for g, s in d.groupby("g"):
        pos, neg = s[s.y == 1], s[s.y == 0]
        rows.append({"group": g, "n": len(s), "approval_rate": 1 - s.flag.mean(), "default_rate": s.y.mean(),
                     "tpr": pos.flag.mean() if len(pos) else np.nan, "fpr": neg.flag.mean() if len(neg) else np.nan,
                     "included": len(s) >= min_n})
    return pd.DataFrame(rows)


def summarize(y, flag, groups, min_n=config.MIN_GROUP_N) -> dict:
    t = group_table(y, flag, groups, min_n)
    k = t[t.included]
    appr = k.approval_rate
    out = {
        "approval_ratio": float(appr.min() / appr.max()),           # disparate impact ratio
        "demographic_parity_diff_pp": float((appr.max() - appr.min()) * 100),
        "tpr_gap_pp": float((k.tpr.max() - k.tpr.min()) * 100),
        "fpr_gap_pp": float((k.fpr.max() - k.fpr.min()) * 100),
        "excluded_groups": t[~t.included]["group"].tolist(),
        "groups": k.round(4).to_dict(orient="records"),
    }
    out["equalized_odds_diff_pp"] = max(out["tpr_gap_pp"], out["fpr_gap_pp"])
    out["passes"] = bool(out["approval_ratio"] > config.MIN_APPROVAL_RATIO
                         and out["tpr_gap_pp"] <= config.MAX_GAP_PP and out["fpr_gap_pp"] <= config.MAX_GAP_PP)
    return out


def audit(model, Xte, yte, gte, min_n=config.MIN_GROUP_N):
    """Before-mitigation fairness for each group attribute, using the operating cutoff."""
    flag = model.flag(model.raw(Xte))
    return {a: summarize(yte, flag, gte[a], min_n) for a in gte.columns}


def mitigate(model, Xva, yva, gva, Xte, yte, gte, min_n=config.MIN_GROUP_N, seed=config.SEED):
    """ThresholdOptimizer (equalized odds), fitted on 2014 validation, applied to 2015 test."""
    out = {}
    for a in gva.columns:
        keep_v = gva[a].map(gva[a].value_counts()) >= min_n
        keep_t = gte[a].map(gte[a].value_counts()) >= min_n
        to = ThresholdOptimizer(estimator=ScoreEstimator(model), constraints="equalized_odds",
                                objective="balanced_accuracy_score", prefit=True, predict_method="predict_proba")
        to.fit(Xva[keep_v], yva[keep_v], sensitive_features=gva[a][keep_v])
        flag = np.asarray(to.predict(Xte[keep_t], sensitive_features=gte[a][keep_t], random_state=seed)).astype(int)
        s = summarize(yte[keep_t], flag, gte[a][keep_t], min_n)
        s["flagged_share"] = float(flag.mean())
        s["precision"] = float(precision_score(yte[keep_t], flag, zero_division=0))
        s["accuracy"] = float(accuracy_score(yte[keep_t], flag))
        out[a] = s
    return out


def calibration_by_group(model, X, y, groups, min_n=config.MIN_GROUP_N) -> list:
    """Predicted vs actual default rate and Brier score per group (is risk over- or under-stated for a group?)."""
    cal = model.predict_proba(X)
    d = pd.DataFrame({"y": np.asarray(y), "p": cal, "g": np.asarray(groups)})
    rows = []
    for g, s in d.groupby("g"):
        if len(s) < min_n:
            continue
        rows.append({"group": g, "n": int(len(s)), "predicted": float(s.p.mean()), "actual": float(s.y.mean()),
                     "relative_error": float(s.p.mean() / s.y.mean() - 1), "brier": float(((s.p - s.y) ** 2).mean())})
    return rows
