"""Metrics: ranking, operating-point, tier and decile calibration."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from . import config


def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def capture_top(y, s, share=config.HIGH_SHARE) -> float:
    """Share of all defaulters found in the highest-scored `share` of loans."""
    y = np.asarray(y)
    k = int(round(share * len(y)))
    order = np.argsort(-np.asarray(s), kind="stable")[:k]
    return float(y[order].sum() / y.sum())


def evaluate(model, X, y) -> dict:
    y = np.asarray(y)
    raw = model.raw(X)
    cal = model.calibrator.predict(raw)
    flag = model.flag(raw)
    tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "default_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, raw)), "ks": ks_stat(y, raw),
        "capture_top20": capture_top(y, raw),
        "flagged_share": float(flag.mean()),
        "precision": float(precision_score(y, flag, zero_division=0)),
        "recall": float(recall_score(y, flag, zero_division=0)),
        "f1": float(f1_score(y, flag, zero_division=0)),
        "accuracy": float(accuracy_score(y, flag)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "defaulters": int(y.sum()),
        "brier": float(np.mean((cal - y) ** 2)),
    }


def baseline(y) -> dict:
    y = np.asarray(y)
    return {"n": int(len(y)), "default_rate": float(y.mean()), "roc_auc": 0.5, "ks": 0.0,
            "capture_top20": config.HIGH_SHARE, "flagged_share": 0.0, "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "accuracy": float(1 - y.mean())}


def tier_calibration(model, X, y) -> pd.DataFrame:
    raw = model.raw(X)
    cal = model.calibrator.predict(raw)
    d = pd.DataFrame({"tier": model.tier(raw), "pred": cal, "y": np.asarray(y)})
    t = d.groupby("tier").agg(n=("y", "size"), predicted=("pred", "mean"), actual=("y", "mean")).reindex(["low", "medium", "high"])
    t["relative_error"] = t["predicted"] / t["actual"] - 1
    return t.reset_index()


def decile_calibration(model, X, y) -> pd.DataFrame:
    raw = model.raw(X)
    cal = model.calibrator.predict(raw)
    d = pd.DataFrame({"raw": raw, "pred": cal, "y": np.asarray(y)})
    d["decile"] = pd.qcut(d["raw"].rank(method="first"), 10, labels=range(1, 11))
    t = d.groupby("decile", observed=True).agg(n=("y", "size"), predicted=("pred", "mean"), actual=("y", "mean"))
    t["relative_error"] = t["predicted"] / t["actual"] - 1
    return t.reset_index()
