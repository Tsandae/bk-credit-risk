"""Continuous monitoring: data drift, model drift, calibration drift and fairness (bias) drift.

    python -m src.final.monitor                      # monitor the 2014 and 2015 windows against 2007-2013
    python -m src.final.monitor --current new.parquet --label 2026-Q1   # monitor a new batch of scored loans

Reference = the training window (2007-2013). Alert limits come from the Module 2 plan and the Module 4 report:
population stability index (PSI) above 0.2, AUC more than 0.03 below the validation AUC, tier calibration outside
+/-10%, approval ratio below 0.80, TPR/FPR gap above 5 points. Two breaches in a row mean 'review or retrain'.
Writes reports/final/monitoring_report.json (+ dashboard/data/monitoring.json for the dashboard).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

from src.module4 import config, data  # noqa: E402
from src.module4.model import feature_frame  # noqa: E402
import helpers as H  # noqa: E402

PSI_LIMIT = 0.20
AUC_DROP_LIMIT = 0.03
TIER_ERR_LIMIT = 0.10
DRIFT_FEATURES = ["int_rate", "annual_inc", "dti", "loan_amnt", "revol_util", "loan_to_income", "term", "grade", "home_ownership", "purpose"]


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                         # near-constant feature (e.g. term has two values)
        edges = np.unique(np.concatenate([[-np.inf], np.unique(ref), [np.inf]]))
    edges[0], edges[-1] = -np.inf, np.inf
    return _psi(np.histogram(ref, edges)[0], np.histogram(cur, edges)[0])


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().astype(str)) | set(cur.dropna().astype(str)))
    r = ref.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    c = cur.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    return _psi(r, c)


def _psi(r, c) -> float:
    r = np.maximum(np.asarray(r, float) / max(np.sum(r), 1), 1e-4)
    c = np.maximum(np.asarray(c, float) / max(np.sum(c), 1), 1e-4)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_stat(y, s) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def score_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = model.raw(feature_frame(df))
    out["raw"] = raw
    out["prob"] = model.calibrator.predict(raw)
    out["tier"] = model.tier(raw)
    return out


def window_metrics(model, w: pd.DataFrame, ref: pd.DataFrame, ref_auc: float) -> dict:
    y = w[config.TARGET].to_numpy()
    flag = model.flag(w["raw"])
    auc = float(roc_auc_score(y, w["raw"]))
    tiers = {}
    for t in ["low", "medium", "high"]:
        m = w["tier"] == t
        pred, act = float(w.loc[m, "prob"].mean()), float(w.loc[m, config.TARGET].mean())
        tiers[t] = {"share": float(m.mean()), "predicted": pred, "actual": act, "relative_error": (pred - act) / act if act else None}
    psi = {"score": psi_numeric(ref["raw"], w["raw"])}
    for f in DRIFT_FEATURES:
        psi[f] = psi_categorical(ref[f], w[f]) if f in config.CATEGORICAL else psi_numeric(ref[f], w[f])
    fair = {}
    for g in config.FAIR_GROUPS:
        try:
            _, s = H.fairness_view(w[[g, config.TARGET, "raw"]].rename(columns={config.TARGET: "default"}), g, model.cut_high, config.MIN_GROUP_N)
            fair[g] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in s.items() if not isinstance(v, (list, dict))}
        except Exception as e:                    # group too small in this window
            fair[g] = {"error": str(e)}
    return {
        "n": int(len(w)), "default_rate": float(y.mean()), "mean_predicted": float(w["prob"].mean()),
        "auc": auc, "ks": ks_stat(y, w["raw"]), "capture_top20": float(((flag == 1) & (y == 1)).sum() / max(y.sum(), 1)),
        "flagged_share": float(flag.mean()), "brier": float(brier_score_loss(y, w["prob"])),
        "auc_drop_vs_reference": float(ref_auc - auc), "tiers": tiers, "psi": psi, "fairness": fair,
    }


def alerts_for(m: dict) -> list:
    a = []
    for name, v in m["psi"].items():
        if v > PSI_LIMIT:
            a.append({"type": "data/score drift", "item": name, "value": round(v, 3), "limit": PSI_LIMIT})
    if m["auc_drop_vs_reference"] > AUC_DROP_LIMIT:
        a.append({"type": "model drift", "item": "AUC drop", "value": round(m["auc_drop_vs_reference"], 3), "limit": AUC_DROP_LIMIT})
    for t, v in m["tiers"].items():
        if v["relative_error"] is not None and abs(v["relative_error"]) > TIER_ERR_LIMIT:
            a.append({"type": "calibration drift", "item": f"{t} tier", "value": round(v["relative_error"], 3), "limit": TIER_ERR_LIMIT})
    for g, f in m["fairness"].items():
        if "error" in f:
            continue
        if f.get("approval_ratio", 1) < config.MIN_APPROVAL_RATIO:
            a.append({"type": "bias drift", "item": f"{g}: approval ratio", "value": round(f["approval_ratio"], 3), "limit": config.MIN_APPROVAL_RATIO})
        for k in ("tpr_gap_pp", "fpr_gap_pp"):
            if f.get(k, 0) > config.MAX_GAP_PP:
                a.append({"type": "bias drift", "item": f"{g}: {k.split('_')[0].upper()} gap", "value": round(f[k], 1), "limit": config.MAX_GAP_PP})
    return a


def run(current_path=None, label=None, write=True) -> dict:
    model = joblib.load(config.MODEL_FILE)
    df = data.load()
    tr, va, te = data.split(df)
    edges = data.income_edges(tr)
    ref = score_frame(model, data.add_groups(tr, edges))
    ref_val_auc = float(roc_auc_score(va[config.TARGET], score_frame(model, data.add_groups(va, edges))["raw"]))
    windows = {"2014 (validation)": va, "2015 (test)": te}
    if current_path:
        windows = {label or Path(current_path).stem: data.load(current_path)}
    results, previous_breach = {}, False
    for name, w in windows.items():
        sw = score_frame(model, data.add_groups(w, edges))
        m = window_metrics(model, sw, ref, ref_val_auc)
        m["alerts"] = alerts_for(m)
        m["breach"] = bool(m["alerts"])
        m["action"] = ("REVIEW / RETRAIN (second breach in a row)" if m["breach"] and previous_breach
                       else "Investigate (first breach)" if m["breach"] else "No action")
        previous_breach = m["breach"]
        results[name] = m
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "reference": "2007-2013 training window",
              "reference_n": int(len(ref)), "reference_validation_auc": ref_val_auc,
              "limits": {"psi": PSI_LIMIT, "auc_drop": AUC_DROP_LIMIT, "tier_error": TIER_ERR_LIMIT,
                         "approval_ratio": config.MIN_APPROVAL_RATIO, "gap_pp": config.MAX_GAP_PP},
              "windows": results}
    if write:
        (ROOT / "reports" / "final").mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, indent=2)
        (ROOT / "reports" / "final" / "monitoring_report.json").write_text(text)
        out_dir = Path(os.environ.get("MONITORING_OUT_DIR", ROOT / "dashboard" / "data"))   # a shared volume in Docker Compose
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "monitoring.json").write_text(text)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", help="parquet of a new batch with the same columns as the curated file")
    ap.add_argument("--label", help="name for the new window, e.g. 2026-Q1")
    a = ap.parse_args()
    rep = run(a.current, a.label)
    for name, m in rep["windows"].items():
        print(f"{name}: n={m['n']:,} AUC={m['auc']:.3f} KS={m['ks']:.3f} score PSI={m['psi']['score']:.3f} alerts={len(m['alerts'])} -> {m['action']}")
        for al in m["alerts"]:
            print("   -", al["type"], al["item"], al["value"], "(limit", al["limit"], ")")


if __name__ == "__main__":
    main()
