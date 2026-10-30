"""Shared logic for the BK credit-risk stakeholder dashboard (kept separate from the UI so it can be tested)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"

# One colour system for the dashboard and the slides (validated categorical order: blue, orange, aqua).
BLUE, ORANGE, AQUA, NAVY, GREY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#0b2545", "#8d99aa", "#eef3f8"

PLAIN = {
    "int_rate": "Interest rate", "term": "Loan term (months)", "annual_inc": "Annual income",
    "dti": "Debt compared with income", "loan_to_income": "Loan size compared with income",
    "revol_util": "Share of credit limits in use", "open_acc": "Number of open credit lines",
    "total_acc": "Total credit accounts", "inq_last_6mths": "Credit enquiries in last 6 months",
    "installment_to_income": "Yearly repayments compared with income", "loan_amnt": "Loan amount",
    "installment": "Monthly instalment", "credit_history_years": "Length of credit history",
    "emp_length": "Years in current job", "emp_length_missing": "Employment length not reported",
    "revol_bal": "Revolving credit balance", "delinq_2yrs": "Late payments in last 2 years",
    "pub_rec": "Public records (e.g. bankruptcies)", "grade": "Lender's credit grade",
    "home_ownership": "Housing status", "verification_status": "Income verification", "purpose": "Loan purpose",
}

GROUP_LABELS = {"income_quartile": "Income band (quartiles)", "home_ownership": "Housing status",
                "employment_length_group": "Time in current job", "region": "Region (US, proxy data)"}

# Module 2 limits
MIN_APPROVAL_RATIO, MAX_GAP_PP, MIN_GROUP_N = 0.80, 5.0, 200


def load_results() -> dict:
    return json.loads((DATA / "results.json").read_text())


def load_scored() -> pd.DataFrame:
    return pd.read_parquet(DATA / "test_scored.parquet")


def load_meta() -> dict:
    return json.loads((DATA / "meta.json").read_text())


# ------------------------------------------------------------------ scoring
def amortised_payment(amount: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 1200.0
    return float(amount * r / (1 - (1 + r) ** (-months))) if r > 0 else float(amount / months)


def build_frame(model, v: dict) -> pd.DataFrame:
    """One applicant -> the 22-column frame the model expects (ratios and instalment derived here)."""
    from src.module4 import config

    income = max(float(v["annual_inc"]), 1.0)
    installment = amortised_payment(v["loan_amnt"], v["int_rate"], int(v["term"]))
    row = {
        "loan_amnt": float(v["loan_amnt"]), "term": int(v["term"]), "int_rate": float(v["int_rate"]),
        "installment": installment, "emp_length": float(v["emp_length"]), "annual_inc": income, "dti": float(v["dti"]),
        "delinq_2yrs": float(v["delinq_2yrs"]), "inq_last_6mths": float(v["inq_last_6mths"]),
        "open_acc": float(v["open_acc"]), "pub_rec": float(v["pub_rec"]), "revol_bal": float(v["revol_bal"]),
        "revol_util": float(v["revol_util"]), "total_acc": float(v["total_acc"]), "emp_length_missing": 0,
        "loan_to_income": float(v["loan_amnt"]) / income, "installment_to_income": installment * 12 / income,
        "credit_history_years": float(v["credit_history_years"]),
        "grade": v["grade"], "home_ownership": v["home_ownership"],
        "verification_status": v["verification_status"], "purpose": v["purpose"],
    }
    return pd.DataFrame([row])[config.FEATURES]


def score(model, frame: pd.DataFrame) -> dict:
    raw = model.raw(frame)
    return {"probability": float(model.calibrator.predict(raw)[0]), "tier": str(model.tier(raw)[0]), "raw": float(raw[0])}


def tier_bands(model) -> dict:
    """Calibrated risk at the tier cut-offs, for plain-language captions."""
    lo = float(model.calibrator.predict([model.cut_medium])[0])
    hi = float(model.calibrator.predict([model.cut_high])[0])
    return {"medium_from": lo, "high_from": hi}


def local_drivers(explainer, frame: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Top-k feature pushes (log-odds units) for one applicant, in plain language."""
    from src.module4 import config

    agg, _ = explainer.values(frame)
    d = pd.DataFrame({"feature": config.FEATURES, "push": agg[0], "value": [frame.iloc[0][c] for c in config.FEATURES]})
    d = d.reindex(d["push"].abs().sort_values(ascending=False).index).head(k)
    d["label"] = d["feature"].map(PLAIN)
    d["direction"] = np.where(d["push"] > 0, "pushes risk up", "pushes risk down")
    return d.reset_index(drop=True)


def fmt_value(feature: str, x) -> str:
    if feature in ("int_rate", "revol_util"):
        return f"{float(x):.1f}%"
    if feature in ("annual_inc", "loan_amnt", "installment", "revol_bal"):
        return f"${float(x):,.0f}"
    if feature in ("loan_to_income", "installment_to_income"):
        return f"{float(x):.2f}"
    if feature in ("term",):
        return f"{int(x)} months"
    if isinstance(x, (int, float, np.floating, np.integer)):
        return f"{float(x):g}"
    return str(x)


# ------------------------------------------------------------------ fairness (Fairlearn)
def fairness_view(scored: pd.DataFrame, attr: str, threshold: float, min_n: int = MIN_GROUP_N):
    """Per-group metrics with Fairlearn's MetricFrame at a given score cut-off.

    A flagged applicant goes to review / possible decline; everyone else counts as approved."""
    from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate, true_positive_rate

    flag = (scored["raw"] >= threshold).astype(int)
    mf = MetricFrame(
        metrics={"flagged": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate, "n": count,
                 "default_rate": lambda y_true, y_pred: float(np.mean(y_true))},
        y_true=scored["default"].astype(int), y_pred=flag, sensitive_features=scored[attr].astype(str))
    t = mf.by_group.reset_index().rename(columns={attr: "group"})
    t["approval_rate"] = 1 - t["flagged"]
    t["included"] = t["n"] >= min_n
    k = t[t.included]
    summary = {
        "approval_ratio": float(k.approval_rate.min() / k.approval_rate.max()),
        "tpr_gap_pp": float((k.tpr.max() - k.tpr.min()) * 100),
        "fpr_gap_pp": float((k.fpr.max() - k.fpr.min()) * 100),
        "flagged_share": float(flag.mean()),
    }
    summary["approval_ok"] = summary["approval_ratio"] > MIN_APPROVAL_RATIO
    summary["tpr_ok"] = summary["tpr_gap_pp"] <= MAX_GAP_PP
    summary["fpr_ok"] = summary["fpr_gap_pp"] <= MAX_GAP_PP
    summary["all_ok"] = summary["approval_ok"] and summary["tpr_ok"] and summary["fpr_ok"]
    return t, summary
