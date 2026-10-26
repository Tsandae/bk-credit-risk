"""Counterfactual what-ifs and sensitivity analysis (custom scripts, per the brief)."""
import numpy as np
import pandas as pd

from . import config
from .data import amortised_payment, derive
from .evaluate import evaluate
from .model import feature_frame


def _score(model, df):
    X = feature_frame(derive(df))
    raw = model.raw(X)
    return raw, model.predict_proba(X), model.tier(raw)


def counterfactuals(model, test: pd.DataFrame, n=3, seed=config.SEED) -> pd.DataFrame:
    """For flagged 60-month applicants: shorter term, smaller amount. Instalment is recomputed at the same rate."""
    X = feature_frame(test)
    raw = model.raw(X)
    flagged = test[(raw >= model.cut_high) & (test["term"] == 60)]
    picks = flagged.sample(min(n, len(flagged)), random_state=seed)
    rows = []
    for idx, a in picks.iterrows():
        base_raw, base_p, base_t = _score(model, a.to_frame().T.infer_objects())
        rows.append({"applicant": int(idx), "scenario": "as applied", "term": int(a.term), "loan_amnt": float(a.loan_amnt),
                     "installment": float(a.installment), "default_probability": float(base_p[0]), "tier": base_t[0]})
        scen = {"term 60 -> 36 months": dict(term=36),
                "amount -10%": dict(loan_amnt=a.loan_amnt * 0.9),
                "amount -20%": dict(loan_amnt=a.loan_amnt * 0.8)}
        for label, ch in scen.items():
            b = a.copy()
            for k, v in ch.items():
                b[k] = v
            b["installment"] = float(amortised_payment(b.loan_amnt, b.int_rate, b.term))
            r, p, t = _score(model, b.to_frame().T.infer_objects())
            rows.append({"applicant": int(idx), "scenario": label, "term": int(b.term), "loan_amnt": float(b.loan_amnt),
                         "installment": float(b.installment), "default_probability": float(p[0]), "tier": t[0]})
    return pd.DataFrame(rows)


def sensitivity(model, test: pd.DataFrame, seed=config.SEED) -> pd.DataFrame:
    """Average change in predicted default probability (percentage points) for +/-10% input changes."""
    base = model.predict_proba(feature_frame(test)).mean()

    def shifted(col, factor, reprice=False):
        d = test.copy()
        d[col] = d[col] * factor
        if reprice:
            d["installment"] = amortised_payment(d.loan_amnt, d.int_rate, d.term)
        return model.predict_proba(feature_frame(derive(d))).mean()

    rows = [
        ("annual_inc", "income -10%", shifted("annual_inc", 0.9)),
        ("annual_inc", "income +10%", shifted("annual_inc", 1.1)),
        ("int_rate", "interest rate +10%", shifted("int_rate", 1.1, True)),
        ("dti", "debt-to-income +10%", shifted("dti", 1.1)),
        ("loan_amnt", "loan amount +10%", shifted("loan_amnt", 1.1, True)),
        ("revol_util", "credit use +10%", shifted("revol_util", 1.1)),
    ]
    out = pd.DataFrame([{"input": a, "scenario": b, "change_pp": float((c - base) * 100)} for a, b, c in rows])
    # 5% random noise on continuous inputs -> AUC stability
    rng = np.random.default_rng(seed)
    noisy = test.copy()
    for c in ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_bal", "revol_util", "credit_history_years"]:
        noisy[c] = noisy[c] * (1 + rng.normal(0, 0.05, len(noisy)))
    auc0 = evaluate(model, feature_frame(test), test[config.TARGET])["roc_auc"]
    auc1 = evaluate(model, feature_frame(derive(noisy)), test[config.TARGET])["roc_auc"]
    out.attrs["auc_clean"], out.attrs["auc_noise5"] = auc0, auc1
    return out
