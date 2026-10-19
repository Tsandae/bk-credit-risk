"""Stage 6 — Bias detection suite: representation bias and outcome-rate disparity (Fairlearn).

Reports are advisory (REVIEW flags), not a hard gate: legitimate risk factors also move default rates,
so a human must document why a flagged gap is acceptable before Module 4 modelling.
"""
import json

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame

from . import config
from .audit import log_event


def representation_report(before: pd.DataFrame, after: pd.DataFrame, attrs=None,
                          min_share=config.MIN_GROUP_SHARE, max_shift=config.MAX_SHARE_SHIFT) -> pd.DataFrame:
    """Compare each group's share of rows before vs after processing; flag small or shrinking groups."""
    rows = []
    for attr in attrs or config.BIAS_ATTRIBUTES:
        sb = before[attr].value_counts(normalize=True)
        sa = after[attr].value_counts(normalize=True)
        for group in sorted(set(sb.index) | set(sa.index), key=str):
            b, a = float(sb.get(group, 0.0)), float(sa.get(group, 0.0))
            rows.append({
                "attribute": attr, "group": str(group),
                "share_before": round(b, 4), "share_after": round(a, 4),
                "shift_pp": round((a - b) * 100, 2),
                "under_represented": a < min_share,
                "shifted": abs(a - b) > max_shift,
            })
    return pd.DataFrame(rows)


def outcome_disparity(df: pd.DataFrame, attrs=None, label="default", min_n=100,
                      min_ratio=config.MIN_RATE_RATIO) -> pd.DataFrame:
    """Per-group default rate via Fairlearn's MetricFrame, with gap/ratio across groups of at least `min_n` rows."""
    rows = []
    y = df[label].astype(int)
    for attr in attrs or config.BIAS_ATTRIBUTES:
        mf = MetricFrame(
            metrics={
                "rate": lambda y_true, y_pred: float(np.mean(y_pred)),
                "n": lambda y_true, y_pred: int(len(y_true)),
            },
            y_true=y, y_pred=y, sensitive_features=df[attr].astype(str),
        )
        by = mf.by_group
        big = by[by["n"] >= min_n]
        gap = float(big["rate"].max() - big["rate"].min()) if len(big) > 1 else 0.0
        ratio = float(big["rate"].min() / big["rate"].max()) if len(big) > 1 and big["rate"].max() > 0 else 1.0
        for group, r in by.iterrows():
            rows.append({
                "attribute": attr, "group": str(group), "n": int(r["n"]), "default_rate": round(float(r["rate"]), 4),
                "attr_gap": round(gap, 4), "attr_rate_ratio": round(ratio, 3),
                "review": ratio < min_ratio and r["n"] >= min_n,
            })
    return pd.DataFrame(rows)


def run_bias_suite(before: pd.DataFrame, after: pd.DataFrame, out_dir=None) -> dict:
    rep = representation_report(before, after)
    disp = outcome_disparity(after)
    summary = {
        "groups_under_represented": rep.loc[rep.under_represented, ["attribute", "group", "share_after"]].to_dict("records"),
        "groups_shifted_by_processing": rep.loc[rep.shifted, ["attribute", "group", "share_before", "share_after", "shift_pp"]].to_dict("records"),
        "attributes_needing_review": sorted(disp.loc[disp.review, "attribute"].unique().tolist()),
        "year_shift": _year_shift(before, after),
    }
    out_dir = out_dir or config.REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_dir / "bias_representation.csv", index=False)
    disp.to_csv(out_dir / "bias_outcome_disparity.csv", index=False)
    (out_dir / "bias_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log_event("BIAS_AUDIT", "curated",
              under_represented=len(summary["groups_under_represented"]),
              shifted=len(summary["groups_shifted_by_processing"]),
              review_attributes=summary["attributes_needing_review"])
    return {"summary": summary, "representation": rep, "disparity": disp}


def _year_shift(before: pd.DataFrame, after: pd.DataFrame) -> list:
    """Issue-year share before vs after label filtering (unresolved recent loans drop out)."""
    if "issue_d" not in before.columns or "issue_year" not in after.columns:
        return []
    b = before["issue_d"].dt.year.value_counts(normalize=True).sort_index()
    a = after["issue_year"].value_counts(normalize=True).sort_index()
    return [{"year": int(y), "share_before": round(float(b.get(y, 0)), 4), "share_after": round(float(a.get(y, 0)), 4)}
            for y in sorted(set(b.index) | set(a.index))]
