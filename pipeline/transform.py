"""Stages 3-4 — Transformation (label + features) and Integration (join to reference data)."""
import numpy as np
import pandas as pd

from . import config
from .audit import audited

GRADE_ORDER = {g: i for i, g in enumerate("ABCDEFG")}


@audited("TRANSFORM", "staging")
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary target from loan outcome; unresolved loans (Current, grace period...) carry no label.
    df["default"] = np.where(df["loan_status"].isin(config.BAD_STATUS), 1,
                             np.where(df["loan_status"].isin(config.GOOD_STATUS), 0, np.nan))
    df = df.dropna(subset=["default"]).copy()
    df["default"] = df["default"].astype("int8")

    # Engineered features.
    income = df["annual_inc"].clip(lower=1)
    df["loan_to_income"] = df["loan_amnt"] / income
    df["installment_to_income"] = df["installment"] * 12 / income
    df["log_annual_inc"] = np.log1p(income)
    df["credit_history_years"] = ((df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25).clip(lower=0)
    df["issue_year"] = df["issue_d"].dt.year.astype("int16")
    df["grade_num"] = df["grade"].map(GRADE_ORDER).astype("int8")
    df["term"] = df["term"].astype("int16")

    return df.drop(columns=["loan_status", "issue_d", "earliest_cr_line"]).reset_index(drop=True)


@audited("INTEGRATE", "curated")
def integrate(df: pd.DataFrame, ref_path=None) -> pd.DataFrame:
    """Join the loan table to the state -> region reference table (many-to-one, validated)."""
    ref = pd.read_csv(ref_path or config.STATE_REGION_FILE)
    out = df.merge(ref, how="left", on="addr_state", validate="many_to_one")
    out["region"] = out["region"].fillna("Other/Unknown")
    return out
