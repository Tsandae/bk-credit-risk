"""Privacy gate — PII removal, pseudonymisation, generalisation and k-anonymity enforcement."""
import hashlib
import hmac
import os
import re

import pandas as pd

from . import config
from .audit import audited, log_event

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def scrub_text(text) -> str:
    """Mask e-mails, phone numbers and SSN-like strings in free text."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text
    text = EMAIL.sub("[EMAIL]", str(text))
    text = SSN.sub("[ID]", text)
    return PHONE.sub("[PHONE]", text)


def pseudonymize(series: pd.Series, salt: str) -> pd.Series:
    """Keyed hash (HMAC-SHA256). Deterministic per salt, so keys stay joinable but cannot be reversed without it."""
    key = salt.encode()
    return series.astype("string").map(lambda v: hmac.new(key, str(v).encode(), hashlib.sha256).hexdigest()[:16])


def pii_scan(df: pd.DataFrame) -> int:
    """Count remaining PII-like patterns in all text columns (should be 0 after anonymisation)."""
    hits = 0
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col == "loan_key":  # hex pseudonym, not free text
            continue
        s = df[col].dropna().astype(str)
        hits += int(s.str.contains(EMAIL).sum() + s.str.contains(SSN).sum() + s.str.contains(PHONE).sum())
    return hits


def k_anonymity(df: pd.DataFrame, quasi=None) -> int:
    quasi = quasi or config.QUASI_IDENTIFIERS
    return int(df.groupby(quasi, observed=True, dropna=False).size().min())


@audited("ANONYMIZE", "staging")
def anonymize(df: pd.DataFrame, salt: str = None, k: int = config.K_ANONYMITY):
    salt = salt or os.environ.get("ANON_SALT")
    if not salt:
        raise RuntimeError("ANON_SALT is not set: refusing to pseudonymise with a default key.")
    df = df.copy()

    # Pseudonymous join key replaces the loan id.
    df["loan_key"] = pseudonymize(df["id"], salt)

    # Generalise quasi-identifiers: 5-digit-style zip -> first 3 digits; income -> bands.
    df["zip3"] = df["zip_code"].astype("string").str.extract(r"(\d{3})")[0].fillna("000")
    df["income_band"] = pd.cut(
        df["annual_inc"], bins=[0, 40_000, 80_000, 120_000, float("inf")],
        labels=["<40k", "40-80k", "80-120k", "120k+"],
    ).astype("string")

    # Remove direct identifiers and free text (data minimisation).
    removed = [c for c in config.DIRECT_PII if c in df.columns]
    df = df.drop(columns=removed)

    # Enforce k-anonymity on the quasi-identifier combination:
    #   1) generalise zip3 (hundreds of values, nearly unique) to OTHER,
    #   2) generalise state to OTHER for rows still in groups smaller than k,
    #   3) suppress (drop) any rows that remain in groups smaller than k.
    q = config.QUASI_IDENTIFIERS
    k_before = k_anonymity(df)
    rows_before = len(df)
    if k_before < k:
        df["zip3"] = "OTHER"
    small = df.groupby(q, observed=True, dropna=False)["loan_key"].transform("size") < k
    df.loc[small, "addr_state"] = "OTHER"
    small = df.groupby(q, observed=True, dropna=False)["loan_key"].transform("size") < k
    df = df.loc[~small].reset_index(drop=True)
    k_after = k_anonymity(df)

    report = {
        "columns_removed": removed,
        "pseudonymised": ["id -> loan_key (HMAC-SHA256)"],
        "generalised": ["zip_code -> zip3", "annual_inc -> income_band (added)"],
        "k_target": k,
        "k_before": k_before,
        "k_after": k_after,
        "rows_suppressed": rows_before - len(df),
        "residual_pii_patterns": pii_scan(df),
    }
    log_event("ANONYMIZE_REPORT", "staging", **{k_: v for k_, v in report.items() if k_ != "pseudonymised"})
    return df, report
