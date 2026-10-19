"""Stage 2 — Cleaning: duplicates, type conversion, invalid values, missing data, outliers."""
import numpy as np
import pandas as pd

from . import config
from .audit import audited

_EMP_MAP = {"< 1 year": 0, "10+ years": 10}


def parse_term(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36"""
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce").astype("float64")


def parse_pct(s: pd.Series) -> pd.Series:
    """'13.5%' or 13.5 -> 13.5"""
    return pd.to_numeric(s.astype("string").str.replace("%", "", regex=False).str.strip(), errors="coerce").astype("float64")


def parse_emp_length(s: pd.Series) -> pd.Series:
    """'< 1 year' -> 0, '10+ years' -> 10, '3 years' -> 3, 'n/a' -> NaN"""
    s = s.astype("string").str.strip()
    mapped = s.map(_EMP_MAP)
    digits = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    return pd.to_numeric(mapped, errors="coerce").astype("float64").fillna(digits.astype("float64"))


@audited("CLEAN", "staging")
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Duplicates: exact duplicates, then repeated loan ids.
    df = df.drop_duplicates().drop_duplicates(subset="id", keep="first")

    # 2. Type conversion.
    df["term"] = parse_term(df["term"])
    df["int_rate"] = parse_pct(df["int_rate"])
    df["revol_util"] = parse_pct(df["revol_util"])
    df["emp_length"] = parse_emp_length(df["emp_length"])
    for col in config.NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")

    # 3. Categorical standardisation.
    for col in ["grade", "sub_grade", "home_ownership", "verification_status", "purpose",
                "addr_state", "application_type", "loan_status"]:
        df[col] = df[col].astype("string").str.strip()
    df["grade"] = df["grade"].str.upper()
    df["addr_state"] = df["addr_state"].str.upper()
    df["home_ownership"] = df["home_ownership"].str.upper().replace({"ANY": "OTHER", "NONE": "OTHER"})

    # 4. Invalid values -> missing (impossible values are not "extreme", they are errors).
    df.loc[(df["dti"] < 0) | (df["dti"] > 100), "dti"] = np.nan
    df.loc[df["annual_inc"] <= 0, "annual_inc"] = np.nan
    df.loc[(df["revol_util"] < 0) | (df["revol_util"] > 150), "revol_util"] = np.nan

    # 5. Rows missing essential fields cannot be repaired.
    df = df.dropna(subset=["id", "loan_amnt", "loan_status", "issue_d", "grade", "int_rate", "term"])

    # 6. Missing data: flag first (missingness can be informative), then impute with the median.
    df["emp_length_missing"] = df["emp_length"].isna().astype("int8")
    for col in ["emp_length", "annual_inc", "dti", "revol_util", "revol_bal", "open_acc",
                "total_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec"]:
        df[col] = df[col].fillna(df[col].median())
    df["earliest_cr_line"] = df["earliest_cr_line"].fillna(df["earliest_cr_line"].median())
    df["verification_status"] = df["verification_status"].fillna("Unknown")
    df["home_ownership"] = df["home_ownership"].fillna("OTHER")

    # 7. Outliers: winsorise heavy-tailed money fields at the 1st/99th percentile.
    for col in ["annual_inc", "revol_bal"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    return df.reset_index(drop=True)
