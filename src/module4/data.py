"""Load the Module 3 curated file, split by issue year and add the fairness group columns."""
import numpy as np
import pandas as pd

from . import config


def load(path=None) -> pd.DataFrame:
    df = pd.read_parquet(path or config.CURATED_FILE)
    for c in df.columns:                      # arrow/'string' dtypes -> plain numpy/object for scikit-learn
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float64") if df[c].isna().any() or str(df[c].dtype).endswith("[pyarrow]") else df[c]
        elif df[c].dtype != "datetime64[ns]":
            df[c] = df[c].astype(object)
    return df


def split(df: pd.DataFrame):
    y = df["issue_year"]
    return (df[y <= config.TRAIN_LAST_YEAR].copy(), df[y == config.VAL_YEAR].copy(), df[y == config.TEST_YEAR].copy())


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute engineered ratios (used by what-if / sensitivity scenarios and the API)."""
    df = df.copy()
    income = df["annual_inc"].clip(lower=1)
    df["loan_to_income"] = df["loan_amnt"] / income
    df["installment_to_income"] = df["installment"] * 12 / income
    return df


def amortised_payment(amount, annual_rate_pct, months):
    r = np.asarray(annual_rate_pct, dtype=float) / 1200.0
    n = np.asarray(months, dtype=float)
    amount = np.asarray(amount, dtype=float)
    return np.where(r > 0, amount * r / (1 - (1 + r) ** (-n)), amount / n)


def add_groups(df: pd.DataFrame, income_edges) -> pd.DataFrame:
    """Fairness groups from Module 2: income quartile, home ownership, employment length, region."""
    df = df.copy()
    labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    df["income_quartile"] = pd.cut(df["annual_inc"], bins=income_edges, labels=labels, include_lowest=True).astype(object)
    e = df["emp_length"]
    # Module 3 median-imputes missing employment length and sets emp_length_missing = 1, so use that flag.
    not_reported = e.isna() | (df["emp_length_missing"] == 1)
    df["employment_length_group"] = np.select(
        [not_reported, e <= 2, e <= 9], ["Not reported", "0-2 years", "3-9 years"], default="10+ years")
    return df


def income_edges(train: pd.DataFrame):
    q = train["annual_inc"].quantile([0, .25, .5, .75, 1]).to_numpy().copy()
    q[0], q[-1] = -np.inf, np.inf
    return q


def prepare(path=None, demo=False):
    """train, val, test frames with fairness groups (same preparation as run_all)."""
    if demo:
        from .demo_data import make_demo_curated
        df = make_demo_curated(60000)
    else:
        df = load(path)
    tr, va, te = split(df)
    edges = income_edges(tr)
    return tuple(add_groups(d, edges) for d in (tr, va, te))
