import numpy as np
import pandas as pd

from pipeline.clean import clean, parse_emp_length, parse_pct, parse_term


def test_parse_term():
    assert parse_term(pd.Series([" 36 months", " 60 months"])).tolist() == [36, 60]


def test_parse_pct_handles_strings_and_floats():
    out = parse_pct(pd.Series(["13.5%", 7.2, "bad"]))
    assert out.iloc[0] == 13.5 and out.iloc[1] == 7.2 and np.isnan(out.iloc[2])


def test_parse_emp_length():
    out = parse_emp_length(pd.Series(["< 1 year", "3 years", "10+ years", "n/a", None]))
    assert out.iloc[:3].tolist() == [0, 3, 10]
    assert out.iloc[3:].isna().all()


def test_clean_removes_duplicates_and_invalid_values(raw_df):
    out = clean(raw_df)
    assert out["id"].is_unique
    assert len(out) < len(raw_df)
    assert out["dti"].between(0, 100).all()
    assert (out["annual_inc"] > 0).all()
    assert out["revol_util"].between(0, 150).all()
    assert out[["annual_inc", "dti", "revol_util", "emp_length"]].isna().sum().sum() == 0


def test_clean_converts_types(raw_df):
    out = clean(raw_df)
    assert out["term"].isin([36, 60]).all()
    assert pd.api.types.is_numeric_dtype(out["int_rate"])
    assert pd.api.types.is_datetime64_any_dtype(out["issue_d"])
    assert set(out["emp_length_missing"].unique()) <= {0, 1}
