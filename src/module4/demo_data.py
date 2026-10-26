"""Synthetic stand-in with the curated schema. ONLY for tests and dry runs - never for reported results."""
import numpy as np
import pandas as pd

from . import config

STATES = {"NY": "Northeast", "CA": "West", "TX": "South", "IL": "Midwest", "FL": "South", "WA": "West"}


def make_demo_curated(n=12000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year = rng.choice(np.arange(2007, 2016), n, p=[.01, .02, .03, .05, .1, .18, .25, .26, .10])
    grade_i = rng.choice(7, n, p=[.17, .30, .29, .15, .06, .02, .01])
    int_rate = np.clip(np.array([7.2, 11, 14, 17.5, 21, 25, 28])[grade_i] + rng.normal(0, 1.2, n), 5.4, 30.9)
    term = rng.choice([36, 60], n, p=[.72, .28])
    amt = np.clip(rng.lognormal(9.5, .5, n), 1000, 40000).round(-2)
    inc = np.clip(rng.lognormal(11.0, .55, n), 12000, 400000).round(-2)
    r = int_rate / 1200
    inst = amt * r / (1 - (1 + r) ** (-term))
    dti = np.clip(rng.normal(17, 8, n), 0, 45)
    revol_util = np.clip(rng.normal(52, 25, n), 0, 100)
    emp = rng.choice(list(range(0, 11)) + [np.nan], n)
    st = rng.choice(list(STATES), n)
    z = (-2.9 + 0.11 * (int_rate - 13) + 0.45 * (term == 60) + 0.02 * (dti - 17) - 0.6 * np.log(inc / 60000)
         + 0.004 * (revol_util - 50) + 0.35 * (year >= 2014) + rng.normal(0, 0.6, n))
    df = pd.DataFrame({
        "loan_amnt": amt, "funded_amnt": amt, "term": term.astype("int16"), "int_rate": int_rate.round(2),
        "installment": inst.round(2), "grade": np.array(list("ABCDEFG"))[grade_i], "sub_grade": "X1",
        "emp_length": emp, "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[.45, .42, .12, .01]),
        "annual_inc": inc, "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement", "other", "small_business"], n),
        "addr_state": st, "dti": dti, "delinq_2yrs": rng.poisson(0.2, n), "inq_last_6mths": rng.poisson(0.8, n),
        "open_acc": rng.poisson(11, n), "pub_rec": rng.poisson(0.1, n), "revol_bal": rng.lognormal(9, 1, n).round(0),
        "revol_util": revol_util, "total_acc": rng.poisson(25, n), "application_type": "Individual",
        "loan_key": [f"{i:016x}" for i in range(n)], "zip3": "OTHER", "income_band": "40-80k",
        "issue_year": year.astype("int16"), "grade_num": grade_i, "log_annual_inc": np.log1p(inc),
        config.TARGET: (rng.random(n) < 1 / (1 + np.exp(-z))).astype("int8"),
    })
    df["emp_length_missing"] = df["emp_length"].isna().astype("int8")
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df["installment_to_income"] = df["installment"] * 12 / df["annual_inc"]
    df["credit_history_years"] = rng.uniform(2, 30, n)
    df["region"] = df["addr_state"].map(STATES)
    return df
