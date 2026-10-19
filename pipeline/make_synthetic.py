"""Generate a noisy synthetic stand-in for LendingClub loan.csv (same columns, same quirks).

Used only when the real file is not available, and for unit tests. Quirks injected on purpose:
duplicate rows, '%' strings, ' 36 months' terms, 'n/a' employment length, missing / impossible values,
e-mails and phone numbers hidden in free text, and a skewed group mix (to exercise the bias suite).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from . import config

STATES = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MA", "WA", "AZ", "CO", "MN", "MO",
          "MD", "TN", "WY", "VT", "ND", "AK"]
STATE_W = np.array([14, 8, 8, 7, 4, 4, 4, 3.5, 3, 3, 3, 3, 2.5, 2.5, 2.5, 2, 2, 2, 2, 2, 0.15, 0.1, 0.08, 0.05])
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase", "small_business", "car"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def generate(n: int = 20_000, seed: int = 42, dup_rate: float = 0.025) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)
    jobs = [fake.job() for _ in range(300)]
    texts = []
    for _ in range(200):
        t = fake.sentence(nb_words=12)
        r = rng.random()
        if r < 0.10:
            t += f" Contact me at {fake.email()}"
        elif r < 0.16:
            t += f" Call {fake.numerify('###-###-####')}"
        texts.append(t)

    grade_idx = rng.choice(7, size=n, p=[.17, .30, .29, .15, .06, .02, .01])
    base_rate = np.array([7.2, 11, 14, 17.5, 21, 25, 28])[grade_idx]
    int_rate = np.clip(base_rate + rng.normal(0, 1.2, n), 5.4, 30.99).round(2)
    loan_amnt = np.clip(np.round(rng.lognormal(9.4, 0.55, n) / 25) * 25, 1000, 35000)
    term = rng.choice([36, 60], size=n, p=[.75, .25])
    r_m = int_rate / 1200
    installment = (loan_amnt * r_m / (1 - (1 + r_m) ** (-term))).round(2)
    annual_inc = rng.lognormal(np.log(65000), 0.5, n).round(0)
    dti = np.clip(rng.gamma(4.0, 4.2, n), 0, 60).round(2)

    year = rng.choice([2012, 2013, 2014, 2015], size=n, p=[.10, .20, .35, .35])
    month = rng.integers(0, 12, n)
    issue_d = [f"{MONTHS[m]}-{y}" for m, y in zip(month, year)]
    cr_year = year - rng.integers(3, 36, n)
    earliest = [f"{MONTHS[m]}-{y}" for m, y in zip(rng.integers(0, 12, n), cr_year)]

    logit = (-2.7 + 0.45 * grade_idx + 0.025 * (dti - 17) - 0.4 * (np.log(annual_inc) - 11.1)
             + 0.2 * (term == 60) + rng.normal(0, 0.4, n))
    p_bad = 1 / (1 + np.exp(-logit))
    u = rng.random(n)
    status = np.where(u < p_bad, "bad", "Fully Paid").astype(object)
    sub = rng.random(n)
    status = np.where(status == "bad",
                      np.where(sub < .80, "Charged Off", np.where(sub < .85, "Default", "Late (31-120 days)")), status)
    current = (year >= 2015) & (rng.random(n) < 0.55)
    status = np.where(current, "Current", status)
    status = np.where(rng.random(n) < 0.015, "In Grace Period", status)

    emp_length = rng.choice(["< 1 year"] + [f"{i} years" for i in range(1, 10)] + ["10+ years", "n/a"], size=n,
                            p=[.08] + [.07] * 9 + [.28, .01]).astype(object)
    emp_length[rng.random(n) < 0.05] = None

    df = pd.DataFrame({
        "id": np.arange(10_000_000, 10_000_000 + n),
        "member_id": rng.integers(1_000_000, 90_000_000, n),
        "loan_amnt": loan_amnt, "funded_amnt": loan_amnt,
        "term": [f" {t} months" for t in term],
        "int_rate": int_rate, "installment": installment,
        "grade": np.array(list("ABCDEFG"))[grade_idx],
        "sub_grade": [f"{g}{rng.integers(1, 6)}" for g in np.array(list("ABCDEFG"))[grade_idx]],
        "emp_title": rng.choice(jobs, n),
        "emp_length": emp_length,
        "home_ownership": rng.choice(["MORTGAGE", "RENT", "OWN", "ANY"], size=n, p=[.50, .40, .095, .005]),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], size=n, p=[.33, .37, .30]),
        "issue_d": issue_d, "loan_status": status,
        "purpose": rng.choice(PURPOSES, n, p=[.55, .20, .07, .08, .04, .03, .03]),
        "title": rng.choice(["Debt consolidation", "Credit card refinancing", "Home improvement", "Other"], n),
        "desc": rng.choice(texts, n),
        "zip_code": [f"{z}xx" for z in rng.integers(100, 999, n)],
        "addr_state": rng.choice(STATES, size=n, p=STATE_W / STATE_W.sum()),
        "dti": dti,
        "delinq_2yrs": rng.poisson(0.3, n), "earliest_cr_line": earliest,
        "inq_last_6mths": rng.poisson(0.8, n), "open_acc": rng.poisson(11, n),
        "pub_rec": rng.poisson(0.15, n),
        "revol_bal": rng.lognormal(9.4, 1.0, n).round(0),
        "revol_util": np.clip(rng.normal(55, 24, n), 0, 120).round(1),
        "total_acc": rng.poisson(25, n) + 2, "application_type": "INDIVIDUAL",
    })

    # --- injected noise ---
    pct = rng.random(n) < 0.015
    df["int_rate"] = df["int_rate"].astype(object)
    df.loc[pct, "int_rate"] = df.loc[pct, "int_rate"].map(lambda v: f"{v}%")
    df.loc[rng.random(n) < 0.005, "revol_util"] = np.nan
    df.loc[rng.random(n) < 0.0005, "annual_inc"] = np.nan
    df.loc[rng.random(n) < 0.0005, "annual_inc"] = 0
    df.loc[rng.random(n) < 0.003, "dti"] = -1
    df.loc[rng.random(n) < 0.001, "dti"] = 999
    df.loc[rng.random(n) < 0.06, "emp_title"] = np.nan
    df.loc[rng.random(n) < 0.03, "home_ownership"] = np.nan
    dups = df.sample(frac=dup_rate, random_state=seed)
    return pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20_000)
    ap.add_argument("--out", default=str(config.RAW_FILE))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate(a.rows).to_csv(out, index=False)
    print(f"wrote {out}")
