"""Dashboard tests: helper maths, fairness numbers equal Module 4's, and every page renders without errors."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
pytest.importorskip("fairlearn")
pytest.importorskip("lightgbm")
pytest.importorskip("shap")

import helpers as H  # noqa: E402

pytestmark = pytest.mark.skipif(not (H.DATA / "test_scored.parquet").exists(), reason="run dashboard/build_dashboard_data.py first")


def test_amortised_payment_known_value():
    assert H.amortised_payment(10000, 12.0, 36) == pytest.approx(332.14, abs=0.01)


def test_live_fairlearn_view_matches_module4_results():
    R, S, meta = H.load_results(), H.load_scored(), H.load_meta()
    for attr in H.GROUP_LABELS:
        _, s = H.fairness_view(S, attr, meta["cut_high"])
        b = R["fairness_before"][attr]
        assert s["approval_ratio"] == pytest.approx(b["approval_ratio"], abs=1e-3)
        assert s["tpr_gap_pp"] == pytest.approx(b["tpr_gap_pp"], abs=0.05)
        assert s["fpr_gap_pp"] == pytest.approx(b["fpr_gap_pp"], abs=0.05)


def test_small_groups_are_excluded():
    import pandas as pd
    df = pd.DataFrame({"raw": [0.1, 0.9] * 300 + [0.9, 0.9], "default": [0, 1] * 300 + [1, 1], "g": ["a"] * 600 + ["tiny"] * 2})
    t, _ = H.fairness_view(df, "g", 0.5)
    assert t.set_index("group").loc["tiny", "included"] == False  # noqa: E712


def test_scoring_orders_risk_sensibly():
    import joblib
    from src.module4 import config
    model = joblib.load(config.MODEL_FILE)
    low = dict(loan_amnt=8000, term=36, int_rate=5.3, grade="A", annual_inc=250000, dti=8, revol_util=4.4, revol_bal=3000,
               credit_history_years=22, home_ownership="MORTGAGE", verification_status="Not Verified", purpose="credit_card",
               delinq_2yrs=0, inq_last_6mths=0, open_acc=12, pub_rec=0, total_acc=30, emp_length=10)
    high = dict(low, loan_amnt=21700, term=60, int_rate=26.0, grade="F", annual_inc=25000, dti=29.7, revol_util=80)
    a, b = H.score(model, H.build_frame(model, low)), H.score(model, H.build_frame(model, high))
    assert a["probability"] < 0.1 < 0.4 < b["probability"]
    assert a["tier"] == "low" and b["tier"] == "high"


@pytest.mark.parametrize("page", ["1. Overview", "2. Score an applicant & what-if", "3. What drives predictions",
                                  "4. Ethical compliance (fairness)", "5. Limits & transparency"])
def test_every_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
