"""Final Project tests: monitoring, explanation store, API audit logging, retraining gate, new dashboard pages."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dashboard"))

from src.final import monitor  # noqa: E402
from src.final.explanation_store import ExplanationStore  # noqa: E402

APP = dict(loan_amnt=12000, term=36, int_rate=13.5, installment=407, grade="C", home_ownership="RENT", annual_inc=55000,
           verification_status="Verified", purpose="debt_consolidation", dti=18, revol_bal=9000, revol_util=45, credit_history_years=12)


def test_psi_is_zero_for_same_data_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert monitor.psi_numeric(a, a) < 0.001
    assert monitor.psi_numeric(a, a + 1.5) > monitor.PSI_LIMIT


def test_psi_categorical_detects_mix_change():
    ref = pd.Series(["A"] * 80 + ["B"] * 20)
    assert monitor.psi_categorical(ref, ref) < 0.001
    assert monitor.psi_categorical(ref, pd.Series(["A"] * 30 + ["B"] * 70)) > monitor.PSI_LIMIT


def test_explanation_store_chain_detects_tampering(tmp_path):
    s = ExplanationStore(tmp_path / "e.db")
    for i in range(3):
        s.record("m:v1", {"x": i}, 0.1 * i, "low", {"int_rate": 0.2, "term": -0.1})
    assert s.verify_chain() == (True, 3)
    rid = s.recent(1)[0]["request_id"]
    s.add_override(rid, "credit officer", "approve", "documented income not in model")
    assert s.verify_chain() == (True, 3)            # overrides are notes beside the record, not edits to it
    with sqlite3.connect(s.path) as c:
        c.execute("UPDATE explanations SET probability = 0.99 WHERE seq = 2")
    ok, n = s.verify_chain()
    assert ok is False and n == 1


def test_explanation_store_purge_and_export(tmp_path):
    s = ExplanationStore(tmp_path / "e.db")
    s.record("m", {}, 0.2, "low", {"a": 1.0})
    assert s.purge(retention_days=2555) == 0 and s.count() == 1
    s.export_csv(tmp_path / "out.csv")
    assert "request_id" in (tmp_path / "out.csv").read_text()


def test_api_writes_a_verifiable_explanation_record(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("EXPLANATION_DB", str(tmp_path / "api.db"))
    from src.module4.api import app
    r = TestClient(app).post("/predict", json=APP)
    assert r.status_code == 200
    s = ExplanationStore(tmp_path / "api.db")
    assert s.count() == 1 and s.verify_chain()[0]
    rec = s.recent(1)[0]
    assert rec["tier"] == r.json()["risk_tier"] and len(rec["shap"]) == 22


def test_api_works_without_explanation_store(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.delenv("EXPLANATION_DB", raising=False)
    from src.module4.api import app
    assert TestClient(app).post("/predict", json=APP).status_code == 200


from src.module4 import config  # noqa: E402

needs_data = pytest.mark.skipif(not config.CURATED_FILE.exists(), reason="curated Module 3 file is not in the repository (privacy/size); run locally")


@pytest.fixture(scope="module")
def report():
    return monitor.run(write=False)


@needs_data
def test_monitoring_matches_module4_results(report):
    r4 = json.loads((ROOT / "reports" / "module4" / "results.json").read_text())
    w = report["windows"]["2015 (test)"]
    assert w["auc"] == pytest.approx(r4["models"]["lightgbm"]["test"]["roc_auc"], abs=1e-6)
    f = w["fairness"]["income_quartile"]
    assert f["approval_ratio"] == pytest.approx(r4["fairness_before"]["income_quartile"]["approval_ratio"], abs=1e-6)


@needs_data
def test_monitoring_raises_the_known_bias_and_calibration_alerts(report):
    w = report["windows"]["2015 (test)"]
    kinds = {(a["type"], a["item"]) for a in w["alerts"]}
    assert ("bias drift", "income_quartile: approval ratio") in kinds
    assert ("calibration drift", "high tier") in kinds
    assert w["action"].startswith("REVIEW")          # second breach in a row (2014 then 2015)


@needs_data
def test_no_input_has_drifted_beyond_the_limit(report):
    assert all(v < monitor.PSI_LIMIT for v in report["windows"]["2015 (test)"]["psi"].values())


@pytest.mark.parametrize("page", ["6. Explanations (SHAP & LIME)", "7. Monitoring"])
def test_new_dashboard_pages_render(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_monitoring_app_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "dashboard" / "monitoring_app.py"), default_timeout=90).run()
    assert not at.exception, at.exception
