"""API tests: /predict contract, validation errors and the three SHAP reasons."""
import pytest

from src.module4 import config  # noqa: F401


def test_api_predict_contract(fitted, tmp_path, monkeypatch):
    import joblib
    from fastapi.testclient import TestClient
    from src.module4 import api
    path = tmp_path / "m.joblib"
    joblib.dump(fitted, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    api.bundle.cache_clear()
    client = TestClient(api.app)
    body = dict(loan_amnt=12000, term=60, int_rate=22.5, installment=333.0, grade="E", home_ownership="RENT",
                annual_inc=30000, verification_status="Verified", purpose="debt_consolidation", dti=28,
                revol_bal=9000, revol_util=80, credit_history_years=6)
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert 0 <= j["default_probability"] <= 1 and j["risk_tier"] in {"low", "medium", "high"}
    assert len(j["top_reasons"]) == 3
    assert client.post("/predict", json={**body, "annual_inc": -5}).status_code == 422
    api.bundle.cache_clear()
