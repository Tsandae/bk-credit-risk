"""FastAPI service: POST /predict -> default probability, risk tier and top-3 SHAP reasons.

Run from the repository root:   uvicorn src.module4.api:app --reload
Docs / try it:                  http://127.0.0.1:8000/docs
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .data import derive
from .explain import Explainer
from .model import feature_frame

app = FastAPI(title="BK Credit-Risk Model API", version="1.0.0",
              description="Decision support for credit review. Not for automated approval or rejection.")


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0, examples=[12000])
    term: int = Field(..., examples=[36], description="36 or 60 months")
    int_rate: float = Field(..., ge=0, le=60, examples=[13.5], description="annual interest rate, percent")
    installment: float = Field(..., gt=0, examples=[407])
    grade: str = Field(..., examples=["C"])
    home_ownership: str = Field(..., examples=["RENT"])
    annual_inc: float = Field(..., gt=0, examples=[55000])
    verification_status: str = Field(..., examples=["Verified"])
    purpose: str = Field(..., examples=["debt_consolidation"])
    dti: float = Field(..., ge=0, examples=[18.0])
    delinq_2yrs: float = Field(0, ge=0)
    inq_last_6mths: float = Field(0, ge=0)
    open_acc: float = Field(10, ge=0)
    pub_rec: float = Field(0, ge=0)
    revol_bal: float = Field(..., ge=0, examples=[9000])
    revol_util: float = Field(..., ge=0, examples=[45.0], description="percent")
    total_acc: float = Field(20, ge=0)
    credit_history_years: float = Field(..., ge=0, examples=[12])
    emp_length: Optional[float] = Field(None, ge=0, le=10, description="years; omit if not reported")


class Reason(BaseModel):
    feature: str
    value: object
    effect: str
    contribution: float


class Prediction(BaseModel):
    default_probability: float
    risk_tier: str
    top_reasons: List[Reason]
    model_version: str
    note: str = "Decision support only; a human reviewer makes the credit decision."


@lru_cache(maxsize=1)
def bundle():
    path = Path(os.environ.get("MODEL_PATH", config.MODEL_FILE))
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}. Run: python -m src.module4.run_all")
    model = joblib.load(path)
    return model, Explainer(model)


def to_frame(app_in: LoanApplication) -> pd.DataFrame:
    row = app_in.model_dump()
    row["emp_length_missing"] = 1 if row["emp_length"] is None else 0
    df = derive(pd.DataFrame([row]))
    df["emp_length"] = df["emp_length"].astype("float64")
    return feature_frame(df)


@app.get("/health")
def health():
    try:
        bundle()
        return {"status": "ok"}
    except FileNotFoundError as e:
        return {"status": "model not found", "detail": str(e)}


@app.post("/predict", response_model=Prediction)
def predict(application: LoanApplication):
    try:
        model, expl = bundle()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    X = to_frame(application)
    raw = model.raw(X)
    return Prediction(default_probability=round(float(model.calibrator.predict(raw)[0]), 4),
                      risk_tier=str(model.tier(raw)[0]), top_reasons=expl.reasons(X, k=3),
                      model_version=f"{config.REGISTERED_MODEL}:{model.name}")
