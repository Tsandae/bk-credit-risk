## Module 4: Predictive model, explainability and fairness (AIA6550)

Trains logistic regression, random forest and LightGBM on the Module 3 curated table (`data/curated/loans_curated.parquet`),
with a time-based split (train 2007-2013, validate 2014, test 2015 used once), MLflow tracking and registry, SHAP, counterfactuals,
sensitivity tests, Fairlearn fairness audit with ThresholdOptimizer mitigation, a Model Card and a FastAPI `/predict` service.

| Brief requirement | Where |
|---|---|
| Training, tuning, MLflow tracking and Model Registry | `src/module4/run_all.py`, `src/module4/model.py` |
| Serialized model (joblib) | `models/module4/credit_risk_model.joblib` |
| FastAPI `/predict` endpoint | `src/module4/api.py` |
| SHAP analysis notebook (global + local) | `notebooks/module4_shap_analysis.ipynb` |
| Counterfactuals and sensitivity analysis | `src/module4/scenarios.py`, `reports/module4/counterfactuals.csv`, `sensitivity.csv` |
| Fairness Metrics Report | `reports/module4/fairness_metrics_report.md` |
| Model Card | `docs/module4/model_card.md` |
| MLflow evidence | `reports/module4/mlflow_runs.csv` and screenshots in `reports/module4/mlflow/` |
| Tests | `tests/module4/` |

```bash
pip install -r requirements-pipeline.txt -r requirements-module4.txt
python -m src.module4.run_all                                  # train, evaluate, explain, audit, register
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db    # inspect runs at http://127.0.0.1:5000
uvicorn src.module4.api:app --reload                           # API docs at http://127.0.0.1:8000/docs
python -m pytest tests/module4 -q
```

Example request: `POST /predict` with the fields shown in `/docs` returns the default probability, risk tier and the three largest SHAP reasons.
`python -m src.module4.run_all --demo --quick` runs on synthetic data for a dry run only; never quote those numbers.
