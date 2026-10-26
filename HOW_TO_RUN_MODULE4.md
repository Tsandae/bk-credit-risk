# Module 4 - how to run it (Windows, PowerShell)

1. Copy the contents of this folder into your repository folder (the one that contains `src`, `tests`, `notebooks`, `data`). Say "yes" to merge folders. Nothing of Module 2 or 3 is overwritten, except `.github/workflows/ci.yml` (replace it).
2. Open the repo folder in VS Code, open a terminal, activate your environment (`.venv\Scripts\activate`), then:
   `python -m pip install -r requirements-pipeline.txt -r requirements-module4.txt`
3. Check that `data\curated\loans_curated.parquet` exists (output of Module 3).
4. Train everything (about 5-15 minutes): `python -m src.module4.run_all`
   It writes: `models\module4\credit_risk_model.joblib`, `reports\module4\*` (tables, figures, fairness report, REPORT_NUMBERS.md) and `docs\module4\model_card.md`, and logs every run to MLflow (`mlflow.db`).
5. MLflow screenshots: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`, open http://127.0.0.1:5000, screenshot the experiment table, one run, and Models > credit_risk_model (version 1, alias staging). Save them in `reports\module4\mlflow\`.
6. SHAP notebook: open `notebooks\module4_shap_analysis.ipynb`, choose your .venv kernel, Run All, save.
7. API: `uvicorn src.module4.api:app --reload`, open http://127.0.0.1:8000/docs, try POST /predict, screenshot the response.
8. Tests: `python -m pytest tests/module4 -q`
9. Open `reports\module4\REPORT_NUMBERS.md` and update the report with the numbers from YOUR run (they can differ slightly from earlier drafts).
10. Upload to GitHub (same way as Module 3): `src/module4`, `tests/module4`, `notebooks/module4_shap_analysis.ipynb`, `models/module4`, `reports/module4`, `docs/module4`, `requirements-module4.txt`, the updated `.github/workflows/ci.yml`, and the Module 4 section of the README. Do NOT upload `mlflow.db`, `mlruns/`, `.venv/`.
