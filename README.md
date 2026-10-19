# BK Credit Risk Intelligence

Credit risk early-warning and decision-support project for Bank of Kigali retail and digital lending. Built on the Module 1 Vision Document and planned in the Module 2 Project Overview Report.

**Data:** LendingClub loan data (Kaggle). It is US peer-to-peer lending data used as a proxy; it is **not** Bank of Kigali data. Raw data is never committed (see `.gitignore`); fetch it with the ingestion script in Sprint 1 and version it with DVC.

## Structure
See the Repository Structure in the Module 1 Vision Document. Key folders: `docs/` (ethics charter, datasheet, model card, decision records), `data/` (DVC-tracked), `notebooks/`, `src/`, `pipelines/`, `models/`, `tests/`, `config/` (feature whitelist and leakage list), `project/` (sprint tickets).

## Setup
```bash
conda env create -f environment.yml
conda activate bk-credit-risk
pytest
```

## Ethical AI Charter (summary)
The model supports credit officers and never decides alone; no applicant is denied banking services solely because of a score; reasons are documented; fairness testing and independent validation precede any operational use. See `docs/ethics/charter.md`.

## Module 3: Data pipeline 

Reproducible, governed pipeline that turns the raw LendingClub `loan.csv` into the curated table used for Module 4 modelling.

```
loan.csv -> ingest -> clean -> privacy gate -> transform -> integrate -> validate (Great Expectations) -> bias audit -> publish
```

| Requirement | Where |
|---|---|
| Orchestration DAG (Prefect) | `flows/credit_risk_flow.py` |
| Validation suite (Great Expectations, 23 checks) | `pipeline/validate.py` |
| Unit tests (22, pytest) | `tests/module3/` |
| Containerisation | `Dockerfile.pipeline` |
| Bias detection (Fairlearn) | `pipeline/bias.py` |
| Anonymisation (HMAC pseudonyms, PII scrub, k-anonymity k=5) | `pipeline/anonymize.py` |
| Governance (access control, retention) | `governance/policy.yaml`, `pipeline/governance.py` |
| Privacy audit log (hash-chained) | `pipeline/audit.py` -> `logs/privacy_audit.log` |

Run from the repository root:

```bash
pip install -r requirements.txt faker        # or: pip install -r requirements-pipeline.txt
# place the raw file at data/raw/loan.csv  (never commit it)
export ANON_SALT="<long random secret>"      # Windows PowerShell: $env:ANON_SALT = "..."
python -m flows.credit_risk_flow
python scripts/make_report_images.py
pytest tests/module3 -q
```

Outputs: `data/curated/loans_curated.parquet` and `reports/` (validation, bias, lineage, summary). The audit log is written to `logs/privacy_audit.log`.

