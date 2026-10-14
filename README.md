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
