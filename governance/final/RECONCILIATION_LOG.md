# Cross-module reconciliation log (Final Project audit)

Purpose: before writing the integrated report, every module was checked against the others and against the files in the repository. This log lists what disagreed, which source was treated as correct and why, and what is still open. "Canonical source" means the number a reader can reproduce from the repository.

| # | Where the conflict is | What each source says | Resolution | Status |
|---|---|---|---|---|
| 1 | Dataset size: Module 4 text vs Module 3 lineage | Module 4 text: 267,774 loans. Module 3 lineage and curated file: 265,774 | 265,774 (reproducible from `reports/lineage.json` and the parquet) | Fixed in final report; correct the Module 4 text |
| 2 | Module 4 report text vs repository run | Text: AUC 0.689, KS 0.283, 36.5% of defaulters caught (2,866), LightGBM 150 trees / 31 leaves. Repository run (`reports/module4/results.json`, 26 Oct 2026, seed 42): AUC 0.687, KS 0.281, 36.2% (2,840), LightGBM 250 trees / 15 leaves | The repository run, because the live API, dashboard, tests and monitoring all use that saved model | Fixed in final report; Module 4 text was written from an earlier run |
| 3 | Fairness after mitigation | Module 4 text: income-quartile TPR gap stays 5.8 points ("nearly all gaps within five points"). Repository run: 4.3 points, all four group types within limits | Repository run (4.3). Wording changed to "all four within limits, at a cost" | Fixed |
| 4 | Income-quartile default rates | Module 2: 23.0% (lowest) and 13.6% (highest). Module 4 / results.json, full curated file: 26.8% and 17.0%; 2015 test: 31.7% and 21.0% | Module 4 values. The Module 2 figures cannot be reproduced from the curated file, so they probably came from a different base (e.g. the raw file) | Open: check the Module 2 base |
| 5 | Regional default rates | Module 4 text: 23.5% to 27.6%. results.json, full file: 20.5% to 22.5% | results.json | Open: the text range may be from the 2015 test set; not reproducible |
| 6 | Interest-rate-only AUC and ablation | Text: 0.675 and 0.681. results.json: 0.670 and 0.676 | results.json | Fixed |
| 7 | 2014 to 2015 "model drift" vs Module 3 bias results | Module 4 reads the AUC fall (0.702 to 0.687) as drift over time. Module 3 `year_shift` shows 2015 loans fell from 47.5% of raw rows to 11.6% of curated rows | Both are reported. The 2015 sample is a filtered subset (most likely loans still open at data extraction were excluded), so part of the 2014 to 2015 change may come from selection, not drift. Not proven | Open: confirm with loan_status counts by year |
| 8 | Orchestration tool | Final brief lists an Airflow DAG. The Module 3 pipeline uses a Prefect flow (`flows/credit_risk_flow.py`) | Described as Prefect throughout. No Airflow DAG exists | Fixed in wording |
| 9 | LIME | Module 4 text: "LIME was unavailable". Final brief requires LIME | LIME added (`dashboard/helpers.py: lime_explain`, dashboard page 6) and compared with SHAP | Fixed |
| 10 | Fairness mitigation vs what is deployed | Module 4 shows group-specific cut-offs fix the gaps. The live API and dashboard score with one cut-off | Stated plainly: mitigation is an analysis, not deployed. The monitoring dashboard therefore keeps reporting the bias breaches | Open: a governance decision |
| 11 | Module 2 objective 4 (10% lower expected loss) | Not assessable without Bank of Kigali cost data; mitigation roughly doubles flagged workload | Reported "not assessed"; conflict with mitigation stated | Open |
| 12 | Module 1 assumptions | Module 4 text notes the Module 1 assumptions "are inconsistent" and need checking with BK data | Carried forward as a limitation; no Module 1 number is used as evidence in the final report | Open |
| 13 | Module 1 risk register vs Module 3 findings | Register not available to this audit | Final report lists the risks the pipeline actually surfaced (under-represented groups, shifted verification mix, 2015 shrinkage) so they can be added to the register | Open: update the register |
| 14 | Dashboard vs model in the text | Dashboard numbers come from the same `results.json` and joblib as the report; a test checks live Fairlearn output equals the Module 4 numbers | No conflict | Verified |

How to re-check: `pytest tests/module5 tests/final` and `python -m src.final.monitor`.
