# Model Card - BK Credit-Risk Default Model (Module 4)

## Model details
- **Task:** binary classification, probability that a funded loan defaults (charged off / default / late 31-120 days = 1; fully paid = 0), estimated at application time.
- **Model:** lightgbm with isotonic probability calibration and three risk tiers (high = top 20%, medium = next 30%, low = bottom 50%).
- **Hyperparameters:** {"num_leaves": 15, "n_estimators": 250, "min_child_samples": 100, "learning_rate": 0.03}
- **Model choice:** lightgbm is the Module 2 main model; kept unless its 2014 validation AUC is more than 0.005 below the best model. Validation AUCs: logistic_regression 0.7011, random_forest 0.6993, lightgbm 0.7020
- **Version:** `credit_risk_model` v1, alias `staging`; trained 2026-10-26T10:18:26.
- **Library versions and parameters** are recorded in the MLflow runs (`mlflow ui`).

## Intended use
- Decision support for credit-risk **review** at Bank of Kigali: prioritise applications for analyst review and explain why (top-3 SHAP reasons).
- **Not** for automated approval or rejection, pricing, or use on BK borrowers before retraining and recalibration on BK data.

## Training and evaluation data
- Source: Module 3 curated LendingClub table (`data/curated/loans_curated.parquet`), 265,774 resolved loans issued 2007-2015, overall default rate 21.8%.
- Time-based split: train 2007-2013 (160,785), validation 2014 (74,154), test 2015 (30,835; used once).
- 22 inputs (18 numeric, 4 categorical). No post-loan fields; preprocessing fitted on training data only.

## Performance (2015 test set)
| Metric | Value | Module 2 target |
| --- | --- | --- |
| ROC-AUC | 0.687 (2014 validation 0.702) | >= 0.7 |
| KS | 0.281 | >= 0.25 |
| Defaulters in top 20% | 34.2% | >= 40% |
| Precision / recall / F1 at operating cutoff | 0.429 / 0.362 / 0.392 | - |
| Tier calibration (relative error) | low -3.4%, medium +1.3%, high +12.0% | within +/-10% |

## Fairness (see `reports/module4/fairness_metrics_report.md`)
- Groups: income quartile, home ownership, employment length, region (indirect; no protected attributes in the data).
- Metrics: approval (disparate impact) ratio, TPR and FPR gaps, demographic parity difference; before and after Fairlearn ThresholdOptimizer.
- Groups within all limits before mitigation: 0 of 4; after: 4 of 4. Mitigation raises review workload; governance sign-off is required.

## Explainability
- Global SHAP: top features are int_rate, term, annual_inc, dti, loan_to_income.
- Local: every API response returns the three largest SHAP contributions. Counterfactual what-ifs (term, amount) and sensitivity tests are in `reports/module4/`.

## Limitations and risks
- US LendingClub data stand in for BK data; retrain and recalibrate on BK loans before any use.
- Only funded loans are observed (selection bias), and performance can shift over time: compare the 2014 validation and 2015 test AUC above.
- Interest rate and grade are set by the lender, so the model partly repeats an earlier lending judgement (AUC without them: 0.676).
- Moderate discrimination (AUC about 0.69): use to support, not replace, human review.

## Monitoring
Population stability index (alert above 0.2), AUC, KS, top-20% capture, calibration by tier and group, approval ratios, TPR/FPR gaps, out-of-range inputs and manual overrides; two consecutive breaches trigger review or retraining.

## References
Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**.
