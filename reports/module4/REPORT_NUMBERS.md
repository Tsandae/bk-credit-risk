# Numbers for the Module 4 report (generated from the last run)

Generated 2026-10-26T10:18:26 | selected model: **lightgbm** | settings: `{"num_leaves": 15, "n_estimators": 250, "min_child_samples": 100, "learning_rate": 0.03}`

- Model choice: lightgbm is the Module 2 main model; kept unless its 2014 validation AUC is more than 0.005 below the best model. Validation AUCs: logistic_regression 0.7011, random_forest 0.6993, lightgbm 0.7020

## Section 2 - data and bias
- Rows 265,774; columns 34; overall default rate 21.8%
- Income-quartile default rate, full file: {"Q1 (lowest)": 0.2683, "Q2": 0.2359, "Q3": 0.2031, "Q4 (highest)": 0.1701}
- Income-quartile default rate, 2015 test: {"Q1 (lowest)": 0.3166, "Q2": 0.2661, "Q3": 0.2394, "Q4 (highest)": 0.2095}
- Region default rates, full file: min 20.5%, max 22.5%
- Home-ownership group sizes: {"MORTGAGE": 130359, "RENT": 111925, "OWN": 23305, "OTHER": 185}
- AUC of interest rate alone (2015): 0.670; full model 0.687; without grade and interest rate 0.676

## Section 3 - split
- Train 160,785 (19.1% default); validation 74,154 (26.2%); test 30,835 (25.5%)
- Baseline (predict all repaid): accuracy 0.745, recall 0, AUC 0.50

## Section 5 - Table 1 (2015 test)
| Model | AUC | KS | Top-20% capture | Precision | Recall | F1 | Accuracy | 2014 val AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.688 | 0.276 | 34.5% | 0.434 | 0.358 | 0.392 | 0.718 | 0.701 |
| random_forest | 0.682 | 0.272 | 33.7% | 0.424 | 0.368 | 0.394 | 0.712 | 0.699 |
| lightgbm | 0.687 | 0.281 | 34.2% | 0.429 | 0.362 | 0.392 | 0.715 | 0.702 |

- Operating cutoff flagged 21.5% of 2015 loans: caught 2,840 of 7,851 defaulters (recall 36.2%); wrongly flagged 3,787 repaid loans
- Tier calibration (relative error): low -3.4%, medium +1.3%, high +12.0%; deciles -13.0% to +16.8%
- Default rate by tier (actual): low 14.9%, medium 30.8%, high 42.9%
- Top decile predicted 54.5% vs actual 46.7%

## Table 2 - status against Module 2 objectives
- ROC-AUC >= 0.7: 0.687 -> Not met
- KS >= 0.25: 0.281 -> Met
- Top-20% capture >= 40%: 34.2% -> Not met
- Tier calibration within +/-10%: Not met (see tiers above)

## Section 6 - explainability
- Global SHAP (mean |SHAP|), top 8: int_rate 0.451, term 0.186, annual_inc 0.175, dti 0.108, loan_to_income 0.094, revol_util 0.062, open_acc 0.057, total_acc 0.047
- highest_risk: probability 0.733; reasons: int_rate=25.99 (raises risk); term=60 (raises risk); annual_inc=25000.0 (raises risk)
- medium_risk: probability 0.235; reasons: annual_inc=114000.0 (lowers risk); dti=28.26 (raises risk); inq_last_6mths=3.0 (raises risk)
- lowest_risk: probability 0.000; reasons: int_rate=5.32 (lowers risk); revol_util=4.4 (lowers risk); annual_inc=250000.0 (lowers risk)

### Counterfactuals (copy the sentence that fits)
| Applicant | Scenario | Term | Amount | Instalment | Default prob. | Tier |
| --- | --- | --- | --- | --- | --- | --- |
| 9372 | as applied | 60 | 21,700 | 565 | 0.442 | high |
| 9372 | term 60 -> 36 months | 36 | 21,700 | 798 | 0.325 | medium |
| 9372 | amount -10% | 60 | 19,530 | 509 | 0.427 | high |
| 9372 | amount -20% | 60 | 17,360 | 452 | 0.400 | high |
| 11875 | as applied | 60 | 20,800 | 538 | 0.490 | high |
| 11875 | term 60 -> 36 months | 36 | 20,800 | 761 | 0.383 | high |
| 11875 | amount -10% | 60 | 18,720 | 484 | 0.461 | high |
| 11875 | amount -20% | 60 | 16,640 | 430 | 0.442 | high |
| 7412 | as applied | 60 | 35,000 | 912 | 0.513 | high |
| 7412 | term 60 -> 36 months | 36 | 35,000 | 1,286 | 0.383 | medium |
| 7412 | amount -10% | 60 | 31,500 | 820 | 0.497 | high |
| 7412 | amount -20% | 60 | 28,000 | 729 | 0.490 | high |

### Sensitivity (percentage-point change in average predicted default)
- income -10%: +1.19
- income +10%: -1.12
- interest rate +10%: +2.20
- debt-to-income +10%: +0.36
- loan amount +10%: +0.52
- credit use +10%: +0.20
- 5% random noise: AUC 0.687 -> 0.685

### Table 3 - fairness (before / after)
| Group | Approval ratio | TPR gap | FPR gap |
| --- | --- | --- | --- |
| income quartile, before | 0.750 | 29.4 | 17.6 |
| income quartile, after | 0.960 | 4.3 | 2.2 |
| home ownership, before | 0.935 | 5.2 | 4.7 |
| home ownership, after | 0.944 | 3.6 | 1.4 |
| employment length group, before | 0.781 | 15.4 | 15.9 |
| employment length group, after | 0.937 | 4.1 | 2.3 |
| region, before | 0.952 | 5.5 | 2.9 |
| region, after | 0.962 | 4.0 | 2.8 |

- Mitigation cost: flagged share 21.5% -> 51%, 42%, 42%, 48%; precision 42.9% -> 35%, 37%, 35%, 36%; accuracy 71.5% -> 59%, 63%, 62%, 61%

### Calibration by group (relative error, 2015 test) - largest overstatements
- home ownership / MORTGAGE: predicted 24.8% vs actual 20.9% (+18.6%), Brier 0.157
- employment length group / 10+ years: predicted 25.5% vs actual 21.7% (+17.4%), Brier 0.160
- income quartile / Q2: predicted 29.5% vs actual 26.6% (+10.9%), Brier 0.182
- region / Midwest: predicted 27.5% vs actual 25.1% (+9.7%), Brier 0.174
- home ownership / OWN: predicted 27.5% vs actual 25.2% (+9.2%), Brier 0.182
