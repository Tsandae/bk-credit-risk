# Ethical AI Framework: Bank of Kigali credit-risk decision-support model

Status: course project (BAN6800 Final Project). Built on US LendingClub data as a stand-in for Bank of Kigali data. Not approved for use in credit decisions.

## Charter (Module 1)
Ethical AI Charter 
1. The model is going to support credit officers and does not make the final decision. 
2. No applicant will be refused only because of a model score. 
3. Predictions and their main reasons will be documented. 
4. Personal data will be protected and used according to privacy rules and bank policies. 
5. The model must be tested for fairness and checked by an independent person before it is used. 
6. The model will not be released if significant unexplained fairness or performance problems remain. 
7. The limitations of the LendingClub dataset will always be disclosed. 


## Principles, controls and evidence

| Principle | Control in this project | Evidence |
|---|---|---|
| Human oversight | The model estimates risk and never approves or declines. Every flag goes to a human reviewer, who can override and must record a reason. | API note on every response; `ExplanationStore.add_override`; dashboard page 5 |
| Fairness | Fairness checked on four indirect groups against Module 2 limits (approval ratio at least 0.80; TPR and FPR gaps at most 5 points); mitigation tested; continuous monitoring | `reports/module4/fairness_metrics_report.md`; `src/final/monitor.py`; dashboard pages 4 and 7 |
| Transparency | Plain-language reasons per prediction (SHAP top 3), LIME cross-check, model card, limitations statement | `docs/module4/model_card.md`; dashboard pages 3, 5, 6 |
| Privacy and data protection | Direct identifiers removed, HMAC pseudonyms, k-anonymity (k=5), role-based access, retention rules, hash-chained audit log; the explanation store holds no names | `governance/policy.yaml`; Module 3 `privacy_audit.log` |
| Accountability | Named owners, approval gate for any model change, versioned registry, tests in CI, incident log | `GOVERNANCE_STRUCTURE.md`; `src/final/retrain.py`; `.github/workflows` |
| Safety and robustness | Time-based validation, noise test (5% noise: AUC 0.687 to 0.685), drift and calibration monitoring | `reports/module4/results.json`; `monitor.py` |
| Honesty about limits | Goals missed are reported as missed; proxy-data caveat on every output | Report section 4; dashboard footer |

## Known tensions (stated, not hidden)
1. Fairness vs workload: group-specific cut-offs bring every group within limits but roughly double flagged applications (21.5% to 42-51%) and lower accuracy (71.5% to 59-63%).
2. Fairness vs legal exposure: using different cut-offs by group may itself be restricted. Legal review is required before any use.
3. Inputs overlap with fairness groups: income, housing status and employment length are model inputs and fairness groups. Different approval rates do not prove unfairness because default rates differ.
4. Lender-set inputs: interest rate and grade carry earlier lending judgement.

## Review cycle
Framework reviewed every six months, after every incident, and before any model change is promoted.
