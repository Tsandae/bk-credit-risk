# Algorithmic Impact Assessment (AIA)

**System:** credit-risk decision-support model (LightGBM, calibrated, three tiers). **Decision supported:** whether a loan application is sent for closer review. **Stage:** pre-deployment, trained on US proxy data.

## 1. Who is affected
| Group | How they are affected |
|---|---|
| Loan applicants | A high-risk flag leads to review, a possible decline, or different terms |
| Credit officers | Use scores and reasons in their review |
| The bank | Credit losses, review workload, regulatory and reputational exposure |
| Regulator, auditors | Need evidence of fair, explainable, controlled use |

## 2. Impact and likelihood (before controls, Low / Medium / High)
| Risk | Who is harmed | Likelihood | Severity | Evidence |
|---|---|---|---|---|
| Lower-income applicants flagged more often | Lowest income band | High | High | Approval 67% vs 90%; ratio 0.75 (floor 0.80); TPR gap 29.4 points |
| Applicants with unreported employment length flagged more often | That group (1,565 test loans) | High | Medium | Approval 62.6% |
| Over-reliance on a moderate model | All applicants | Medium | High | AUC 0.687; 64% of defaulters missed at the project setting |
| Wrong risk level in the High tier | Applicants in that tier | Medium | Medium | High-tier estimate +12.0% |
| Performance falls over time | All | Medium | Medium | AUC 0.702 (2014) to 0.687 (2015) |
| Model trained on US data used on Rwandan borrowers | All | High if used unchanged | High | Proxy data |
| Rejected applicants never seen by the model | Excluded groups | High | Medium | Only funded loans in the data |
| Explanations misread as causes | Applicants, officers | Medium | Medium | SHAP and LIME show associations |

## 3. Controls and residual risk
| Risk | Control | Residual |
|---|---|---|
| Income-band disparity | Group-specific cut-offs tested (ratio 0.96, gaps 4.3 and 2.2 points); fairness monitored; legal review before adoption | Medium: cost is doubled workload and legal uncertainty |
| Over-reliance | Decision support only; mandatory human review; override log | Low to Medium |
| Wrong High-tier level | Recalibrate on local data; calibration alert at +/-10% | Medium until recalibrated |
| Drift | PSI, AUC, calibration, fairness monitoring; two breaches trigger review or retraining | Low to Medium |
| Proxy data | Do not use before re-testing on Bank of Kigali data; acceptance criteria defined | High until re-tested (blocking) |
| Missing rejected applicants | Reject-inference study on BK data; documented limitation | Medium |
| Misreading explanations | Plain-language wording; "associations, not causes" caveat; SHAP and LIME agreement shown | Low |

## 4. Conclusion
Not suitable for production use in its current form. Suitable as a documented prototype and as the basis for a pilot after (1) re-testing on Bank of Kigali data, (2) a governance decision on group-specific cut-offs, and (3) legal and data-protection sign-off.
