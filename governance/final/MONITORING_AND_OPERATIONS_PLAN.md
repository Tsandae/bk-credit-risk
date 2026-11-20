# Implementation, monitoring, incident response and decommissioning plan

## 1. Deployment path
1. **Prototype (now):** API on Render or Hugging Face Spaces (Docker), dashboards on Streamlit Cloud, public US data only.
2. **Local-data re-test:** retrain and recalibrate on recent Bank of Kigali data (via `python -m src.final.retrain --data`); acceptance criteria below.
3. **Shadow pilot (3 months):** the model scores applications alongside human review; its output is not shown to approvers. Compare decisions and outcomes.
4. **Supervised pilot:** scores shown to credit officers as decision support, with override logging.
5. **Scale-up:** only after the committee reviews pilot evidence.

**Acceptance criteria (from the Module 4 report):** AUC at least 0.70, KS at least 0.25, at least 40% of defaulters in the top 20%, tier calibration within +/-10%, fairness limits met or a documented exception, expected loss 10% lower at the same workload, and approval by risk, compliance and data governance.

## 2. Performance KPIs
| KPI | Target | Alert |
|---|---|---|
| AUC (rolling 3 months, on matured loans) | at least 0.70 | falls more than 0.03 below the validation AUC |
| KS | at least 0.25 | below 0.25 |
| Defaulters in riskiest 20% | at least 40% | below 35% |
| Tier calibration (predicted vs actual) | within +/-10% | outside +/-10% |
| Share of scores overridden by officers | tracked | above 20% or rising two months in a row |
| API availability / latency | 99% / under 1 second | below target |

## 3. Continuous fairness monitoring
Monitored weekly (`monitor.yml` or the monitor container) for income band, housing status, time in job and region: approval ratio (floor 0.80), TPR gap and FPR gap (at most 5 points), calibration by group, and groups with fewer than 200 loans reported as "too small to test". The monitoring dashboard (`dashboard/monitoring_app.py`) shows each window against the limits with a status icon and label, not colour alone.

## 4. Drift detection
- **Data drift:** population stability index per input and for the score; alert above 0.2. An Evidently report (`reports/final/evidently_drift_2015.html`) gives the full per-column view.
- **Model drift:** AUC and KS by window versus validation.
- **Calibration drift:** predicted vs actual by tier.
- **Bias drift:** the fairness metrics above, per window.
- **Rule:** one breach means investigate; two in a row means review or retrain.

## 5. Retraining
`python -m src.final.retrain` retrains a challenger with MLflow tracking (alias "challenger"), runs the promotion gate (test AUC not more than 0.005 worse, KS at least 0.25, tier calibration not worse, no group approval ratio down by more than 0.05), and writes `retrain_decision.json`. Promotion needs a named human (`--approve NAME`); the previous model is archived.

## 6. Incident response plan (ethical failures)
| Step | Action | Owner | Time |
|---|---|---|---|
| Detect | Alert from monitoring, an override pattern, a complaint, or a media or regulator query | Anyone; logged | Immediately |
| Triage | Classify: **S1** (people harmed or unlawful, e.g. systematic exclusion of a group), **S2** (limit breached, no confirmed harm), **S3** (near miss or documentation) | Model risk lead | 1 working day |
| Contain | S1: switch the model to "review everything manually" mode (feature flag) and stop automated flagging; S2: continue with extra human review | Model owner | 1 day (S1) |
| Investigate | Pull the explanation-store records for the affected period, find the cause (data, drift, bug, policy) | Validation lead + fairness reviewer | 5 days |
| Remediate | Fix, retrain or revert to the archived model; re-run fairness checks; review affected decisions and offer re-review to affected applicants | Model owner | As agreed by committee |
| Communicate | Inform the committee, internal audit, the regulator where required, and affected applicants; update the public trust statement if material | DPO + model owner | Per legal duty |
| Learn | Post-incident review, update controls and this plan | Committee | 30 days |

## 7. Decommissioning plan
**Triggers:** committee decision; unresolved S1 incident; performance below acceptance criteria for two consecutive review periods; legal change; replacement model; end of business need.
1. Announce the retirement date and freeze new versions.
2. Run the replacement (or manual process) in parallel for one review period.
3. Switch the API to return "model retired" and stop scoring; disable the dashboards.
4. Archive: model file, training data reference and hash, model card, final monitoring report, committee minutes (7 years, access-controlled).
5. Delete: explanation-store records beyond the retention period, caches, temporary data, public demo endpoints, and cloud copies; record the deletions in the audit log.
6. Tell users and regulators; keep a short "lessons learned" note.
7. Confirm that no downstream system still calls the API.
