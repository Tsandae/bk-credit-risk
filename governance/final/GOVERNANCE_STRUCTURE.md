# Ethical governance structure (ongoing oversight)

| Role | Responsibility | Decides |
|---|---|---|
| **Model owner** (head of credit risk) | Accountable for model use and outcomes | Whether the model is used, paused or retired |
| **Model risk and validation lead** | Independent review of performance, calibration and drift | Whether a challenger passes the promotion gate |
| **Fairness and ethics reviewer** (compliance) | Reviews fairness metrics, mitigation, complaints | Whether fairness limits are met or an exception is justified and documented |
| **Data protection officer** | Privacy, retention, data subject requests, legal basis | Whether data use and sharing are lawful |
| **Data engineering lead** | Pipeline, data quality, access control | Release of new data to modelling |
| **Credit officers (users)** | Use scores in review; record overrides | Final credit decision for each applicant |
| **Internal audit** | Annual audit of logs, overrides and controls | Reports to the board risk committee |

**Ethics and model risk committee.** Meets quarterly, and within 5 working days of any incident. Members: model owner, validation lead, fairness reviewer, data protection officer, one credit-officer representative, and an independent member. Quorum: four, including the fairness reviewer and data protection officer.

**Approval gates (nothing moves without the named sign-off).**
1. First use: committee approval after local-data re-test (acceptance criteria in the report).
2. Any retrained model: passes the automated gate (`src/final/retrain.py`), then validation lead and fairness reviewer sign off (`--approve NAME` is recorded in `retrain_decision.json`).
3. Any change to cut-offs or group-specific thresholds: committee decision with legal advice.
4. Exceptions to fairness limits: written, time-limited, reviewed at the next meeting.

**Records.** Explanation store (7 years), monitoring reports (7 years), override log, committee minutes, incident log.
