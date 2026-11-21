# BK Credit-Risk stakeholder dashboard (Module 5)

Five pages: Overview, Score an applicant & what-if, What drives predictions, Ethical compliance (fairness, live Fairlearn),
Limits & transparency. Built on the saved Module 4 model; data are US LendingClub loans used as a stand-in for BK data.

```bash
pip install -r dashboard/requirements.txt
python dashboard/build_dashboard_data.py    
streamlit run dashboard/app.py
python -m pytest tests/module5 -q
```

