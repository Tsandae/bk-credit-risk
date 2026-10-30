# BK Credit-Risk stakeholder dashboard (Module 5)

Five pages: Overview, Score an applicant & what-if, What drives predictions, Ethical compliance (fairness, live Fairlearn),
Limits & transparency. Built on the saved Module 4 model; data are US LendingClub loans used as a stand-in for BK data.

```bash
pip install -r dashboard/requirements.txt
python dashboard/build_dashboard_data.py     # only if you re-trained: refreshes dashboard/data/*
streamlit run dashboard/app.py
python -m pytest tests/module5 -q
```

Deploy: share.streamlit.io -> New app -> repo `Tsandae/bk-credit-risk`, branch `main`, main file path `dashboard/app.py`.
`dashboard/data/` holds only a score, outcome and group labels for 30,835 test loans (no identifiers).
