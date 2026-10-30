"""API demo: score three applicants through the Module 4 FastAPI service and print plain-language results.

1) Start the API (from the repo root):      uvicorn src.module4.api:app --port 8000
2) Run this script in a second terminal:    python scripts/api_demo.py
   Or let the script start the API itself:  python scripts/api_demo.py --serve
   Another host (e.g. a deployed API):      python scripts/api_demo.py --base-url https://your-api.example.com
Postman users: import scripts/postman_collection.json instead.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]

APPLICANTS = {
    "Applicant A (typical)": dict(loan_amnt=12000, term=36, int_rate=13.5, installment=407, grade="C", home_ownership="RENT",
                                  annual_inc=55000, verification_status="Verified", purpose="debt_consolidation", dti=18,
                                  revol_bal=9000, revol_util=45, credit_history_years=12, emp_length=5),
    "Applicant B (higher-risk profile)": dict(loan_amnt=21700, term=60, int_rate=26.0, installment=565, grade="F", home_ownership="RENT",
                                              annual_inc=25000, verification_status="Verified", purpose="debt_consolidation", dti=29.7,
                                              revol_bal=15000, revol_util=80, credit_history_years=6, inq_last_6mths=3, delinq_2yrs=1),
    "Applicant C (lower-risk profile)": dict(loan_amnt=8000, term=36, int_rate=5.3, installment=241, grade="A", home_ownership="MORTGAGE",
                                             annual_inc=250000, verification_status="Not Verified", purpose="credit_card", dti=8,
                                             revol_bal=3000, revol_util=4.4, credit_history_years=22, emp_length=10),
}
PLAIN = {"int_rate": "interest rate", "term": "loan term", "annual_inc": "annual income", "dti": "debt compared with income",
         "revol_util": "share of credit limits in use", "inq_last_6mths": "recent credit enquiries", "loan_to_income": "loan size compared with income"}


def wait_for(base, seconds=60):
    for _ in range(seconds):
        try:
            if requests.get(f"{base}/health", timeout=2).json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--serve", action="store_true", help="start the API locally for the demo")
    a = ap.parse_args()
    proc = None
    if a.serve:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.module4.api:app", "--port", "8000"], cwd=ROOT,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not wait_for(a.base_url):
            sys.exit(f"API not reachable at {a.base_url}. Start it with: uvicorn src.module4.api:app --port 8000")
        print(f"API healthy at {a.base_url}\n")
        for name, body in APPLICANTS.items():
            r = requests.post(f"{a.base_url}/predict", json=body, timeout=30)
            r.raise_for_status()
            j = r.json()
            print(f"{name}\n  estimated default chance: {j['default_probability']:.1%}   risk group: {j['risk_tier'].upper()}")
            for x in j["top_reasons"]:
                v = round(x["value"], 2) if isinstance(x["value"], float) else x["value"]
                print(f"  - {PLAIN.get(x['feature'], x['feature'])} = {v}: {x['effect']}")
            print()
        bad = dict(APPLICANTS["Applicant A (typical)"], annual_inc=-5)
        r = requests.post(f"{a.base_url}/predict", json=bad, timeout=30)
        print(f"Invalid input check (negative income) -> HTTP {r.status_code} (rejected, as designed)")
        print("\nDecision support only: a human reviewer makes the credit decision.")
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
