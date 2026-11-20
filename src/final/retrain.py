"""Automated retraining with a promotion gate (Python + MLflow).

    python -m src.final.retrain                       # retrain only if monitoring says so; challenger is NOT promoted
    python -m src.final.retrain --force --quick       # retrain now (fast settings) - for testing the flow
    python -m src.final.retrain --data new.parquet    # retrain on an updated curated file
    python -m src.final.retrain --approve             # promote a challenger that passed the gate (needs human sign-off)

Flow: monitor -> trigger? -> train challenger into models/challenger (MLflow alias 'challenger') ->
promotion gate against the champion -> decision file -> a named human approves -> promote (MLflow alias 'staging').
The live model is never replaced by a script without --approve.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.module4 import config  # noqa: E402

CHAMP_RESULTS = ROOT / "reports" / "module4" / "results.json"
CHALL_DIR = ROOT / "models" / "challenger"
CHALL_REPORTS = ROOT / "reports" / "challenger"
DECISION = ROOT / "reports" / "final" / "retrain_decision.json"

AUC_TOLERANCE = 0.005           # challenger may be at most this much worse than the champion on the test year
RATIO_TOLERANCE = 0.05          # ...and no group's approval ratio may fall by more than this


def should_retrain(report: dict) -> tuple[bool, str]:
    """Trigger rule from the Module 4 monitoring plan: a data, model or calibration breach two windows in a row."""
    wins = list(report["windows"].values())
    kinds = {"data/score drift", "model drift", "calibration drift"}
    hits = [any(a["type"] in kinds for a in w["alerts"]) for w in wins]
    if len(hits) >= 2 and hits[-1] and hits[-2]:
        return True, "data, model or calibration breach in two consecutive windows"
    if hits and hits[-1] and any("second breach" in w["action"] for w in wins):
        return True, "latest window breached and follows an earlier breach"
    return False, "no data, model or calibration breach in two consecutive windows"


def gate(champ: dict, chall: dict) -> dict:
    """Compare challenger to champion on the same checks the Module 4 report uses. Returns pass/fail per check."""
    cm, hm = champ["selected_model"], chall["selected_model"]
    c_auc, h_auc = champ["models"][cm]["test"]["roc_auc"], chall["models"][hm]["test"]["roc_auc"]
    h_ks = chall["models"][hm]["test"]["ks"]
    c_cal = max(abs(r["relative_error"]) for r in champ["tier_calibration"])
    h_cal = max(abs(r["relative_error"]) for r in chall["tier_calibration"])
    ratio_drops = {g: champ["fairness_before"][g]["approval_ratio"] - chall["fairness_before"][g]["approval_ratio"]
                   for g in champ["fairness_before"] if g in chall["fairness_before"]}
    checks = {
        "test AUC not worse than champion by more than 0.005": (h_auc >= c_auc - AUC_TOLERANCE, f"champion {c_auc:.3f}, challenger {h_auc:.3f}"),
        "KS at least 0.25 (Module 2 goal)": (h_ks >= config.OBJ_KS, f"{h_ks:.3f}"),
        "worst tier calibration error not worse than champion": (h_cal <= c_cal + 1e-9, f"champion {c_cal:.3f}, challenger {h_cal:.3f}"),
        "no group approval ratio falls by more than 0.05": (all(d <= RATIO_TOLERANCE for d in ratio_drops.values()), json.dumps({k: round(v, 3) for k, v in ratio_drops.items()})),
    }
    return {"checks": {k: {"passed": bool(v[0]), "detail": v[1]} for k, v in checks.items()}, "passed": all(v[0] for v in checks.values())}


def train_challenger(data_path=None, quick=False):
    from src.module4 import run_all
    CHALL_DIR.mkdir(parents=True, exist_ok=True)
    CHALL_REPORTS.mkdir(parents=True, exist_ok=True)
    keep = (config.MODELS, config.MODEL_FILE, config.REPORTS, config.FIGS, config.DOCS, config.MLFLOW_ALIAS)
    config.MODELS, config.MODEL_FILE = CHALL_DIR, CHALL_DIR / "credit_risk_model.joblib"
    config.REPORTS, config.FIGS, config.DOCS = CHALL_REPORTS, CHALL_REPORTS / "figures", CHALL_REPORTS / "docs"
    config.MLFLOW_ALIAS = "challenger"
    try:
        run_all.run(path=data_path, quick=quick)
    finally:
        (config.MODELS, config.MODEL_FILE, config.REPORTS, config.FIGS, config.DOCS, config.MLFLOW_ALIAS) = keep
    return json.loads((CHALL_REPORTS / "results.json").read_text())


def promote(approver: str):
    """Human-approved promotion: back up the champion, copy the challenger in, move the MLflow alias."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = ROOT / "models" / "archive" / stamp
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.MODEL_FILE, backup / "credit_risk_model.joblib")
    shutil.copy(CHAMP_RESULTS, backup / "results.json")
    shutil.copy(CHALL_DIR / "credit_risk_model.joblib", config.MODEL_FILE)
    shutil.copy(CHALL_REPORTS / "results.json", CHAMP_RESULTS)
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()
        v = client.get_model_version_by_alias(config.REGISTERED_MODEL, "challenger").version
        client.set_registered_model_alias(config.REGISTERED_MODEL, "staging", v)
    except Exception as e:  # MLflow registry is optional on small hosts
        print("MLflow alias not moved:", e)
    return str(backup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--quick", action="store_true", help="fast settings for testing the flow (not for reporting)")
    ap.add_argument("--force", action="store_true", help="retrain even if monitoring did not ask for it")
    ap.add_argument("--approve", metavar="NAME", help="name of the person approving promotion of a challenger that passed the gate")
    a = ap.parse_args()

    if a.approve:
        d = json.loads(DECISION.read_text())
        if not d.get("gate", {}).get("passed"):
            sys.exit("No passed challenger on record: nothing to promote.")
        backup = promote(a.approve)
        d.update({"promoted": True, "approved_by": a.approve, "approved_at": datetime.now().isoformat(timespec="seconds"), "champion_backup": backup})
        DECISION.write_text(json.dumps(d, indent=2))
        print(f"Promoted. Previous champion archived at {backup}. Re-run: python dashboard/build_dashboard_data.py && python -m src.final.monitor")
        return

    from src.final import monitor
    report = monitor.run(a.data, write=False) if a.data else monitor.run(write=False)
    trigger, why = should_retrain(report)
    decision = {"generated": datetime.now().isoformat(timespec="seconds"), "trigger": trigger, "reason": why, "forced": a.force, "promoted": False}
    if not (trigger or a.force):
        decision["outcome"] = "No retraining needed."
    else:
        champ = json.loads(CHAMP_RESULTS.read_text())
        chall = train_challenger(a.data, a.quick)
        decision["gate"] = gate(champ, chall)
        decision["challenger_test_auc"] = chall["models"][chall["selected_model"]]["test"]["roc_auc"]
        decision["quick_settings"] = a.quick
        decision["outcome"] = ("Challenger passed the gate: READY FOR HUMAN APPROVAL (run with --approve NAME)." if decision["gate"]["passed"]
                               else "Challenger failed the gate: champion stays in place.")
    DECISION.parent.mkdir(parents=True, exist_ok=True)
    DECISION.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
