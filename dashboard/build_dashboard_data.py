"""Create the small, anonymous data files the dashboard needs from the Module 3 / Module 4 outputs.

    python dashboard/build_dashboard_data.py

Reads  data/curated/loans_curated.parquet, models/module4/credit_risk_model.joblib, reports/module4/*
Writes dashboard/data/test_scored.parquet (score + outcome + fairness groups only - no identifiers),
       dashboard/data/results.json, dashboard/data/meta.json, dashboard/data/fig2_shap_summary.png
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

from src.module4 import config, data  # noqa: E402
from src.module4.model import feature_frame  # noqa: E402

OUT = ROOT / "dashboard" / "data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.MODEL_FILE)
    _, _, test = data.prepare()
    test = test.reset_index(drop=True)
    scored = test[config.FAIR_GROUPS + [config.TARGET]].copy()
    scored["raw"] = model.raw(feature_frame(test))
    scored.to_parquet(OUT / "test_scored.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "cut_medium": model.cut_medium, "cut_high": model.cut_high, "n_test": int(len(scored)),
        "model": model.name}, indent=2))
    shutil.copy(config.REPORTS / "results.json", OUT / "results.json")
    fig = config.FIGS / "fig2_shap_summary.png"
    if fig.exists():
        shutil.copy(fig, OUT / "fig2_shap_summary.png")
    print(f"wrote {OUT} ({len(scored):,} scored 2015 test loans)")


if __name__ == "__main__":
    main()
