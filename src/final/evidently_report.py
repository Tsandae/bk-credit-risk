"""Evidently AI data-drift and prediction-drift report (HTML) for the monitoring dashboard / auditors.

    python -m src.final.evidently_report            # 2007-2013 training window vs 2015 test window
Writes reports/final/evidently_drift_2015.html
"""
import sys
import warnings
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.module4 import config, data  # noqa: E402
from src.module4.model import feature_frame  # noqa: E402

COLS = ["int_rate", "annual_inc", "dti", "loan_amnt", "revol_util", "loan_to_income", "term", "grade", "home_ownership", "purpose"]


def main(n=8000):
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    model = joblib.load(config.MODEL_FILE)
    tr, _, te = data.split(data.load())
    out = []
    for d in (tr.sample(n, random_state=0), te.sample(min(n, len(te)), random_state=0)):
        f = feature_frame(d.reset_index(drop=True))[COLS].copy()
        f["model_score"] = model.raw(feature_frame(d.reset_index(drop=True)))
        for c in config.CATEGORICAL:
            if c in f:
                f[c] = f[c].astype(str)
        out.append(f)
    ref, cur = out
    definition = DataDefinition(numerical_columns=[c for c in COLS if c not in config.CATEGORICAL] + ["model_score"],
                                categorical_columns=[c for c in COLS if c in config.CATEGORICAL])
    snap = Report([DataDriftPreset()]).run(Dataset.from_pandas(cur, data_definition=definition),
                                           Dataset.from_pandas(ref, data_definition=definition))
    dest = ROOT / "reports" / "final" / "evidently_drift_2015.html"
    snap.save_html(str(dest))
    print("wrote", dest, f"({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
