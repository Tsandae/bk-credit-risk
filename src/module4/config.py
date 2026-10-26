"""Central settings for Module 4. Everything the report quotes is controlled from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURATED_FILE = ROOT / "data" / "curated" / "loans_curated.parquet"
MODELS = ROOT / "models" / "module4"
REPORTS = ROOT / "reports" / "module4"
FIGS = REPORTS / "figures"
DOCS = ROOT / "docs" / "module4"
MODEL_FILE = MODELS / "credit_risk_model.joblib"

MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlruns").as_posix()
EXPERIMENT = "bk-credit-risk-module4"
REGISTERED_MODEL = "credit_risk_model"

TARGET = "default"
NUMERIC = [
    "loan_amnt", "term", "int_rate", "installment", "emp_length", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "emp_length_missing",
    "loan_to_income", "installment_to_income", "credit_history_years",
]
CATEGORICAL = ["grade", "home_ownership", "verification_status", "purpose"]
FEATURES = NUMERIC + CATEGORICAL
# Fields excluded on purpose (duplicates, identifiers, constants, split-only):
EXCLUDED = ["funded_amnt", "sub_grade", "grade_num", "log_annual_inc", "loan_key", "zip3",
            "application_type", "addr_state", "issue_year"]

# Time-based split (Module 2 plan): train 2007-2013, validate 2014, test 2015 (used once).
TRAIN_LAST_YEAR, VAL_YEAR, TEST_YEAR = 2013, 2014, 2015

SEED = 42
CV_SAMPLE = 30_000          # random-search sample size (computing limits)
CV_FOLDS = 3
N_ITER = 12
HIGH_SHARE, MEDIUM_SHARE = 0.20, 0.30     # tiers: top 20% high, next 30% medium, bottom 50% low
MONOTONE_UP = ["dti", "revol_util"]       # risk may not fall when these rise

# Fairness (Module 2): groups, limits
FAIR_GROUPS = ["income_quartile", "home_ownership", "employment_length_group", "region"]
MIN_GROUP_N = 200
MAX_GAP_PP = 5.0
MIN_APPROVAL_RATIO = 0.80

# Module 2 objectives
OBJ_AUC, OBJ_KS, OBJ_CAPTURE, OBJ_CALIB = 0.70, 0.25, 0.40, 0.10

# Model choice: LightGBM is the Module 2 main model. It is kept unless its 2014 validation AUC is more than
# SELECT_TOL below the best model (differences this small are within noise); then the best model is used.
MAIN_MODEL = "lightgbm"
SELECT_TOL = 0.005
