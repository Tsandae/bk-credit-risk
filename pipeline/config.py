"""Central configuration for the Bank of Kigali credit-risk data pipeline (LendingClub loan.csv)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
STAGING = DATA / "staging"
CURATED = DATA / "curated"
REF = DATA / "reference"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

RAW_FILE = RAW / "loan.csv"
CURATED_FILE = CURATED / "loans_curated.parquet"
STATE_REGION_FILE = REF / "state_region.csv"

# Columns kept at ingestion (everything else in the 70+ column LendingClub file is ignored).
RAW_COLUMNS = [
    "id", "member_id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_title", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "title", "desc",
    "zip_code", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "application_type",
]

# Direct identifiers / free text removed outright (data minimisation).
DIRECT_PII = ["member_id", "emp_title", "title", "desc", "zip_code", "id"]

# Quasi-identifiers used for the k-anonymity check.
QUASI_IDENTIFIERS = ["addr_state", "zip3", "home_ownership", "income_band"]
K_ANONYMITY = 5

# Loan outcome mapping -> binary target. Everything else is unresolved and excluded.
BAD_STATUS = {"Charged Off", "Default", "Late (31-120 days)"}
GOOD_STATUS = {"Fully Paid"}

NUMERIC_COLS = [
    "loan_amnt", "funded_amnt", "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "total_acc",
]

# Attributes audited for representation bias (LendingClub has no race/gender fields,
# so these are the available proxy/segment attributes).
BIAS_ATTRIBUTES = ["home_ownership", "region", "income_band", "verification_status"]
MIN_GROUP_SHARE = 0.05      # groups below 5% of rows are flagged as under-represented
MAX_SHARE_SHIFT = 0.05      # >5 pp change in share between cleaned and curated data is flagged
MIN_RATE_RATIO = 0.80       # four-fifths rule on outcome-rate ratio
