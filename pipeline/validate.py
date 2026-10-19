"""Stage 5 — Data validation gate with Great Expectations (1.x API)."""
import json

import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd

from . import config
from .audit import log_event

REQUIRED_COLUMNS = [
    "loan_key", "loan_amnt", "term", "int_rate", "installment", "grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "purpose", "addr_state", "dti",
    "revol_util", "default", "loan_to_income", "credit_history_years", "region",
]


def build_expectations() -> list:
    e = []
    e.append(gxe.ExpectTableRowCountToBeBetween(min_value=5000, max_value=None))  # project minimum: 5,000 records
    e.append(gxe.ExpectTableColumnsToMatchSet(column_set=REQUIRED_COLUMNS, exact_match=False))
    e.append(gxe.ExpectColumnValuesToBeUnique(column="loan_key"))
    for col in ["loan_key", "loan_amnt", "annual_inc", "int_rate", "dti", "grade", "default", "region"]:
        e.append(gxe.ExpectColumnValuesToNotBeNull(column=col))
    e += [
        gxe.ExpectColumnValuesToBeBetween(column="loan_amnt", min_value=500, max_value=40_000),
        gxe.ExpectColumnValuesToBeBetween(column="int_rate", min_value=5, max_value=31),
        gxe.ExpectColumnValuesToBeBetween(column="dti", min_value=0, max_value=100),
        gxe.ExpectColumnValuesToBeBetween(column="annual_inc", min_value=1, max_value=10_000_000),
        gxe.ExpectColumnValuesToBeBetween(column="revol_util", min_value=0, max_value=150),
        gxe.ExpectColumnValuesToBeBetween(column="emp_length", min_value=0, max_value=10),
        gxe.ExpectColumnValuesToBeBetween(column="credit_history_years", min_value=0, max_value=80),
        gxe.ExpectColumnValuesToBeInSet(column="term", value_set=[36, 60]),
        gxe.ExpectColumnValuesToBeInSet(column="default", value_set=[0, 1]),
        gxe.ExpectColumnValuesToBeInSet(column="grade", value_set=list("ABCDEFG")),
        gxe.ExpectColumnValuesToBeInSet(column="home_ownership", value_set=["MORTGAGE", "RENT", "OWN", "OTHER"]),
        # Drift guard: a default rate far outside this band usually signals a broken label mapping.
        gxe.ExpectColumnMeanToBeBetween(column="default", min_value=0.03, max_value=0.45),
    ]
    return e


def validate(df: pd.DataFrame, out_path=None, fail_on_error: bool = True) -> dict:
    context = gx.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("curated_source")
    asset = source.add_dataframe_asset("curated_loans")
    batch_def = asset.add_batch_definition_whole_dataframe("whole_df")
    suite = context.suites.add(gx.ExpectationSuite(name="credit_risk_curated_suite"))
    for exp in build_expectations():
        suite.add_expectation(exp)
    vdef = context.validation_definitions.add(
        gx.ValidationDefinition(name="curated_validation", data=batch_def, suite=suite)
    )
    result = vdef.run(batch_parameters={"dataframe": df})

    rows = []
    for r in result.results:
        cfg = r.expectation_config
        kwargs = {k: v for k, v in cfg.kwargs.items() if k not in ("batch_id",)}
        rows.append({
            "expectation": cfg.type,
            "column": kwargs.get("column", "(table)"),
            "params": {k: v for k, v in kwargs.items() if k != "column"},
            "success": bool(r.success),
            "unexpected_percent": r.result.get("unexpected_percent"),
            "observed_value": r.result.get("observed_value"),
        })
    summary = {
        "suite": "credit_risk_curated_suite",
        "success": bool(result.success),
        "evaluated": len(rows),
        "passed": sum(x["success"] for x in rows),
        "failed": sum(not x["success"] for x in rows),
        "results": rows,
    }
    if out_path is not False:
        path = out_path or (config.REPORTS / "validation_results.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, default=str))
    log_event("VALIDATE", "curated", suite=summary["suite"], passed=summary["passed"], failed=summary["failed"])
    if fail_on_error and not summary["success"]:
        failed = [f'{x["expectation"]}({x["column"]})' for x in rows if not x["success"]]
        raise ValueError(f"Data validation failed: {failed}")
    return summary
