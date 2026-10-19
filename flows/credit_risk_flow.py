"""Orchestration DAG (Prefect): ingest -> clean -> privacy gate -> transform -> integrate -> validate -> bias audit -> publish.

Run locally:   python -m flows.credit_risk_flow
Schedule:      prefect deploy  (e.g. nightly cron) — retries and logging come from the task decorators.
"""
import json
import os
from pathlib import Path

import pandas as pd
from prefect import flow, task, get_run_logger

from pipeline import config
from pipeline.anonymize import anonymize
from pipeline.audit import actor, get_lineage, log_event, reset_lineage, verify_chain
from pipeline.bias import run_bias_suite
from pipeline.clean import clean
from pipeline.governance import check_access, enforce_retention
from pipeline.ingest import ingest
from pipeline.transform import integrate, transform
from pipeline.validate import validate


@task(name="1-ingest", retries=2, retry_delay_seconds=5, persist_result=False)
def t_ingest(path: str) -> pd.DataFrame:
    return ingest(path)


@task(name="2-clean", persist_result=False)
def t_clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean(df)


@task(name="3-privacy-gate", persist_result=False)
def t_anonymize(df: pd.DataFrame):
    out, report = anonymize(df)
    if report["k_after"] < report["k_target"] or report["residual_pii_patterns"] > 0:
        raise ValueError(f"Privacy gate failed: {report}")
    check_access(actor()[1], "staging", "write")
    config.STAGING.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.STAGING / "loans_anonymized.parquet", index=False)
    return out, report


@task(name="4-transform", persist_result=False)
def t_transform(df: pd.DataFrame) -> pd.DataFrame:
    return transform(df)


@task(name="5-integrate", persist_result=False)
def t_integrate(df: pd.DataFrame) -> pd.DataFrame:
    return integrate(df)


@task(name="6-validate", persist_result=False)
def t_validate(df: pd.DataFrame) -> dict:
    return validate(df, fail_on_error=True)  # a failed expectation stops the DAG before anything is published


@task(name="7-bias-audit", persist_result=False)
def t_bias(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    return run_bias_suite(before, after)["summary"]


@task(name="8-publish", persist_result=False)
def t_publish(df: pd.DataFrame, extras: dict) -> str:
    check_access(actor()[1], "curated", "write")
    config.CURATED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CURATED_FILE, index=False)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "lineage.json").write_text(json.dumps(get_lineage(), indent=2))
    (config.REPORTS / "pipeline_summary.json").write_text(json.dumps(extras, indent=2, default=str))
    log_event("PUBLISH", "curated", file=config.CURATED_FILE.name, rows=int(len(df)), cols=int(df.shape[1]))
    return str(config.CURATED_FILE)


@flow(name="credit-risk-data-pipeline", log_prints=True)
def credit_risk_pipeline(raw_path: str = str(config.RAW_FILE)) -> str:
    reset_lineage()
    os.environ.setdefault("ANON_SALT", "")  # must be provided by the environment (see README)
    raw = t_ingest(raw_path)
    cleaned = t_clean(raw)
    anon, anon_report = t_anonymize(cleaned)
    features = t_transform(anon)
    curated = t_integrate(features)
    validation = t_validate(curated)
    bias = t_bias(anon.assign(region=anon["addr_state"].map(_region_map()).fillna("Other/Unknown")), curated)
    out = t_publish(curated, {"anonymization": anon_report,
                              "validation": {k: validation[k] for k in ("success", "evaluated", "passed", "failed")},
                              "bias": bias,
                              "audit_chain_intact": verify_chain()})
    for zone, folder in (("staging", config.STAGING), ("curated", config.CURATED)):
        enforce_retention(zone, folder)
    get_run_logger().info("Published %s (%d rows)", out, len(curated))
    return out


def _region_map() -> dict:
    ref = pd.read_csv(config.STATE_REGION_FILE)
    return dict(zip(ref["addr_state"], ref["region"]))


if __name__ == "__main__":
    credit_risk_pipeline()
