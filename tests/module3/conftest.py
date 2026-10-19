import os

import pytest

from pipeline import audit, make_synthetic


@pytest.fixture(autouse=True)
def _isolated_audit_log(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_PATH", str(tmp_path / "audit.log"))
    monkeypatch.setenv("ANON_SALT", "test-salt")
    monkeypatch.setenv("PIPELINE_ROLE", "data_engineer")
    audit.reset_lineage()
    yield


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic.generate(n=12000, seed=7)
