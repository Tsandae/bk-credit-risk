import json
import os
import time

import pytest

from pipeline import audit
from pipeline.governance import check_access, enforce_retention


def test_log_event_writes_hash_chain():
    audit.log_event("A", "raw", rows=1)
    audit.log_event("B", "raw", rows=2)
    assert audit.verify_chain()


def test_tampering_is_detected():
    audit.log_event("A", "raw", rows=1)
    audit.log_event("B", "raw", rows=2)
    p = audit.log_path()
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["details"]["rows"] = 999
    lines[0] = json.dumps(rec, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")
    assert not audit.verify_chain()


def test_audited_decorator_logs_counts_not_values():
    import pandas as pd

    @audit.audited("DEMO", "staging")
    def f(df):
        return df.assign(c=1).drop(columns=["secret"])

    f(pd.DataFrame({"secret": ["alice@example.com"], "a": [1]}))
    text = audit.log_path().read_text()
    assert "alice@example.com" not in text
    entry = audit.get_lineage()[-1]
    assert entry["cols_added"] == ["c"] and entry["cols_dropped"] == ["secret"]


def test_access_granted_and_denied_are_logged():
    assert check_access("data_engineer", "raw", "read")
    with pytest.raises(PermissionError):
        check_access("analyst", "raw", "read")
    actions = [json.loads(l)["action"] for l in audit.log_path().read_text().splitlines()]
    assert actions == ["ACCESS_GRANTED", "ACCESS_DENIED"]


def test_retention_deletes_only_expired_files(tmp_path):
    old, new = tmp_path / "old.parquet", tmp_path / "new.parquet"
    old.write_text("x"); new.write_text("x")
    long_ago = time.time() - 40 * 86400
    os.utime(old, (long_ago, long_ago))
    removed = enforce_retention("staging", tmp_path)  # staging retention = 30 days
    assert removed == ["old.parquet"] and new.exists() and not old.exists()
