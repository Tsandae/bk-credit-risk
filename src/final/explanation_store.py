"""Explanation store: every prediction is saved with its inputs, score, tier and full SHAP explanation.

Why: an auditor, regulator or applicant can ask "why did the model say this on that day?" and get the same answer
back, from the same model version. Rows are hash-chained (like the Module 3 privacy audit log), so any later edit
or deletion is detectable. No names or identifiers are stored; the request id is a random UUID.

    store = ExplanationStore("logs/explanations.db")
    store.record(model_version, inputs, probability, tier, shap_values)
    store.verify_chain()           # -> (True, n_rows)
    store.export_csv("audit.csv")  # for auditors
    store.purge(retention_days=2555)
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """CREATE TABLE IF NOT EXISTS explanations (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT UNIQUE, created_utc TEXT, model_version TEXT,
    inputs_json TEXT, probability REAL, tier TEXT, shap_json TEXT, override_json TEXT,
    prev_hash TEXT, row_hash TEXT)"""


def _hash(prev: str, payload: dict) -> str:
    return hashlib.sha256((prev + json.dumps(payload, sort_keys=True, default=str)).encode()).hexdigest()


class ExplanationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path)

    def record(self, model_version, inputs: dict, probability: float, tier: str, shap: dict) -> str:
        rid, now = str(uuid.uuid4()), datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            last = c.execute("SELECT row_hash FROM explanations ORDER BY seq DESC LIMIT 1").fetchone()
            prev = last[0] if last else "GENESIS"
            payload = {"request_id": rid, "created_utc": now, "model_version": model_version, "inputs": inputs,
                       "probability": round(float(probability), 6), "tier": tier,
                       "shap": {k: round(float(v), 6) for k, v in shap.items()}}
            c.execute("INSERT INTO explanations (request_id, created_utc, model_version, inputs_json, probability, tier, "
                      "shap_json, override_json, prev_hash, row_hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (rid, now, model_version, json.dumps(inputs, default=str), payload["probability"], tier,
                       json.dumps(payload["shap"]), None, prev, _hash(prev, payload)))
        return rid

    def get(self, request_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT request_id, created_utc, model_version, inputs_json, probability, tier, shap_json, override_json "
                          "FROM explanations WHERE request_id=?", (request_id,)).fetchone()
        if not r:
            return None
        return {"request_id": r[0], "created_utc": r[1], "model_version": r[2], "inputs": json.loads(r[3]),
                "probability": r[4], "tier": r[5], "shap": json.loads(r[6]), "override": json.loads(r[7]) if r[7] else None}

    def recent(self, n=20) -> list[dict]:
        with self._conn() as c:
            ids = [r[0] for r in c.execute("SELECT request_id FROM explanations ORDER BY seq DESC LIMIT ?", (n,))]
        return [self.get(i) for i in ids]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM explanations").fetchone()[0]

    def verify_chain(self):
        """Recompute every hash. Returns (ok, rows_checked). Override notes are not part of the hash by design."""
        prev, n = "GENESIS", 0
        with self._conn() as c:
            for rid, ts, mv, inp, p, tier, shap, ph, rh in c.execute(
                    "SELECT request_id, created_utc, model_version, inputs_json, probability, tier, shap_json, prev_hash, row_hash "
                    "FROM explanations ORDER BY seq"):
                payload = {"request_id": rid, "created_utc": ts, "model_version": mv, "inputs": json.loads(inp),
                           "probability": p, "tier": tier, "shap": json.loads(shap)}
                if ph != prev or _hash(prev, payload) != rh:
                    return False, n
                prev, n = rh, n + 1
        return True, n

    def add_override(self, request_id: str, reviewer_role: str, decision: str, reason: str):
        """Record a human override (who by role, what, why) next to the model's explanation."""
        note = {"reviewer_role": reviewer_role, "decision": decision, "reason": reason,
                "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with self._conn() as c:
            c.execute("UPDATE explanations SET override_json=? WHERE request_id=?", (json.dumps(note), request_id))

    def export_csv(self, out):
        with self._conn() as c, open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["request_id", "created_utc", "model_version", "probability", "tier", "inputs", "shap", "override"])
            for r in c.execute("SELECT request_id, created_utc, model_version, probability, tier, inputs_json, shap_json, override_json "
                               "FROM explanations ORDER BY seq"):
                w.writerow(r)

    def purge(self, retention_days: int = 2555) -> int:
        """Delete rows older than the retention period (default 7 years, as in governance/policy.yaml)."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat(timespec="seconds")
        with self._conn() as c:
            n = c.execute("DELETE FROM explanations WHERE created_utc < ?", (cutoff,)).rowcount
        return n
