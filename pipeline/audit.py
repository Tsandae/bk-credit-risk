"""Privacy audit logging: tamper-evident, PII-free log of every data access and transformation.

Each record is one JSON line holding who/what/when plus row/column *counts and names* — never values.
Records are hash-chained (each carries the previous record's SHA-256) so edits or deletions are detectable.
"""
import functools
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

_LINEAGE: list = []
_LOGGER_NAME = "privacy_audit"


def log_path() -> Path:
    return Path(os.environ.get("AUDIT_LOG_PATH", config.LOGS / "privacy_audit.log"))


def actor():
    return os.environ.get("PIPELINE_USER", "svc_pipeline"), os.environ.get("PIPELINE_ROLE", "data_engineer")


def _get_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    current = [h for h in logger.handlers if getattr(h, "baseFilename", None) == str(path.resolve())]
    if not current:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def _last_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "GENESIS"
    with open(path, "rb") as f:
        lines = f.read().splitlines()
    return json.loads(lines[-1])["hash"]


def log_event(action: str, dataset: str, **details) -> dict:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    user, role = actor()
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": user,
        "role": role,
        "action": action,
        "dataset": dataset,
        "details": details,
        "prev_hash": _last_hash(path),
    }
    rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
    _get_logger(path).info(json.dumps(rec, sort_keys=True, default=str))
    return rec


def verify_chain(path=None) -> bool:
    """Return True if the audit log is intact (no edited, removed or reordered records)."""
    path = Path(path or log_path())
    prev = "GENESIS"
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        h = rec.pop("hash")
        if rec["prev_hash"] != prev:
            return False
        if hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest() != h:
            return False
        prev = h
    return True


def audited(action: str, dataset: str):
    """Decorator: log a transformation and append a lineage entry (row/column counts and names only)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            df_in = next((a for a in list(args) + list(kwargs.values()) if isinstance(a, pd.DataFrame)), None)
            out = fn(*args, **kwargs)
            items = out if isinstance(out, tuple) else (out,)
            df_out = next((o for o in items if isinstance(o, pd.DataFrame)), None)
            cols_in = list(df_in.columns) if df_in is not None else []
            cols_out = list(df_out.columns) if df_out is not None else []
            entry = {
                "step": action,
                "dataset": dataset,
                "rows_in": None if df_in is None else int(len(df_in)),
                "rows_out": None if df_out is None else int(len(df_out)),
                "cols_in": len(cols_in),
                "cols_out": len(cols_out),
                "cols_added": [c for c in cols_out if c not in cols_in],
                "cols_dropped": [c for c in cols_in if c not in cols_out],
            }
            _LINEAGE.append(entry)
            log_event(action, dataset, **{k: v for k, v in entry.items() if k not in ("step", "dataset")})
            return out

        return wrapper

    return deco


def get_lineage() -> list:
    return list(_LINEAGE)


def reset_lineage() -> None:
    _LINEAGE.clear()
