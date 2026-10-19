"""Stage 1 — Ingestion: governed read of the raw LendingClub file."""
import hashlib
from pathlib import Path

import pandas as pd

from . import config
from .audit import actor, audited, log_event
from .governance import check_access


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@audited("INGEST", "raw")
def ingest(path=None) -> pd.DataFrame:
    path = Path(path or config.RAW_FILE)
    check_access(actor()[1], "raw", "read")
    log_event("RAW_READ", "raw", file=path.name, sha256=_sha256(path))  # checksum pins the exact input version
    return pd.read_csv(path, usecols=lambda c: c in config.RAW_COLUMNS, low_memory=False)
