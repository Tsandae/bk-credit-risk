"""Access control and retention enforcement driven by governance/policy.yaml."""
import time
from pathlib import Path

import yaml

from . import config
from .audit import log_event

POLICY_PATH = config.ROOT / "governance" / "policy.yaml"


def load_policy() -> dict:
    with open(POLICY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_access(role: str, zone: str, action: str) -> bool:
    """Raise PermissionError unless `role` may perform `action` on `zone`. Both outcomes are logged."""
    allowed = load_policy()["access_control"].get(zone, {}).get(role, [])
    ok = action in allowed
    log_event("ACCESS_GRANTED" if ok else "ACCESS_DENIED", zone, requested_role=role, requested_action=action)
    if not ok:
        raise PermissionError(f"Role '{role}' may not '{action}' on zone '{zone}'")
    return True


def enforce_retention(zone: str, directory, now: float = None) -> list:
    """Delete files in `directory` older than the zone's retention period; log each deletion (name only)."""
    days = load_policy()["retention_days"][zone]
    cutoff = (now or time.time()) - days * 86400
    removed = []
    for p in Path(directory).glob("*"):
        if p.is_file() and p.name != ".gitkeep" and p.stat().st_mtime < cutoff:
            p.unlink()
            removed.append(p.name)
            log_event("RETENTION_DELETE", zone, file=p.name, retention_days=days)
    return removed
