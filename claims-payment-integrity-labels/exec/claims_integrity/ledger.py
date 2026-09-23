"""Run ledger: seed, stable config hash, capacity and measured runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "claims-integrity-ledger-v1"


def config_hash(config: Mapping[str, Any]) -> str:
    """Canonical SHA-256 over sorted-key compact JSON."""

    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def record_run(
    seed: int,
    config: Mapping[str, Any],
    capacity: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    """One observability row for a study run."""

    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "config_hash": config_hash(config),
        "capacity": int(capacity),
        "runtime_seconds": float(runtime_seconds),
    }


def write_ledger(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write JSONL ledger rows, preserving order."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return target
