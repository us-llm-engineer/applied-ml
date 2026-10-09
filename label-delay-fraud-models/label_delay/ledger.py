"""The run ledger and the on-disk estimator cache.

Every call to :func:`run_estimator` appends exactly one JSON line to
``config["ledger_path"]`` in run order, with the eight ledger keys
``(estimator, run_id, seed, config_hash, data_hash, wall_clock_s, label_cost,
cached)``. The run identity (``config_hash``/``run_id``) is computed on the
config with the infrastructure keys ``ledger_path`` and ``cache_dir`` removed,
so two runs that differ only in where they write report the same identity.

Repeating the identical ``(cache_dir, estimator, experiment config)`` is served
from an on-disk JSON cache: the row is written again with ``cached=True`` and
still reports the stored ``label_cost`` and the stored ``wall_clock_s`` of the
run that actually produced the result -- a cached re-run is fast, but the cost
it reports is the cost it took to compute, not the cost of the cache hit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from label_delay.config import run_identity, seed_of
from label_delay.recovery import STREAM_ESTIMATORS, effective_config, estimator_sidecar

LEDGER_KEYS: tuple[str, ...] = (
    "estimator",
    "run_id",
    "seed",
    "config_hash",
    "data_hash",
    "wall_clock_s",
    "label_cost",
    "cached",
)


def _cache_path(config: Mapping[str, Any], name: str, config_hash: str) -> Path:
    cache_dir = Path(str(config["cache_dir"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}-{config_hash[:24]}.json"


def _append_ledger_row(ledger_path: Path, row: Mapping[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def run_estimator(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one estimator on the shared stream, ledged and cached on disk."""
    if name not in STREAM_ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; expected one of {STREAM_ESTIMATORS}")
    effective = effective_config(config)
    identity = run_identity(effective)
    path = _cache_path(config, name, identity["config_hash"])
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        sidecar = stored["sidecar"]
        wall_clock_s = float(stored["wall_clock_s"])
        label_cost = int(stored["label_cost"])
        cached = True
    else:
        started = time.perf_counter()
        sidecar = estimator_sidecar(name, effective)
        wall_clock_s = max(float(time.perf_counter() - started), 1e-9)
        label_cost = int(sidecar["label_cost"])
        payload = {
            "estimator": name,
            "sidecar": sidecar,
            "wall_clock_s": wall_clock_s,
            "label_cost": label_cost,
            "config_hash": identity["config_hash"],
            "run_id": identity["run_id"],
        }
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        cached = False
    ledger_row = {
        "estimator": name,
        "run_id": identity["run_id"],
        "seed": seed_of(effective),
        "config_hash": identity["config_hash"],
        "data_hash": str(sidecar["data_hash"]),
        "wall_clock_s": wall_clock_s,
        "label_cost": label_cost,
        "cached": cached,
    }
    _append_ledger_row(Path(str(config["ledger_path"])), ledger_row)
    return {"ledger_row": ledger_row, "result": {"sidecar": sidecar}}
