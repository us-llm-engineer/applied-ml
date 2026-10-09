"""Shared helpers for the three notebooks.

The notebooks use these to print their
canonical run identity, to run heavy experiments through an on-disk cache (so
NB3 can execute NB2 and reuse exactly the same results), and to print
``SELF-CHECK`` lines whose verdict is computed rather than asserted by hand.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "label_delay" / "results" / "notebook_cache"
MANIFEST = ROOT / "label_delay" / "results" / "run_manifest.json"

#: Reduced sizes so each notebook executes headlessly well under ten minutes.
#: The base keys (``n_transactions``, ``e_min_values``, ``watch_*``) are the
#: sizes the base callables read; the remaining keys are the sizes the shared
#: stream, the recovery/evaluation/monitoring/policy studies and the ledger read.
DEMO_OVERRIDES: dict[str, Any] = {
    "stream_n": 12_000,
    "stream_t_drift": 6_000,
    "n_seeds": 6,
    "n_propensity_strata": 6,
    "n_replicates": 8,
    "n_trials": 5,
    "n_windows": 8,
    "n_runs": 6,
    "label_budgets": [50, 100, 200],
    "n_boot": 200,
    "n_transactions": 800,
    "e_min_values": [0.10, 0.50],
    "watch_threshold": 20,
    "watch_streams": 120,
    #: Ledger infrastructure: label_delay.ledger excludes these two keys from the run
    #: identity, so the printed identity still equals the canonical manifest's.
    "cache_dir": str(ROOT / "label_delay" / "results" / "notebook_ledger_cache"),
    "ledger_path": str(ROOT / "label_delay" / "results" / "notebook_ledger.jsonl"),
}


def bootstrap() -> pathlib.Path:
    """Put the project root on ``sys.path``; return it."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return ROOT


def demo_config() -> dict[str, Any]:
    """The canonical config with the notebook-demo size overrides."""
    from label_delay.config import canonical_study_config

    config = canonical_study_config()
    config.update(DEMO_OVERRIDES)
    return config


def manifest_identity() -> dict[str, Any]:
    """The canonical identity printed by every notebook (== run_manifest.json)."""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "config_name": payload["config_name"],
        "run_id": payload["run_id"],
        "seed": payload.get("seed"),
        "config_hash": payload.get("config_hash"),
    }


def print_identity() -> dict[str, Any]:
    """Print the canonical config name and run id once per notebook."""
    identity = manifest_identity()
    print(f"config_name: {identity['config_name']}")
    print(f"run_id: {identity['run_id']}")
    print(f"seed: {identity['seed']} | config_hash: {identity['config_hash']}")
    return identity


def _cache_file(key: str) -> pathlib.Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{digest}.json"


def cached(key: str, compute: Callable[[], Any]) -> Any:
    """Return ``compute()``, persisted as JSON under the shared notebook cache."""
    path = _cache_file(key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = compute()
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return value


def self_check(
    number: int,
    name: str,
    value: float | None = None,
    threshold: float | None = None,
    rule: str = "le",
    note: str = "",
) -> bool:
    """Print one ``SELF-CHECK`` line and return the verdict.

    ``rule`` is the comparator for a target that should hold: ``le``/``lt``/
    ``ge``/``gt``. With no value/threshold the line records the measurement
    itself; with both it prints the exact comparator form
    and never prints PASS when the comparison fails.
    """
    comparisons = {
        "le": ("<=", lambda a, b: a <= b),
        "lt": ("<", lambda a, b: a < b),
        "ge": (">=", lambda a, b: a >= b),
        "gt": (">", lambda a, b: a > b),
    }
    if value is None or threshold is None:
        print(f"SELF-CHECK {number}: {name} ... PASS (recorded) {note}".rstrip())
        return True
    symbol, compare = comparisons[rule]
    ok = compare(float(value), float(threshold))
    verdict = "PASS" if ok else "FAIL"
    print(
        f"SELF-CHECK {number}: {name} ... {verdict} "
        f"value={float(value):.6g} {symbol} threshold={float(threshold):.6g} {note}".rstrip()
    )
    return ok


def finite(value: Any) -> bool:
    """True when ``value`` is a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def deep_copy(value: Any) -> Any:
    """Explicit copy for cached structures the caller may mutate."""
    return copy.deepcopy(value)
