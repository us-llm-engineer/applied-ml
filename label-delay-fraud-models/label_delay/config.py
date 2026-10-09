"""Canonical configuration and hashing helpers.

The canonical config is the small base configuration. Code
must read every experiment size from the config it is handed rather than
hard-coding 1200/400/500, because notebooks run smaller demo configs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

BASE_CONFIG: dict[str, Any] = {
    "seed": 20260921,
    "n_transactions": 1_200,
    "n_replicates": 400,
    "e_min_values": [0.10, 0.50],
    "watch_threshold": 20,
    "watch_streams": 500,
}

#: Reduced-size config for notebook demonstrations and smoke checks only.
DEMO_CONFIG: dict[str, Any] = {
    **BASE_CONFIG,
    "n_transactions": 600,
    "n_replicates": 80,
    "watch_streams": 200,
}


def canonical_config() -> dict[str, Any]:
    """Return a fresh copy of the canonical experiment config."""
    return copy.deepcopy(BASE_CONFIG)


#: Study config: one shared stream plus the study sizes the figure builders
#: and experiments read. Every size is fixed here, never tuned from measured results.
STUDY_CONFIG: dict[str, Any] = {
    "config_name": "study",
    "seed": 20260923,
    "stream_n": 60_000,
    "stream_t_drift": 30_000,
    "n_propensity_strata": 6,
    "n_replicates": 100,
    "n_seeds": 24,
    "label_budgets": [50, 100, 200, 400, 800],
    "fpr_target": 0.01,
    "n_trials": 200,
    "n_windows": 10,
    "alert_budget": 0.01,
    "wctm_threshold_c": 20,
    "n_runs": 20,
    "n_boot": 500,
}


def canonical_study_config() -> dict[str, Any]:
    """Return a fresh copy of the canonical experiment config."""
    return copy.deepcopy(STUDY_CONFIG)


def config_key(config: Mapping[str, Any]) -> str:
    """Canonical JSON key used for deterministic memoisation and hashing."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable sha256 of a config mapping (used in audits and provenance)."""
    return hashlib.sha256(config_key(config).encode("utf-8")).hexdigest()


def seed_of(config: Mapping[str, Any]) -> int:
    """Integer seed for the config's deterministic generators."""
    return int(config.get("seed", 0))


def run_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """The single run identity shared by figures, manifest and notebooks."""
    return {
        "config_name": str(config.get("config_name", "study")),
        "run_id": f"r2-{config_hash(config)[:12]}-seed{seed_of(config)}",
        "seed": seed_of(config),
        "config_hash": config_hash(config),
    }
