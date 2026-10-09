"""Frozen configuration for the synthetic claims-integrity study.

All numbers here are *derived here* design choices for a synthetic benchmark;
none of them is a claim about real reimbursement economics.  The config is a
plain nested dict so that :func:`claims_integrity.ledger.config_hash` can
hash it with canonical JSON.
"""

from __future__ import annotations

from typing import Any, Mapping

SCHEMA = "claims-integrity-config-v1"

#: Latent claim classes used by the CCEM model (Ibrahim, Nguyen, Fu 2023).
CLASS_NAMES = ("compliant", "billing_error", "medically_unnecessary")

#: Frozen scenario.  A single shared config keeps the stream, the
#: splits, the seeds and the reviewer-minute accounting comparable across the
#: SCRC, SCoRC and derived capacity branches.
CONFIG: dict[str, Any] = {
    "scenario": "synthetic-claims-v1",
    "k_latent": 3,
    "n_features": 10,
    "n_reviewers": 6,
    "class_prior": (0.94, 0.04, 0.02),
    "feature_shift": 0.75,
    "queue_noise": 0.35,
    "specialist_bias": 0.12,
    "ambiguous_upper": 0.60,
    "billed_median": 320.0,
    "billed_sigma": 1.6,
    "loss_bound_dollars": 5000.0,
    "value_per_autopay": 9.5,
    "cost_per_review": 12.0,
    "cost_per_minute": 1.5,
    "minutes_per_review": 3.0,
    "alpha": 0.05,
    "alpha_sensitivity": 0.10,
    "pi_min": 0.50,
    "delta": 0.10,
    "capacity_claims": (20, 40, 80),
    "lambda_grid": (0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50),
    "tau_grid": ("msp", "margin"),
    "seeds": (11, 23, 37, 53),
    "fit_seed": 20260921,
    "n_claims": 1500,
    "n_cert": 400,
}

#: Provenance tag written next to every artifact.
PROVENANCE = "claims-integrity-v1; config claims-integrity-config-v1"


def canonical_config(**overrides: Any) -> dict[str, Any]:
    """Return the frozen config with per-call overrides applied."""

    cfg = dict(CONFIG)
    cfg.update(overrides)
    return cfg


def as_mapping(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return dict(CONFIG if cfg is None else cfg)
