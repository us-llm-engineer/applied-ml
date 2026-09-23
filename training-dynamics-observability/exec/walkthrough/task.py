"""Synthetic teacher--student task with a known latent truth (NB2 artifact 1).

The task is a *project choice* derived here, not something the papers specify:
a linear teacher with a small nonlinear term, heteroscedastic observation noise,
and regimes named by the latent margin ``|x . w_star|`` relative to the noise
scale.  Every row carries the latent truth, so an incident can be audited
against a known answer *after* the fact while the online diagnosis never sees it.

All values written into ``dataset_rows`` are strict-JSON safe (no NaN, no
Infinity), and ``dataset_data_hash`` recomputes exactly what
``tests/test_30_walkthrough_data.py`` recomputes from the published rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exec.common import sha256_json  # noqa: E402  (frozen shared helper)

REGIME_HARD = "low_margin_high_noise"
REGIME_AMBIGUOUS = "boundary_margin"
REGIME_EASY = "comfortable_margin"

GENERATION_CONFIG = {
    "version": "nb2-walkthrough-v1",
    "seed": 20260921,
    "n_rows": 320,
    "n_features": 6,
    "teacher_scale": 1.0,
    "nonlinearity_scale": 0.8,
    "noise_sigma": 0.75,
    "hard_margin_sigma": 0.30,
    "ambiguous_margin_sigma": 0.75,
    "hard_noise_multiplier": 1.5,
    "easy_noise_multiplier": 0.7,
    "fit_fraction": 0.70,
    "normalization": "zscore_train_only",
    "round_decimals": 6,
}


def generation_config() -> dict:
    """The single config whose hash NB2 publishes and NB3 reuses."""
    return dict(GENERATION_CONFIG)


def generation_config_hash() -> str:
    return sha256_json(GENERATION_CONFIG)


def _teacher_weights(n_features: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    draw = rng.normal(size=int(n_features))
    draw = draw / np.linalg.norm(draw)
    return draw * float(scale)


def _round(value, decimals):
    return float(round(float(value), int(decimals)))


def generate_rows(config: dict | None = None) -> list:
    """Generate the observable rows plus their latent truth columns.

    The random draw order is fixed and documented here so the task is
    re-derivable from ``(seed, config)`` alone:

    1. ``X ~ N(0, I)``                  observable features
    2. ``w_star ~ N(0, I) / ||.||``     teacher weights (the known optimum)
    3. ``s = X w_star``                 latent linear score
    4. per-row regime from ``|s|`` vs the noise scale
    5. heteroscedastic Gaussian noise, then ``y = latent_signal + noise``
    """
    rows, _weights = _generate_core(config)
    return rows


def _generate_core(config: dict | None = None):
    cfg = dict(GENERATION_CONFIG if config is None else config)
    seed = int(cfg["seed"])
    n_rows = int(cfg["n_rows"])
    n_features = int(cfg["n_features"])
    decimals = int(cfg["round_decimals"])

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_features))
    weights = _teacher_weights(n_features, cfg["teacher_scale"], rng)
    score = x @ weights

    centered_cos = np.mean(np.cos(x), axis=1) - float(np.exp(-0.5))
    latent_signal = score + float(cfg["nonlinearity_scale"]) * centered_cos

    sigma = float(cfg["noise_sigma"])
    margin = np.abs(score)
    hard = margin < float(cfg["hard_margin_sigma"]) * sigma
    ambiguous = (margin >= float(cfg["hard_margin_sigma"]) * sigma) & (
        margin < float(cfg["ambiguous_margin_sigma"]) * sigma
    )

    row_sigma = np.where(
        hard,
        sigma * float(cfg["hard_noise_multiplier"]),
        np.where(ambiguous, sigma, sigma * float(cfg["easy_noise_multiplier"])),
    )
    noise = rng.normal(size=n_rows) * row_sigma
    target = latent_signal + noise

    rows = []
    for index in range(n_rows):
        row = {"row_id": int(index)}
        for feature in range(n_features):
            row[f"x{feature}"] = _round(x[index, feature], decimals)
        if bool(hard[index]):
            regime = REGIME_HARD
        elif bool(ambiguous[index]):
            regime = REGIME_AMBIGUOUS
        else:
            regime = REGIME_EASY
        row["regime"] = regime
        row["latent_teacher_score"] = _round(score[index], decimals)
        row["latent_signal"] = _round(latent_signal[index], decimals)
        row["noise"] = _round(noise[index], decimals)
        row["noise_sigma"] = _round(row_sigma[index], decimals)
        row["y"] = _round(target[index], decimals)
        rows.append(row)
    return rows, weights


def dataset_data_hash(rows: list) -> str:
    """Exactly the payload test_30 hashes: canonical JSON of the rows."""
    return sha256_json(rows)


def observable_columns(config: dict | None = None) -> list:
    cfg = dict(GENERATION_CONFIG if config is None else config)
    return [f"x{feature}" for feature in range(int(cfg["n_features"]))]


def teacher_weights(config: dict | None = None) -> list:
    _rows, weights = _generate_core(config)
    return [round(float(value), 12) for value in weights]


def generate_task(config: dict | None = None) -> dict:
    """The ``task_truth`` object published in ``exec/walkthrough_results.json``."""
    cfg = dict(GENERATION_CONFIG if config is None else config)
    rows = generate_rows(cfg)
    latent = np.asarray([row["latent_signal"] for row in rows], dtype=float)
    target = np.asarray([row["y"] for row in rows], dtype=float)
    noise = np.asarray([row["noise"] for row in rows], dtype=float)

    signal_variance = float(np.var(latent))
    target_variance = float(np.var(target))
    signal_strength = signal_variance / target_variance
    if not 0.0 < signal_strength < 1.0:
        raise ValueError(f"signal_strength out of range: {signal_strength}")
    irreducible_noise = float(np.mean(noise ** 2))
    if not irreducible_noise > 0.0:
        raise ValueError("irreducible noise must be positive")

    regime_counts = {name: 0 for name in (REGIME_HARD, REGIME_AMBIGUOUS, REGIME_EASY)}
    for row in rows:
        regime_counts[row["regime"]] += 1
    if regime_counts[REGIME_HARD] == 0 or regime_counts[REGIME_AMBIGUOUS] == 0:
        raise ValueError("both hard and ambiguous regimes must be populated")

    truth_columns = ["latent_teacher_score", "latent_signal", "noise", "noise_sigma", "regime"]
    return {
        "task_kind": "teacher_student",
        "dataset_rows": rows,
        "n_rows": len(rows),
        "observable_feature_count": int(cfg["n_features"]),
        "observable_columns": observable_columns(cfg),
        "truth_columns": truth_columns,
        "truth_description": (
            "y = x . w_star + nonlinearity_scale * (mean_j cos(x_j) - exp(-1/2)) + noise, "
            "with per-row noise sigma depending on the latent margin |x . w_star|. "
            "The latent truth (latent_teacher_score, latent_signal, noise, regime) is "
            "recorded but is never an input to training or to the online diagnosis."
        ),
        "known_optimum": {
            "teacher_weights": teacher_weights(cfg),
            "intercept": 0.0,
            "objective": "mean_squared_error",
            "linear_part": "x . w_star",
            "note": (
                "w_star is the known optimum of the linear part; the small cosine term keeps "
                "the task from being a purely separable linear demo"
            ),
        },
        "irreducible_noise": round(irreducible_noise, 10),
        "signal_strength": round(signal_strength, 10),
        "signal_variance": round(signal_variance, 10),
        "target_variance": round(target_variance, 10),
        "hard_regimes": [REGIME_HARD],
        "ambiguous_regimes": [REGIME_AMBIGUOUS],
        "easy_regimes": [REGIME_EASY],
        "regime_counts": regime_counts,
        "data_hash": dataset_data_hash(rows),
        "config_hash": generation_config_hash(),
        "generation_config": cfg,
    }
