"""Train-only preprocessing for the NB2 synthetic task.

Project choice (derived here, not specified by the papers): a seeded 70/30
fit/evaluation split and a z-score scaler fitted on the fit rows only.  The
numbers the tests recompute are the split indices, the config hash and the hash
of the exported rows.

Nothing in this module ever looks at an evaluation row while fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_dynamics.walkthrough import task  # noqa: E402


def split_indices(n_rows: int, seed: int, fit_fraction: float) -> tuple:
    """Seeded permutation split; every row lands in exactly one side."""
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(int(n_rows))
    n_fit = int(round(int(n_rows) * float(fit_fraction)))
    fit_rows = sorted(int(index) for index in permutation[:n_fit])
    evaluation_rows = sorted(int(index) for index in permutation[n_fit:])
    return fit_rows, evaluation_rows


def feature_matrix(rows: list, indices: list, feature_columns: list) -> np.ndarray:
    return np.asarray(
        [[float(rows[index][column]) for column in feature_columns] for index in indices],
        dtype=float,
    )


def fit_scaler(rows: list, fit_rows: list, feature_columns: list) -> dict:
    """Mean/std from the fit rows only (population std, ddof=0)."""
    matrix = feature_matrix(rows, fit_rows, feature_columns)
    return {
        "feature_columns": list(feature_columns),
        "mean": [float(value) for value in matrix.mean(axis=0)],
        "std": [float(value) for value in matrix.std(axis=0)],
        "n_fit_rows": int(matrix.shape[0]),
    }


def leakage_probe(rows: list, fit_rows: list, feature_columns: list) -> dict:
    """How far an all-rows scaler would have drifted from the train-only one.

    Reported as evidence that the train-only fit matters; it is a diagnostic,
    never a scaling option.
    """
    train_only = fit_scaler(rows, fit_rows, feature_columns)
    all_rows = fit_scaler(rows, list(range(len(rows))), feature_columns)
    mean_shift = [
        abs(a - b) for a, b in zip(train_only["mean"], all_rows["mean"])
    ]
    scale_shift = [
        abs(a - b) for a, b in zip(train_only["std"], all_rows["std"])
    ]
    return {
        "all_rows_mean": all_rows["mean"],
        "all_rows_std": all_rows["std"],
        "max_abs_mean_shift": float(max(mean_shift)),
        "max_abs_std_shift": float(max(scale_shift)),
        "note": (
            "z-scoring with all-rows statistics shifts each feature mean by at most "
            f"{max(mean_shift):.4g}; the published pipeline fits on the fit rows only "
            "so no evaluation row influences the transform"
        ),
    }


def build_preprocessing(rows: list, fit_rows: list | None = None, evaluation_rows: list | None = None) -> dict:
    """The ``preprocessing`` object published in walkthrough_results.json."""
    config = task.generation_config()
    if fit_rows is None or evaluation_rows is None:
        fit_rows, evaluation_rows = split_indices(
            len(rows), config["seed"], config["fit_fraction"]
        )
    feature_columns = task.observable_columns(config)
    if set(fit_rows) & set(evaluation_rows):
        raise ValueError("fit and evaluation rows must be disjoint")
    if set(fit_rows) | set(evaluation_rows) != set(range(len(rows))):
        raise ValueError("the split must reach every generated row")
    scaler = fit_scaler(rows, fit_rows, feature_columns)
    probe = leakage_probe(rows, fit_rows, feature_columns)
    return {
        "fit_rows": [int(index) for index in fit_rows],
        "evaluation_rows": [int(index) for index in evaluation_rows],
        "fit_scope": "train_only",
        "seed": int(config["seed"]),
        "config_hash": task.generation_config_hash(),
        "data_hash": task.dataset_data_hash(rows),
        "n_fit_rows": len(fit_rows),
        "n_evaluation_rows": len(evaluation_rows),
        "normalization": config["normalization"],
        "scaler": {
            "fit_scope": "train_only",
            "mean": scaler["mean"],
            "std": scaler["std"],
            "feature_columns": scaler["feature_columns"],
        },
        "leakage_probe": probe,
        "note": (
            "evaluation rows are held out of every fitted statistic; the rows themselves "
            "stay in the published task so the two sides can be inspected"
        ),
    }


def standardize(rows: list, indices: list, preprocessing: dict) -> np.ndarray:
    """Apply the published train-only scaler to selected rows."""
    feature_columns = preprocessing["scaler"]["feature_columns"]
    matrix = feature_matrix(rows, indices, feature_columns)
    mean = np.asarray(preprocessing["scaler"]["mean"], dtype=float)
    std = np.asarray(preprocessing["scaler"]["std"], dtype=float)
    return (matrix - mean) / std
