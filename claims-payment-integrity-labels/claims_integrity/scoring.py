"""Pre-processing, splits and scorers for the claim stream.

Pre-processing builds provider-level features **as of the claim's adjudication
day** (no peeking): only strictly earlier claims of the same provider enter an
aggregate.  ``no_leakage_probe`` mutates the future of a random claim and
rebuilds the matrix to prove the row is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

from .ccem import fit_ccem


def asof_features(
    billed: np.ndarray,
    observed_overpayment: np.ndarray,
    provider_id: np.ndarray,
    adjudication_day: np.ndarray,
    claim_features: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Claim-level attributes plus strictly-past provider aggregates."""

    billed = np.asarray(billed, dtype=float)
    observed = np.asarray(observed_overpayment, dtype=float)
    pid = np.asarray(provider_id)
    day = np.asarray(adjudication_day, dtype=float)
    x = np.asarray(claim_features, dtype=float)
    n = billed.size

    past_count = np.zeros(n)
    past_billed = np.zeros(n)
    past_overpay = np.zeros(n)
    for provider in np.unique(pid):
        rows = np.flatnonzero(pid == provider)
        rows = rows[np.argsort(day[rows], kind="stable")]
        cum_count = np.concatenate([[0.0], np.cumsum(np.ones(rows.size))[:-1]])
        cum_billed = np.concatenate([[0.0], np.cumsum(billed[rows])[:-1]])
        seen = np.where(observed[rows] >= 0, (observed[rows] > 0).astype(float), 0.0)
        cum_seen = np.concatenate([[0.0], np.cumsum(seen)[:-1]])
        past_count[rows] = cum_count
        past_billed[rows] = cum_billed
        past_overpay[rows] = cum_seen

    rate = np.where(past_count > 0, past_overpay / np.maximum(past_count, 1.0), 0.0)
    matrix = np.column_stack(
        [
            x,
            np.log1p(billed),
            past_count,
            past_billed / np.maximum(past_count, 1.0),
            rate,
        ]
    )
    names = [f"x{i}" for i in range(x.shape[1])] + [
        "log_billed",
        "provider_past_claims",
        "provider_past_billed_mean",
        "provider_past_overpay_rate",
    ]
    return matrix, names


def build_features(stream: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    return asof_features(
        stream["billed"],
        stream["observed_overpayment"],
        stream["provider_id"],
        stream["adjudication_day"],
        stream["features"],
    )


def no_leakage_probe(stream: dict[str, Any], probes: int = 25, seed: int = 7) -> dict[str, Any]:
    """Prove as-of features do not depend on any future claim.

    For random probe claims, every strictly-later claim's billed dollars and
    observed verdicts are replaced by new values; the probe rows must be
    bit-identical.
    """

    rng = np.random.default_rng(seed)
    base, _ = build_features(stream)
    billed = np.asarray(stream["billed"], dtype=float).copy()
    observed = np.asarray(stream["observed_overpayment"]).copy()
    day = np.asarray(stream["adjudication_day"])
    pid = np.asarray(stream["provider_id"])
    x = np.asarray(stream["features"], dtype=float)
    probes_found = 0
    checked = 0
    for idx in rng.choice(day.size, size=min(60, day.size), replace=False):
        future = (day > day[idx]) & (pid == pid[idx])
        if not future.any():
            continue
        probes_found += 1
        mutated_billed = billed.copy()
        mutated_observed = observed.copy()
        mutated_billed[future] = rng.lognormal(0.0, 2.0, size=int(future.sum()))
        mutated_observed[future] = rng.integers(-1, 2, size=int(future.sum()))
        mutated, _ = asof_features(mutated_billed, mutated_observed, pid, day, x)
        if not np.array_equal(base[idx], mutated[idx]):
            return {"ok": False, "probe": int(idx), "checked": checked + 1, "probes_with_future": probes_found}
        checked += 1
        if checked >= probes:
            break
    return {"ok": True, "checked": checked, "probes_with_future": probes_found}


def three_way_split(n: int, seed: int, fractions: tuple[float, float, float] = (0.4, 0.2, 0.4)) -> dict[str, np.ndarray]:
    """Disjoint train / tune / certification splits (SCoRC Assumption 6)."""

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_tune = int(round(fractions[1] * n))
    return {
        "train": perm[:n_train],
        "tune": perm[n_train : n_train + n_tune],
        "cert": perm[n_train + n_tune :],
    }


def fit_naive_scorer(x_train: np.ndarray, y_naive: np.ndarray, mask: np.ndarray, seed: int) -> LogisticRegression:
    """Trust the single reviewer verdict: logistic regression on noisy labels."""

    rows = np.asarray(mask, dtype=bool) & (np.asarray(y_naive) >= 0)
    if rows.sum() < 10 or len(np.unique(np.asarray(y_naive)[rows])) < 2:
        model = LogisticRegression(random_state=seed, max_iter=500)
        model.fit(np.asarray(x_train)[rows] if rows.sum() else np.zeros((2, x_train.shape[1])), np.array([0, 1]))
        return model
    model = LogisticRegression(random_state=seed, max_iter=1000, C=1.0)
    model.fit(np.asarray(x_train)[rows], np.asarray(y_naive)[rows].astype(int))
    return model


def fit_noise_aware_scorer(
    x_train: np.ndarray,
    labels_train: np.ndarray,
    k: int,
    n_reviewers: int,
    seed: int,
    prior: np.ndarray,
) -> dict[str, Any]:
    """CCEM posterior on the training split (noise-aware soft-label model)."""

    from .ccem import vote_posterior

    warm = vote_posterior(labels_train, k, prior)
    return fit_ccem(
        x_train,
        labels_train,
        k=k,
        n_reviewers=n_reviewers,
        seed=seed,
        n_restarts=2,
        posterior_init=warm,
    )


def ccem_posterior(fit: dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Apply a fitted CCEM model to new features and return the full posterior."""

    x = (np.asarray(features, dtype=float) - fit["feature_mean"]) / fit["feature_std"]
    k = fit["posterior"].shape[1]
    d = x.shape[1]
    params = fit["params"]
    w = params[: d * k].reshape(d, k)
    b = params[d * k : d * k + k]
    logits = x @ w + b
    shifted = logits - logits.max(axis=1, keepdims=True)
    posterior = np.exp(shifted)
    return posterior / posterior.sum(axis=1, keepdims=True)


def score_matrix(fit: dict[str, Any], features: np.ndarray, class_index: int = 1) -> np.ndarray:
    """P(fitted latent class) for new features."""

    return ccem_posterior(fit, features)[:, class_index]


def identify_clean_class(
    fit: dict[str, Any],
    tune_features: np.ndarray,
    tune_labels: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Post-hoc identification of the fitted class meaning (Theorem 1 permutation).

    CCEM is identified only up to a class permutation, so a deployed scorer must
    map fitted classes to observable verdict classes.  We match fitted argmax
    classes to the tune split's reviewer verdicts with the Hungarian algorithm
    (never the certification split), and return ``(clean_class, mapping)`` where
    ``mapping[fitted_class] = observed_verdict_class``.
    """

    posterior = ccem_posterior(fit, tune_features)
    fitted = posterior.argmax(axis=1)
    labels = np.asarray(tune_labels).astype(int)
    valid = labels >= 0
    k = posterior.shape[1]
    counts = np.zeros((k, k + 1), dtype=float)
    np.add.at(counts, (fitted[valid], labels[valid]), 1.0)
    row, col = linear_sum_assignment(-counts[:, :k])
    mapping = np.arange(k)
    mapping[row] = col
    clean_rows = np.flatnonzero(mapping == 0)
    if clean_rows.size:
        clean_class = int(clean_rows[0])
    else:
        counts_per_class = np.bincount(fitted, minlength=k)
        clean_class = int(np.argmax(counts_per_class))
    return clean_class, mapping


def overpayment_score(
    posterior_or_model,
    features: np.ndarray | None = None,
    clean_class: int | None = None,
) -> np.ndarray:
    """P(overpaid) under either a fitted CCEM model or a raw posterior matrix.

    For a fitted model, pass the class index identified by
    :func:`identify_clean_class` (defaults to 0; raw class indices are
    permutation-arbitrary).
    """

    if isinstance(posterior_or_model, dict):
        assert features is not None, "features are required to score new data with a fitted model"
        index = 0 if clean_class is None else int(clean_class)
        return 1.0 - ccem_posterior(posterior_or_model, features)[:, index]
    posterior = np.asarray(posterior_or_model, dtype=float)
    index = 0 if clean_class is None else int(clean_class)
    return 1.0 - posterior[:, index]
