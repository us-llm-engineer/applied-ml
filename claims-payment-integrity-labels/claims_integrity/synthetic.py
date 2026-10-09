"""Synthetic health-plan claim stream with known latent truth.

Everything in this module is *derived here*: the generator, the typologies, the
dollar proxy and the reviewer model are synthetic design choices, not claims
about real claims data.  The stream is deliberately hard: rare overpayment,
heavy-tailed recoverable dollars, genuinely ambiguous claims, imperfect and
selection-biased reviewer labels.

The CCEM reviewer model follows Ibrahim, Nguyen, Fu (2023), Eq. (1)-(3): the
observed label distribution is ``A_m f(x)`` with column-stochastic annotator
matrices.  Our arrays store the transpose convention used by this
package: ``confusion[r, latent_class, observed_label]`` (each row is a
distribution over observed labels), i.e. ``confusion[r] = A_r^T``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import CLASS_NAMES, CONFIG

REGIMES = ("anchored", "specialist_weak_anchor")


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def latent_posterior(features: np.ndarray, means: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Exact class posterior under shared-covariance Gaussian class models.

    With a shared identity covariance the log-posterior is linear in ``x``, so
    the true posterior lies exactly in the CCEM softmax function class
    (realizability, Ibrahim et al. Assumption 3).
    """

    x = np.asarray(features, dtype=float)
    mu = np.asarray(means, dtype=float)
    logits = np.log(np.asarray(prior, dtype=float))[None, :] + x @ mu.T - 0.5 * (mu**2).sum(axis=1)[None, :]
    return _softmax(logits, axis=1)


def _class_means(k: int, d: int, shift: float) -> np.ndarray:
    means = np.zeros((k, d), dtype=float)
    if k >= 2:
        means[1, 0:3] = shift
        means[1, 6] = 0.35 * shift
    if k >= 3:
        means[2, 3:6] = shift
        means[2, 7] = 0.35 * shift
    if k >= 4:
        means[3, 8:10] = shift
    return means


def _reviewer_confusion(rng: np.random.Generator, m: int, k: int, specialist_bias: float = 0.30) -> np.ndarray:
    """Column-stochastic annotator matrix, stored as [latent, observed] rows."""

    specialty = m % k
    kind = ("specialist", "specialist", "generalist", "generalist", "sloppy", "sharp")[m % 6]
    if kind == "specialist":
        own = 0.88
        bias = specialist_bias
    elif kind == "sloppy":
        own = 0.58
        bias = 0.0
    elif kind == "sharp":
        own = 0.85
        bias = 0.0
    else:
        own = 0.72
        bias = 0.0
    matrix = np.zeros((k, k), dtype=float)
    for true_class in range(k):
        if true_class == specialty:
            row = np.full(k, (1.0 - own) / max(k - 1, 1))
            row[specialty] = own
        elif kind == "specialist":
            # Accuracy-dominant on the non-specialty classes with a modest
            # specialty over-call bias; the weak-anchor arm later merges these
            # two columns for the specialty queue.
            row = np.zeros(k)
            row[true_class] = own - 0.16
            row[specialty] = bias
            remaining = 1.0 - row[true_class] - row[specialty]
            others = [c for c in range(k) if c not in (true_class, specialty)]
            for c in others:
                row[c] = remaining / max(len(others), 1)
        else:
            # Generalist / sloppy / sharp reviewers: accuracy-dominant on every
            # class with the remaining mass spread evenly.
            row = np.full(k, (1.0 - own) / max(k - 1, 1))
            row[true_class] = own
        matrix[true_class] = row
    jitter = rng.normal(0.0, 0.01, size=(k, k))
    matrix = np.clip(matrix + jitter, 1e-3, None)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def _provider_weights(rng: np.random.Generator, n_providers: int) -> np.ndarray:
    raw = rng.lognormal(mean=0.0, sigma=1.1, size=n_providers)
    return raw / raw.sum()


def make_claim_stream(
    seed: int,
    n: int,
    regime: str = "anchored",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate one synthetic claim stream.

    Parameters
    ----------
    seed, n:
        Reproducibility key and stream size.
    regime:
        ``anchored`` — reviewers are assigned uniformly at random, so every
        reviewer sees every latent class (cross-class coverage; CCEM
        Assumption 1 roughly holds).
        ``specialist_weak_anchor`` — each reviewer works a specialty queue
        built from the model-implied class score, so reviewer x latent-class
        bipartite coverage is near-disconnected (the stress arm).
    """

    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    cfg = dict(CONFIG if config is None else config)
    k = int(cfg["k_latent"])
    d = int(cfg["n_features"])
    m = int(cfg["n_reviewers"])
    prior = np.asarray(cfg["class_prior"], dtype=float)
    prior = prior[:k] / prior[:k].sum()
    rng = np.random.default_rng(seed)

    means = _class_means(k, d, float(cfg["feature_shift"]))
    y = rng.choice(k, size=n, p=prior)
    x = means[y] + rng.normal(0.0, 1.0, size=(n, d))
    true_post = latent_posterior(x, means, prior)
    ambiguous = (1.0 - true_post[:, 0]) > 0.15

    typology = np.where(y == 0, "compliant", np.where(y == 1, "billing_error", "medically_unnecessary"))

    multiplier = np.ones(n)
    multiplier[y == 1] = 1.45
    multiplier[y == 2] = 1.25
    billed = float(cfg["billed_median"]) * np.exp(rng.normal(0.0, float(cfg["billed_sigma"]), size=n)) * multiplier
    billed = np.round(billed, 2)
    recovery_rate = np.clip(rng.beta(2.5, 2.5, size=n), 0.05, 0.95)
    recoverable = np.round(np.where(y > 0, billed * recovery_rate, 0.0), 2)

    n_providers = max(8, n // 40)
    provider_id = rng.choice(n_providers, size=n, p=_provider_weights(rng, n_providers))
    adjudication_day = rng.integers(0, 365, size=n)

    confusion = np.stack(
        [_reviewer_confusion(rng, reviewer, k, float(cfg.get("specialist_bias", 0.30))) for reviewer in range(m)],
        axis=0,
    )
    if regime == "specialist_weak_anchor":
        # Weak-anchor stress (derived here): each specialist has an anchor only
        # for its own class; the two non-specialty classes are merged into one
        # response (columns averaged), so the reviewer cannot tell them apart.
        # With specialty queues and no shared items, the merged response cannot
        # be cross-checked by another reviewer group.
        merged = confusion.copy()
        for reviewer in range(m):
            specialty = reviewer % k
            others = [c for c in range(k) if c != specialty]
            if len(others) == 2:
                blend = 0.5 * (confusion[reviewer][:, others[0]] + confusion[reviewer][:, others[1]])
                merged[reviewer][:, others[0]] = blend
                merged[reviewer][:, others[1]] = blend
        confusion = merged

    reviewer_label_matrix = np.full((n, m), -1, dtype=int)
    reviewer_id = np.full(n, -1, dtype=int)
    if regime == "anchored":
        # Uniform random reviewer assignment: every reviewer sees every latent
        # class, so the reviewer x class bipartite graph is connected (CCEM
        # Assumption 1 roughly holds; NCSA/NAPA anchors available).
        slots = 3
        perm = rng.permuted(np.tile(np.arange(m), (n, 1)), axis=1)[:, :slots]
        assigned = np.zeros((n, m), dtype=bool)
        for slot in range(slots):
            assigned[np.arange(n), perm[:, slot]] = True
        reviewer_id = perm[:, 0]
    else:
        # Specialty queues: each reviewer works the claims whose model-implied
        # affinity for their specialty is highest.  The reviewer budget is the
        # same (quota per reviewer); what changes is that coverage is
        # concentrated and the specialist bias cannot be cross-checked.
        quota = (3 * n) // m
        available = np.ones(n, dtype=bool)
        gumbel = rng.gumbel(0.0, 1.0, size=(n, k))
        assigned = np.zeros((n, m), dtype=bool)
        for reviewer in range(m):
            specialty = reviewer % k
            score = np.log(np.clip(true_post[:, specialty], 1e-12, None)) + float(cfg.get("queue_noise", 0.35)) * gumbel[:, specialty]
            score = np.where(available, score, -np.inf)
            chosen = np.argpartition(-score, quota - 1)[:quota]
            available[chosen] = False
            assigned[chosen, reviewer] = True
            unset = reviewer_id[chosen] < 0
            reviewer_id[chosen[unset]] = reviewer
    for reviewer in range(m):
        items = np.flatnonzero(assigned[:, reviewer])
        if items.size == 0:
            continue
        truth = y[items]
        probs = confusion[reviewer][truth]
        draws = (np.cumsum(probs, axis=1) > rng.random((items.size, 1))).argmax(axis=1)
        reviewer_label_matrix[items, reviewer] = draws

    # A randomized audit arm: uniformly random reviewer on a random subsample,
    # independent of any scorer or specialty queue.
    audit_mask = rng.random(n) < 0.25
    audit_label_matrix = np.full((n, m), -1, dtype=int)
    audit_items = np.flatnonzero(audit_mask)
    audit_reviewer = rng.integers(0, m, size=audit_items.size)
    for reviewer in range(m):
        subset = audit_items[audit_reviewer == reviewer]
        if subset.size == 0:
            continue
        probs = confusion[reviewer][y[subset]]
        draws = (np.cumsum(probs, axis=1) > rng.random((subset.size, 1))).argmax(axis=1)
        audit_label_matrix[subset, reviewer] = draws

    single = reviewer_label_matrix[np.arange(n), np.clip(reviewer_id, 0, m - 1)]
    observed_overpayment = np.where(reviewer_id >= 0, (single > 0).astype(int), -1)
    # Public 1-D aggregate: the single trusted reviewer verdict per claim (-1 = never reviewed).
    reviewer_labels = np.where(reviewer_id >= 0, single, -1).astype(int)

    return {
        "seed": int(seed),
        "n": int(n),
        "regime": regime,
        "config": cfg,
        "class_names": CLASS_NAMES[:k],
        "features": x,
        "feature_means": means,
        "class_prior": prior,
        "y_latent": y,
        "latent_overpayment": y > 0,
        "latent_posterior_true": true_post,
        "ambiguous": ambiguous,
        "typology": typology,
        "billed": billed,
        "recoverable_dollars": recoverable,
        "provider_id": provider_id,
        "adjudication_day": adjudication_day,
        "reviewer_confusion_true": confusion,
        "reviewer_id": reviewer_id,
        "reviewer_labels": reviewer_labels,
        "reviewer_label_matrix": reviewer_label_matrix,
        "observed_overpayment": observed_overpayment,
        "audit_mask": audit_mask,
        "audit_label_matrix": audit_label_matrix,
        "audit_labels": np.where(
            audit_mask,
            audit_label_matrix[np.arange(n), np.clip(np.argmax(audit_label_matrix >= 0, axis=1), 0, m - 1)],
            -1,
        ).astype(int),
    }


def generate_synthetic_claims(seed: int, n: int):
    """Public contract: deterministic synthetic claim records."""

    return make_claim_stream(seed=seed, n=n, regime="anchored")


def stream_digest(stream: dict[str, Any]) -> str:
    """Content hash of the generated stream (used to prove reuse, not re-roll)."""

    import hashlib

    parts = [
        np.ascontiguousarray(stream["features"]),
        np.ascontiguousarray(stream["y_latent"]),
        np.ascontiguousarray(stream["recoverable_dollars"]),
        np.ascontiguousarray(stream["reviewer_label_matrix"]),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.tobytes())
    return digest.hexdigest()
