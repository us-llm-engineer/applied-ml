"""CCEM: coupled cross-entropy minimization for noisy reviewer labels.

Source-backed: Ibrahim, Nguyen, Fu (2023), arXiv:2306.03288.
  Eq. (1)-(2):  observed response distribution ``p_n^(m) = A_m f(x_n)``
  Eq. (3):      categorical draw of the response
  Eq. (6a-b):   min over ``f`` and column-stochastic ``A_m`` of the empirical
                negative log-likelihood of the observed annotations.
  Theorem 1:    finite-sample identification *up to a label permutation*.

Anything about the synthetic generator, the specialist queue or the weak-anchor
stress is *derived here*.

Array convention: the public helpers take ``confusion[r, latent, observed]``
(each row a distribution over observed labels) because that is this package's
convention; this is the transpose of the paper's ``A_m`` whose columns are
distributions over responses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from .synthetic import make_claim_stream

_EPS = 1e-300


def ccem_observed_probability(posterior: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """``Pr(observed = y | x)`` for every item / reviewer / observed class.

    Implements ``p_n^(m) = A_m f(x_n)`` (Eq. 2) with ``confusion[r, z, y]``.
    """

    p = np.asarray(posterior, dtype=float)
    c = np.asarray(confusion, dtype=float)
    return np.einsum("iz,rzy->iry", p, c)


def ccem_objective(
    posterior: np.ndarray,
    observed_labels: np.ndarray,
    confusion: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean negative log-likelihood of the observed labels (Eq. 6a).

    Entries with ``label < 0`` (not reviewed) are skipped when ``mask`` is not
    given.  Probabilities are floored so impossible observations stay finite.
    """

    probs = ccem_observed_probability(posterior, confusion)
    y = np.asarray(observed_labels).astype(int)
    if mask is None:
        mask = y >= 0
    mask = np.asarray(mask, dtype=bool) & (y >= 0)
    if not mask.any():
        return float("nan")
    picked = np.take_along_axis(probs, np.clip(y, 0, probs.shape[2] - 1)[:, :, None], axis=2)[:, :, 0]
    picked = np.clip(picked, _EPS, None)
    return float(-np.mean(np.log(picked[mask])))


def _hungarian_mapping(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Class alignment by maximum cross-agreement.

    Returns ``mapping`` with ``mapping[estimate_class] = reference_class``.
    """

    ref = np.asarray(reference)
    est = np.asarray(estimate)
    k = max(int(ref.max()) + 1, int(est.max()) + 1)
    counts = np.zeros((k, k), dtype=float)
    np.add.at(counts, (est.ravel(), ref.ravel()), 1.0)
    row, col = linear_sum_assignment(-counts)
    mapping = np.arange(k)
    mapping[row] = col
    return mapping


def best_label_permutation(reference: np.ndarray, estimate: np.ndarray):
    """Align ``estimate`` labels to ``reference`` up to a label permutation.

    Returns ``(permutation, aligned_estimate)`` where ``permutation[e]`` is the
    reference class matched to estimate class ``e``.  Matching classes with
    identical agreement are broken by the smallest class index (deterministic).
    """

    ref = np.asarray(reference).ravel()
    est = np.asarray(estimate).ravel()
    mapping = _hungarian_mapping(ref, est)
    aligned = mapping[est]
    return tuple(int(v) for v in mapping), aligned


def align_posteriors(true_posterior: np.ndarray, estimate_posterior: np.ndarray):
    """Permutation-align a *posterior* estimate to the known truth (Theorem 1)."""

    true = np.asarray(true_posterior, dtype=float)
    est = np.asarray(estimate_posterior, dtype=float)
    cross = true.T @ est
    row, col = linear_sum_assignment(-cross)
    mapping = np.arange(est.shape[1])
    mapping[row] = col
    aligned = est[:, mapping]
    return tuple(int(v) for v in mapping), aligned


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_ccem(
    features: np.ndarray,
    labels: np.ndarray,
    k: int,
    n_reviewers: int,
    seed: int = 0,
    max_iter: int = 300,
    l2: float = 1e-4,
    l2_confusion: float = 5e-2,
    confusion_prior: str = "identity",
    n_restarts: int = 1,
    posterior_init: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit the CCEM model (Eq. 6) with a linear-softmax classifier.

    ``labels[n, r] < 0`` marks an unobserved item/reviewer pair.  The returned
    ``posterior`` is ``f_hat(x)`` and ``confusion[r]`` is ``A_hat_r^T`` with
    rows over observed labels (same convention as the public helpers).

    The confusion matrices are shrunk toward a reference logit matrix chosen
    by ``confusion_prior``: ``"identity"`` (a near-class-specialist prior in
    the spirit of the paper's NCSA) or ``"uniform"`` (a maximum-entropy
    prior on annotators).  Without this regularization the unregularized MLE
    slides into identifiability-degenerate solutions whose likelihood is fine
    but whose latent ordering is not recovered.
    """

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels).astype(int)
    n, d = x.shape
    mask = y >= 0
    n_annotations = int(mask.sum())
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0
    xs = (x - mean) / std

    best: dict[str, Any] | None = None
    for restart in range(max(1, n_restarts)):
        rng = np.random.default_rng(seed + 1000 * restart)
        w0 = 0.05 * rng.standard_normal((d, k))
        if posterior_init is not None and restart == 0:
            # Warm start: regress the standardized features onto the log of a
            # reference posterior, then let L-BFGS refine it.
            logits = np.log(np.clip(posterior_init, 1e-6, None))
            w0 = np.linalg.lstsq(xs, logits, rcond=None)[0]
        b0 = np.zeros(k)
        if confusion_prior not in {"identity", "uniform"}:
            raise ValueError("confusion_prior must be 'identity' or 'uniform'")
        u_ref = 2.5 * np.eye(k)[None, :, :] if confusion_prior == "identity" else np.zeros((1, k, k))
        u0 = u_ref + 0.05 * rng.standard_normal((n_reviewers, k, k))
        x0 = np.concatenate([w0.ravel(), b0, u0.ravel()])

        def unpack(params: np.ndarray):
            idx = 0
            w = params[idx : idx + d * k].reshape(d, k)
            idx += d * k
            b = params[idx : idx + k]
            idx += k
            u = params[idx:].reshape(n_reviewers, k, k)
            return w, b, u

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            w, b, u = unpack(params)
            a = _softmax(u.reshape(-1, k)).reshape(n_reviewers, k, k)  # rows over observed
            p = _softmax(xs @ w + b)
            probs = ccem_observed_probability(p, a)
            picked = np.take_along_axis(
                probs, np.clip(y, 0, k - 1)[:, :, None], axis=2
            )[:, :, 0]
            picked = np.clip(picked, _EPS, None)
            g = np.where(mask, -1.0 / picked, 0.0) / max(n_annotations, 1)
            loss = float(-(np.log(picked) * mask).sum() / max(n_annotations, 1))
            loss += 0.5 * l2 * float((w * w).sum() + (b * b).sum())
            loss += 0.5 * l2_confusion * float(((u - u_ref) ** 2).sum())

            dp = np.zeros_like(p)
            da = np.zeros_like(a)
            for r in range(n_reviewers):
                yy = np.clip(y[:, r], 0, k - 1)
                cols = np.flatnonzero(mask[:, r])
                if cols.size:
                    dp += g[:, r][:, None] * a[r][:, yy].T
                    contrib = p * g[:, r][:, None]
                    for c in range(k):
                        sel = cols[yy[cols] == c]
                        if sel.size:
                            da[r, :, c] += contrib[sel].sum(axis=0)
            row_sum = (a * da).sum(axis=2, keepdims=True)
            du = a * (da - row_sum) + l2_confusion * (u - u_ref)
            dz = p * (dp - (p * dp).sum(axis=1, keepdims=True))
            dw = xs.T @ dz + l2 * w
            db = dz.sum(axis=0) + l2 * b
            grad = np.concatenate([dw.ravel(), db, du.ravel()])
            return loss, grad

        result = minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "maxls": 50, "ftol": 1e-12, "gtol": 1e-9},
        )
        w, b, u = unpack(result.x)
        a = _softmax(u.reshape(-1, k)).reshape(n_reviewers, k, k)
        posterior = _softmax(xs @ w + b)
        record = {
            "posterior": posterior,
            "confusion": a,
            "loss": float(result.fun),
            "n_annotations": n_annotations,
            "n_iterations": int(result.nit),
            "converged": bool(result.success),
            "feature_mean": mean,
            "feature_std": std,
            "params": result.x.copy(),
        }
        if best is None or record["loss"] < best["loss"]:
            best = record
    assert best is not None
    return best


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator, repeats: int = 2000) -> tuple[float, float]:
    if values.size < 2:
        value = float(np.mean(values)) if values.size else float("nan")
        return value, value
    draws = rng.choice(values, size=(repeats, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def vote_posterior(labels: np.ndarray, k: int, prior: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Soft-vote posterior from raw reviewer labels (warm-start helper only)."""

    y = np.asarray(labels)
    counts = np.stack([(y == c).sum(axis=1) for c in range(k)], axis=1).astype(float)
    prior = np.asarray(prior, dtype=float)
    post = (counts + alpha * prior[None, :]) / (counts.sum(axis=1, keepdims=True) + alpha)
    return post


def run_ccem_identifiability_experiment(
    seed: int,
    regime: str,
    n: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure aligned identifiability under one reviewer-coverage regime.

    ``anchored`` is the control (uniform reviewer assignment, cross-class
    coverage); ``specialist_weak_anchor`` is the stress arm (specialty queues,
    near-disconnected reviewer x class coverage).  Latent class priors, sample
    size, reviewer budget, model capacity and seed schedule are held constant;
    only the assignment mechanism changes.  All metrics are aligned with the
    Hungarian mapping before comparison because CCEM is identifiable only up to
    a class permutation (Theorem 1).
    """

    stream = make_claim_stream(seed=seed, n=n, regime=regime, config=config)
    warm = vote_posterior(stream["reviewer_label_matrix"], int(stream["config"]["k_latent"]), stream["class_prior"])
    fit = fit_ccem(
        stream["features"],
        stream["reviewer_label_matrix"],
        k=int(stream["config"]["k_latent"]),
        n_reviewers=int(stream["config"]["n_reviewers"]),
        seed=seed,
        n_restarts=2,
        posterior_init=warm,
    )
    permutation, aligned = align_posteriors(stream["latent_posterior_true"], fit["posterior"])
    true_post = stream["latent_posterior_true"]
    per_item = ((aligned - true_post) ** 2).sum(axis=1)
    aligned_error = float(per_item.mean())
    rng = np.random.default_rng(seed + 7)
    low, high = _bootstrap_interval(per_item, rng)

    true_confusion = stream["reviewer_confusion_true"]
    aligned_confusion = np.empty_like(fit["confusion"])
    order = np.asarray(permutation)
    aligned_confusion[:, order, :] = fit["confusion"]
    per_reviewer = np.abs(aligned_confusion - true_confusion).mean(axis=(1, 2))
    confusion_error = float(per_reviewer.mean())
    c_low, c_high = _bootstrap_interval(per_reviewer, rng)

    unresolved = float((1.0 - aligned.max(axis=1)).mean())
    return {
        "seed": int(seed),
        "regime": regime,
        "n": int(n),
        "sample_size": int(n),
        "aligned_identifiability_error": aligned_error,
        "aligned_latent_error": aligned_error,
        "latent_error": aligned_error,
        "error_low": low,
        "error_high": high,
        "confusion_error": confusion_error,
        "confusion_low": c_low,
        "confusion_high": c_high,
        "unresolved_mass": unresolved,
        "permutation": permutation,
        "n_annotations": fit["n_annotations"],
        "fit_loss": fit["loss"],
        "aligned_posterior": aligned,
    }
