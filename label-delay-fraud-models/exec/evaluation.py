"""C4: Bayesian reject inference as a deliberately conditional synthetic experiment.

Synthetic design
----------------
* Population: ``n_transactions`` i.i.d. applicants with ``x ~ N(0, 1)`` and a
  latent bad outcome ``Y ~ Bernoulli(sigmoid(beta * x))`` (``beta = 1.5``). The
  population is synthetic and fully known to the experiment, so the target
  ``population_truth = mean(Y)`` is available for scoring.  A lender approves
  iff ``x > t`` (``t = 0`` by default, overridable with ``config["approval_x"]``
  and ``config["beta"]``), and labels (``Y``) exist only for approved
  transactions: the accepted group is therefore a biased sample of the
  population.
* ``accepts_only`` estimates the target as ``mean(Y | accepted)`` -- the
  selection-biased operational view.
* ``informative_synthetic_prior`` places a Beta(a, b) prior over the rejected
  group's positive rate whose mean is the *mechanism* rate
  ``E[Y | x <= t]`` under the known data-generating process (computed by
  deterministic quadrature from the design constants, not from the sample).
  No rejected labels are observed, so the posterior mean equals the prior mean
  and the estimator is
  ``pi_acc * mean(Y | accepted) + pi_rej * prior_mean``.  Its error against the
  population truth is only the accepted-group sampling error, so it must beat
  accepts-only by a comfortable margin; :func:`_run` asserts that.
* ``wrong_prior`` repeats the same estimator with a deliberately wrong prior
  mean (0.5, displaced from the mechanism's rejected-group rate). It is
  reported as a finite ``derived_here`` measurement and no error direction is
  asserted.

``likelihood_offset`` is the signed offset of the scenario's prior mean from
the likelihood-implied (mechanism) rejected-group rate: 0.0 for the informative
prior and non-zero for the wrong prior.

Fully deterministic: one seeded population draw, no resampling.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from exec.config import config_hash, config_key, seed_of

BETA = 1.5
APPROVAL_X = 0.0
PRIOR_STRENGTH = 10.0
WRONG_PRIOR_MEAN = 0.5
PRIOR_CLIP: tuple[float, float] = (0.02, 0.98)
QUADRATURE_SPAN = 14.0
QUADRATURE_STEPS = 28001
SCENARIOS: tuple[str, ...] = ("informative_synthetic_prior", "wrong_prior")

ACCEPTS_ONLY_RULE = "mean(latent outcome) over accepted transactions only (selection-biased sample)"
BAYESIAN_RULE = (
    "accepted_share * mean(Y | accepted) + rejected_share * posterior_mean(rejected); "
    "no rejected labels are observed so the Beta posterior mean equals the prior mean"
)
POPULATION_RULE = "mean(Y) over the full synthetic population (known ground truth)"

_CACHE: dict[str, dict[str, Any]] = {}


def _standard_normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


def _mechanism_rejected_rate(beta: float, threshold: float) -> float:
    """``E[Y | x <= threshold]`` under the synthetic DGP, by quadrature."""
    # P(x > threshold) for x ~ N(0, 1) via the complementary error function.
    upper_mass = 0.5 * math.erfc(threshold / math.sqrt(2.0))
    if upper_mass <= 0.0:
        raise ValueError("approval threshold leaves no accepted mass")
    grid = np.linspace(threshold, threshold + QUADRATURE_SPAN, QUADRATURE_STEPS)
    density = _standard_normal_pdf(grid)
    positive_probability = 1.0 / (1.0 + np.exp(-beta * grid))
    marked_mass = _trapezoid(positive_probability * density, grid)
    accepted_rate = min(1.0, max(0.0, marked_mass / upper_mass))
    rejected_mass = 1.0 - upper_mass
    if rejected_mass <= 0.0:
        raise ValueError("approval threshold leaves no rejected mass")
    # Total E[Y] = 0.5 by symmetry of x and the sigmoid; rejected-group rate is
    # the residual mass over the rejected region.
    rejected_mass_value = max(0.0, 0.5 - marked_mass)
    return float(min(1.0, max(0.0, rejected_mass_value / rejected_mass)))


def run_evaluation_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic C4 Bayesian reject-inference experiment."""
    key = config_key(config)
    if key not in _CACHE:
        _CACHE[key] = _run(config)
    return copy.deepcopy(_CACHE[key])


def _run(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = seed_of(config)
    n_transactions = int(config["n_transactions"])
    threshold = float(config.get("approval_x", APPROVAL_X))
    beta = float(config.get("beta", BETA))
    prior_strength = float(config.get("prior_strength", PRIOR_STRENGTH))

    rng = np.random.default_rng([int(seed), 31])
    x = rng.standard_normal(n_transactions)
    latent = (rng.random(n_transactions) < 1.0 / (1.0 + np.exp(-beta * x))).astype(
        np.float64
    )
    accepted = x > threshold
    n_accepted = int(accepted.sum())
    n_rejected = int(n_transactions - n_accepted)
    if n_accepted == 0 or n_rejected == 0:
        raise ValueError(
            "the approval rule must leave both accepted and rejected transactions"
        )

    population_truth = float(latent.mean())
    accepted_rate = float(latent[accepted].mean())
    rejected_rate = float(latent[~accepted].mean())
    accepted_share = n_accepted / float(n_transactions)
    rejected_share = n_rejected / float(n_transactions)

    mechanism_rejected_rate = _mechanism_rejected_rate(beta, threshold)
    prior_means = {
        "informative_synthetic_prior": float(
            min(PRIOR_CLIP[1], max(PRIOR_CLIP[0], mechanism_rejected_rate))
        ),
        "wrong_prior": float(
            min(PRIOR_CLIP[1], max(PRIOR_CLIP[0], WRONG_PRIOR_MEAN))
        ),
    }

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        prior_mean = prior_means[scenario]
        prior_alpha = prior_strength * prior_mean
        prior_beta = prior_strength * (1.0 - prior_mean)
        # No rejected labels are observed, so the Beta posterior mean is the
        # prior mean; the likelihood offset records the prior's displacement
        # from the mechanism-implied rejected-group rate.
        posterior_rejected_mean = prior_alpha / (prior_alpha + prior_beta)
        bayesian_estimate = (
            accepted_share * accepted_rate + rejected_share * posterior_rejected_mean
        )
        bayesian_error = abs(bayesian_estimate - population_truth)
        accepts_only_error = abs(accepted_rate - population_truth)
        rows.append(
            {
                "scenario": scenario,
                "population_truth": population_truth,
                "bayesian_estimate": float(bayesian_estimate),
                "accepts_only_estimate": accepted_rate,
                "bayesian_absolute_error": float(bayesian_error),
                "accepts_only_absolute_error": float(accepts_only_error),
                "n_accepted": int(n_accepted),
                "n_rejected": int(n_rejected),
                "prior_mean": float(prior_mean),
                "likelihood_offset": float(prior_mean - mechanism_rejected_rate),
                "prior_strength": float(prior_strength),
                "prior_alpha": float(prior_alpha),
                "prior_beta": float(prior_beta),
                "posterior_rejected_mean": float(posterior_rejected_mean),
                "accepted_share": float(accepted_share),
                "rejected_share": float(rejected_share),
                "sample_accepted_rate": accepted_rate,
                "sample_rejected_rate": rejected_rate,
                "mechanism_rejected_rate": float(mechanism_rejected_rate),
                "margin_ratio": (
                    float(accepts_only_error / bayesian_error)
                    if bayesian_error > 0.0
                    else None
                ),
                "claim_status": "derived_here",
                "accepts_only_rule": ACCEPTS_ONLY_RULE,
                "bayesian_rule": BAYESIAN_RULE,
                "population_rule": POPULATION_RULE,
            }
        )

    informative = next(row for row in rows if row["scenario"] == "informative_synthetic_prior")
    wrong = next(row for row in rows if row["scenario"] == "wrong_prior")
    assert informative["bayesian_absolute_error"] < informative["accepts_only_absolute_error"], (
        "C4 regression: the deliberately informative synthetic prior must beat "
        "accepts-only against the known population truth."
    )
    assert 2.0 * informative["bayesian_absolute_error"] < informative["accepts_only_absolute_error"], (
        "C4 regression: the informative prior's win over accepts-only must be a "
        "comfortable margin, not a sampling fluke."
    )
    assert math.isfinite(wrong["bayesian_absolute_error"]), (
        "C4: the wrong-prior scenario must still report a finite measured error."
    )

    return {
        "scenarios": rows,
        "population_truth": population_truth,
        "approval_rule": f"approve iff x > {threshold:g}",
        "prior_rule": (
            "informative: Beta prior over the rejected group with mean equal to "
            "the synthetic rejection-mechanism rate E[Y | x <= t]; wrong: same "
            "shape with a deliberately displaced mean (0.5)"
        ),
        "n_transactions": int(n_transactions),
        "seed": int(seed),
        "config_hash": config_hash(config),
        "claim_status": "derived_here",
        "beats_accepts_only": bool(
            informative["bayesian_absolute_error"] < informative["accepts_only_absolute_error"]
        ),
    }


# ======================================================================================
# Round-2 (C14 core, C19 metric): reject-inference metrics and Algorithm 1.
#
# Appended per `exec/INTERFACES-R2.md`; the Round-1 C4 experiment above is untouched.
# Label convention: `1` = bad. Scores are predicted probabilities of bad; the rank-only
# metrics (AUC, PAUC, ABR) depend on the ordering of the scores only.
# ======================================================================================

R2_METRICS: tuple[str, ...] = ("auc", "brier", "pauc", "abr")
R2_PAUC_REGION = 0.2
R2_ABR_LOW = 0.2
R2_ABR_HIGH = 0.4
R2_ABR_GRID: np.ndarray = np.linspace(R2_ABR_LOW, R2_ABR_HIGH, 101)
R2_FPR_EPS = 1e-12


def _r2_arrays(labels: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    return y, s


def _r2_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC with half credit for tied positive/negative pairs."""
    n_pos = float(np.sum(y == 1.0))
    n_neg = float(np.sum(y == 0.0))
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 0.5
    ranks = rankdata(s, method="average")
    positive_rank_sum = float(np.sum(ranks[y == 1.0]))
    return (positive_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _r2_roc_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC vertices at every distinct score (descending), prepended with (0, 0)."""
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    n_pos = float(np.sum(y_sorted == 1.0))
    n_neg = float(np.sum(y_sorted == 0.0))
    cumulative_positive = np.cumsum(y_sorted == 1.0, dtype=float)
    cumulative_negative = np.cumsum(y_sorted == 0.0, dtype=float)
    boundaries = np.concatenate([np.flatnonzero(np.diff(s_sorted) != 0.0), [s_sorted.size - 1]])
    fpr = cumulative_negative[boundaries] / n_neg
    tpr = cumulative_positive[boundaries] / n_pos
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr])


def _r2_partial_auc(y: np.ndarray, s: np.ndarray, fpr_max: float = R2_PAUC_REGION) -> float:
    """Partial AUC over ``FPR in [0, fpr_max]`` (trapezoid, normalised by ``fpr_max``).

    Any normalisation is contract-free (see `tests/r2-interface-t1.md`); this one keeps
    the metric in [0, 1] with 1.0 for a perfect ranking. Rank-only.
    """
    n_pos = float(np.sum(y == 1.0))
    n_neg = float(np.sum(y == 0.0))
    if n_pos <= 0.0 or n_neg <= 0.0:
        return 0.5
    fpr, tpr = _r2_roc_points(y, s)
    right = int(np.searchsorted(fpr, fpr_max, side="right"))
    area = 0.0
    if right >= 2:
        area = float(np.sum(0.5 * (fpr[1:right] - fpr[: right - 1]) * (tpr[1:right] + tpr[: right - 1])))
    if right < fpr.size:
        x0, y0 = float(fpr[right - 1]), float(tpr[right - 1])
        x1, y1 = float(fpr[right]), float(tpr[right])
        tpr_at_max = y0 if x1 <= x0 else y0 + (y1 - y0) * (fpr_max - x0) / (x1 - x0)
        area += 0.5 * (fpr_max - x0) * (y0 + tpr_at_max)
    return area / fpr_max


def _r2_abr_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    """Bad rate among the ``k`` lowest-score applicants (ties broken by stable order)."""
    n = int(y.size)
    if n == 0:
        return 0.0
    k = int(min(max(k, 1), n))
    order = np.argsort(s, kind="mergesort")
    return float(np.cumsum(y[order] == 1.0)[k - 1] / k)


def _r2_abr(y: np.ndarray, s: np.ndarray) -> float:
    """Mean of :func:`abr_at` over the acceptance grid in [0.2, 0.4]."""
    n = int(y.size)
    if n == 0:
        return 0.0
    order = np.argsort(s, kind="mergesort")
    cumulative = np.cumsum(y[order] == 1.0)
    ks = [int(min(max(round(float(rate) * n), 1), n)) for rate in R2_ABR_GRID]
    values = [float(cumulative[k - 1] / k) for k in ks]
    return float(np.mean(values))


def _r2_metric_value(metric: str, y: np.ndarray, s: np.ndarray) -> float:
    if metric == "auc":
        return _r2_auc(y, s)
    if metric == "brier":
        return float(np.mean((s - y) ** 2))
    if metric == "pauc":
        return _r2_partial_auc(y, s)
    if metric == "abr":
        return _r2_abr(y, s)
    raise ValueError(f"unknown metric {metric!r}; expected one of {R2_METRICS}")


def abr_at(labels: Any, scores: Any, acceptance_rate: float) -> float:
    """Bad rate among the ``round(acceptance_rate * n)`` lowest-score applicants.

    The lowest-risk share is the lowest *score* share (``1`` = bad); the result is a
    plain Python float and depends on the score ranking, never on the row order.
    """
    y, s = _r2_arrays(labels, scores)
    n = int(y.size)
    if n == 0:
        return 0.0
    k = int(min(max(round(float(acceptance_rate) * n), 1), n))
    return _r2_abr_at_k(y, s, k)


def compute_metrics(labels: Any, scores: Any) -> dict[str, float]:
    """The four C14 metrics on one labelled sample (label ``1`` = bad).

    * ``auc`` -- ROC AUC, half credit for tied positive/negative pairs.
    * ``brier`` -- mean squared error of the bad-probability score.
    * ``pauc`` -- partial AUC over the low-FPR region (normalised), higher better.
    * ``abr`` -- bad rate among the lowest-score accepts, integrated over the 20%-40%
      acceptance band, lower better.

    ``auc``, ``pauc`` and ``abr`` are rank-only; only ``brier`` uses the score scale.
    """
    y, s = _r2_arrays(labels, scores)
    return {name: float(_r2_metric_value(name, y, s)) for name in R2_METRICS}


_R2_BAYES_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def bayesian_metric(
    labels_accept: Any,
    scores_accept: Any,
    scores_reject: Any,
    prior_reject: Any,
    metric: str,
    seed: int = 1,
    tolerance: float = 1e-6,
    min_draws: int = 20,
    max_draws: int = 60,
) -> dict[str, Any]:
    """Kozodoi's Algorithm 1: repeatedly pseudo-label the rejects from ``prior_reject``.

    Each draw samples one pseudo-label per reject as ``Bernoulli(prior_reject)`` from a
    seeded generator, appends them to the accepts (whose labels are observed), and
    evaluates ``metric`` on the completed data. The estimate ``value`` is the running
    mean of the draws; the loop stops only once ``n_draws >= min_draws`` *and* the
    running mean's last change is within ``tolerance``, otherwise it runs to
    ``max_draws``. Returns plain Python/JSON-serialisable values; the same seed always
    reproduces the same draw sequence, a looser tolerance never needs more draws.
    """
    if metric not in R2_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {R2_METRICS}")
    y_a = np.asarray(labels_accept, dtype=float).reshape(-1)
    s_a = np.asarray(scores_accept, dtype=float).reshape(-1)
    s_r = np.asarray(scores_reject, dtype=float).reshape(-1)
    prior = np.asarray(prior_reject, dtype=float).reshape(-1)
    if y_a.size != s_a.size:
        raise ValueError("labels_accept and scores_accept must have the same length")
    if s_r.size != prior.size:
        raise ValueError("scores_reject and prior_reject must have the same length")
    min_draws = max(1, int(min_draws))
    max_draws = max(min_draws, int(max_draws))
    tolerance = abs(float(tolerance))
    key = (
        metric,
        int(seed),
        tolerance,
        min_draws,
        max_draws,
        tuple(y_a.tolist()),
        tuple(s_a.tolist()),
        tuple(s_r.tolist()),
        tuple(prior.tolist()),
    )
    cached = _R2_BAYES_CACHE.get(key)
    if cached is not None:
        return copy.deepcopy(cached)

    n_accept = int(y_a.size)
    n_reject = int(s_r.size)
    labels_full = np.concatenate([y_a, np.zeros(n_reject, dtype=float)])
    scores_full = np.concatenate([s_a, s_r])
    rng = np.random.default_rng([int(seed), 0])
    draw_values: list[float] = []
    running_mean: list[float] = []
    converged = False
    for draw_index in range(1, max_draws + 1):
        pseudo = (rng.random(n_reject) < prior).astype(float)
        labels_full[n_accept:] = pseudo
        draw_values.append(float(_r2_metric_value(metric, labels_full, scores_full)))
        running_mean.append(float(np.mean(draw_values)))
        if draw_index >= min_draws:
            delta = running_mean[-1] - running_mean[-2]
            if abs(delta) <= tolerance:
                converged = True
                break
    result = {
        "value": float(running_mean[-1]),
        "n_draws": int(len(draw_values)),
        "converged": bool(converged),
        "draw_values": draw_values,
        "running_mean": running_mean,
    }
    _R2_BAYES_CACHE[key] = result
    return copy.deepcopy(result)


def _r2_fit_scorecard(features: Any, labels: Any) -> tuple[Any, float]:
    """Deterministic, unpenalised logistic scorecard trained on the given rows only.

    Returns ``(model, constant)``; ``model`` is ``None`` when the training labels carry
    a single class, in which case ``constant`` (the observed rate) is used instead. The
    unpenalised MLE with an intercept is exactly calibrated on its training rows, so
    ``mean(predicted) == mean(labels)`` up to the optimiser's tolerance.
    """
    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if x.reshape(x.shape[0], -1).shape[0] != y.size:
        raise ValueError("features and labels must have the same number of rows")
    if y.size == 0:
        return None, 0.5
    if np.unique(y).size < 2:
        return None, float(np.clip(y.mean(), 0.0, 1.0))
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=5000, tol=1e-10)
    model.fit(np.asarray(features, dtype=float), y)
    return model, 0.0


def _r2_scorecard_predict(model: Any, constant: float, features: Any) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.size == 0:
        return np.zeros(0, dtype=float)
    if model is None:
        return np.full(x.shape[0], float(constant))
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def reject_prior(stream: Mapping[str, Any]) -> list[float]:
    """One prior P(y_reject | X_reject) per reject, from an accepts-trained scorecard.

    The scorecard is a logistic regression fitted on ``X_accept``/``y_accept`` ONLY and
    calibrated by that fit; the reject features are then scored. The latent reject and
    holdout outcomes are never read, so changing them cannot change this list.
    """
    model, constant = _r2_fit_scorecard(stream["X_accept"], stream["y_accept"])
    prior = _r2_scorecard_predict(model, constant, stream["X_reject"])
    return [float(value) for value in np.clip(prior, 0.0, 1.0)]


def _r2_corrupt_prior(prior_clean: list[float], fraction: float, seed: int) -> list[float]:
    """Keep or invert each prior; exactly ``round(fraction * n)`` entries are inverted."""
    n = len(prior_clean)
    fraction = min(1.0, max(0.0, float(fraction)))
    n_flip = int(round(fraction * n))
    if n_flip <= 0 or n <= 0:
        return list(prior_clean)
    prior = list(prior_clean)
    rng = np.random.default_rng([int(seed), 7])
    for index in rng.permutation(n)[:n_flip]:
        position = int(index)
        prior[position] = 1.0 - prior[position]
    return prior


def run_reject_inference_evaluation(stream: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """C14: one accept-trained scorecard plus the Bayesian (Algorithm 1) evaluation.

    Uses only the accepts' labels, the feature matrices, and the seeded pseudo-labels.
    ``scores_accept``/``scores_reject``/``scores_holdout`` come from one deterministic
    logistic scorecard fitted on ``X_accept``/``y_accept`` (exactly calibrated on the
    accepts); ``prior_clean`` is ``reject_prior(stream)`` and ``prior`` is its
    ``prior_flip_fraction`` corruption (each entry kept or inverted, the inverted set
    chosen by the config seed). ``accepts_only`` reuses :func:`compute_metrics` and
    ``bayesian``/``n_draws`` come from :func:`bayesian_metric` per metric.
    """
    seed = int(config.get("seed", 1))
    tolerance = float(config.get("tolerance", 1e-6))
    min_draws = int(config.get("min_draws", 20))
    max_draws = int(config.get("max_draws", 60))
    flip_fraction = float(config.get("prior_flip_fraction", 0.0))

    y_accept = np.asarray(stream["y_accept"], dtype=float).reshape(-1)
    model, constant = _r2_fit_scorecard(stream["X_accept"], y_accept)
    scores_accept = np.clip(_r2_scorecard_predict(model, constant, stream["X_accept"]), 0.0, 1.0)
    scores_reject = np.clip(_r2_scorecard_predict(model, constant, stream["X_reject"]), 0.0, 1.0)
    scores_holdout = np.clip(_r2_scorecard_predict(model, constant, stream["X_holdout"]), 0.0, 1.0)

    prior_clean = reject_prior(stream)
    prior = _r2_corrupt_prior(prior_clean, flip_fraction, seed)

    accepts_only = compute_metrics(y_accept, scores_accept)
    bayesian: dict[str, float] = {}
    n_draws: dict[str, int] = {}
    converged: dict[str, bool] = {}
    for metric in R2_METRICS:
        draw = bayesian_metric(
            labels_accept=y_accept,
            scores_accept=scores_accept,
            scores_reject=scores_reject,
            prior_reject=prior,
            metric=metric,
            seed=seed,
            tolerance=tolerance,
            min_draws=min_draws,
            max_draws=max_draws,
        )
        bayesian[metric] = float(draw["value"])
        n_draws[metric] = int(draw["n_draws"])
        converged[metric] = bool(draw["converged"])

    return {
        "scores_accept": [float(value) for value in scores_accept],
        "scores_reject": [float(value) for value in scores_reject],
        "scores_holdout": [float(value) for value in scores_holdout],
        "prior": [float(value) for value in prior],
        "prior_clean": [float(value) for value in prior_clean],
        "prior_flip_fraction": float(flip_fraction),
        "accepts_only": accepts_only,
        "bayesian": bayesian,
        "n_draws": n_draws,
        "converged": converged,
        "seed": seed,
        "tolerance": tolerance,
        "min_draws": min_draws,
        "max_draws": max_draws,
    }


def precision_recall_at_fpr(labels: Any, scores: Any, fpr: float) -> dict[str, float]:
    """C19: the lowest score threshold whose flagged-negative share is at most ``fpr``.

    The flag rule is ``score >= threshold`` and the threshold is chosen over the
    distinct observed scores; because the flagged-negative share is non-decreasing as
    the threshold falls, the selected threshold is the last one that still meets the
    target, so with tied scores the achieved FPR is never pushed above ``fpr``.
    Returns ``threshold``, ``precision`` among the flagged, ``recall`` of the
    positives, and ``achieved_fpr`` -- all plain floats, rank/order-invariant.
    """
    y, s = _r2_arrays(labels, scores)
    target = float(fpr)
    n_neg = int(np.sum(y == 0.0))
    n_pos = int(np.sum(y == 1.0))
    if y.size == 0 or n_neg <= 0:
        raise ValueError("precision_recall_at_fpr needs at least one negative example")
    best_threshold: float | None = None
    best_true_positive = 0
    best_false_positive = 0
    for threshold in np.unique(s)[::-1]:
        flagged = s >= threshold
        false_positive = int(np.sum(flagged & (y == 0.0)))
        if false_positive / n_neg <= target + R2_FPR_EPS:
            best_threshold = float(threshold)
            best_true_positive = int(np.sum(flagged & (y == 1.0)))
            best_false_positive = false_positive
        else:
            break
    if best_threshold is None:
        return {"threshold": float(np.max(s)), "precision": 0.0, "recall": 0.0, "achieved_fpr": 0.0}
    flagged_total = best_true_positive + best_false_positive
    return {
        "threshold": best_threshold,
        "precision": float(best_true_positive / flagged_total) if flagged_total > 0 else 0.0,
        "recall": float(best_true_positive / n_pos) if n_pos > 0 else 0.0,
        "achieved_fpr": float(best_false_positive / n_neg),
    }


# ======================================================================================
# Round-2 (C14): the derived-here reject-inference stream and the paired study.
#
# Appended per `exec/INTERFACES-R2.md`. Applicant approval uses a feature-only initial
# risk score (never the latent outcome); latent reject/holdout outcomes are read only by
# the study's holdout truth and by BASL's final comparison. Fully deterministic.
# ======================================================================================

_R2_STREAM_WIDTH = 3
_R2_STREAM_SLOPE = 0.8
_R2_STREAM_LOADING: np.ndarray = np.asarray([0.75, 0.45, 0.3], dtype=float)
_R2_STREAM_LOADING = _R2_STREAM_LOADING / float(np.linalg.norm(_R2_STREAM_LOADING))
_R2_ALPHA_GRID: np.ndarray = np.linspace(-8.0, 8.0, 4001)
_R2_HIGHER_IS_BETTER: dict[str, bool] = {"auc": True, "pauc": True, "brier": False, "abr": False}


def _r2_sigmoid(value: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _r2_latent_intercept(bad_rate: float, slope: float) -> float:
    """The intercept for which ``E[sigmoid(intercept + slope * z)] == bad_rate``, z ~ N(0, 1).

    The expectation is taken over a fixed standard-normal quadrature grid by bisection,
    so the population bad rate is calibrated without ever looking at a sampled outcome.
    """
    weights = np.exp(-0.5 * _R2_ALPHA_GRID**2)
    weights = weights / float(weights.sum())
    lower, upper = -40.0, 40.0
    for _ in range(140):
        middle = 0.5 * (lower + upper)
        rate = float(np.sum(_r2_sigmoid(middle + slope * _R2_ALPHA_GRID) * weights))
        if rate < float(bad_rate):
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _r2_population(rng: Any, n: int, intercept: float, slope: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One i.i.d. applicant sample: features, feature-only risk driver, latent bad outcome."""
    features = rng.normal(0.0, 1.0, (int(n), _R2_STREAM_WIDTH))
    latent = features @ _R2_STREAM_LOADING
    labels = (rng.random(int(n)) < _r2_sigmoid(intercept + slope * latent)).astype(int)
    return features, latent, labels


def generate_reject_inference_stream(config: Mapping[str, Any]) -> dict[str, Any]:
    """C14: a selected accepted/rejected split plus an unselected representative holdout.

    The applicant pool is drawn from a known logistic DGP whose intercept is calibrated
    to ``bad_rate``; the ``acceptance_rate`` share with the *lowest feature-only risk
    score* is approved (approval never reads the latent outcome), so rejects are riskier
    than accepts. The holdout is a fresh, unselected sample from the same DGP carrying
    latent labels only. Keyed by ``seed``, ``n_applicants``, ``n_holdout``,
    ``acceptance_rate`` and ``bad_rate``; returns exactly the contract keys.
    """
    seed = int(config.get("seed", 1))
    n_applicants = int(config.get("n_applicants", 1200))
    n_holdout = int(config.get("n_holdout", 1000))
    acceptance_rate = min(max(float(config.get("acceptance_rate", 0.3)), 1e-9), 1.0)
    bad_rate = min(max(float(config.get("bad_rate", 0.4)), 1e-9), 1.0 - 1e-9)
    intercept = _r2_latent_intercept(bad_rate, _R2_STREAM_SLOPE)
    rng = np.random.default_rng([seed, 11])
    pool_features, pool_latent, pool_labels = _r2_population(rng, n_applicants, intercept, _R2_STREAM_SLOPE)
    holdout_features, _, holdout_labels = _r2_population(rng, n_holdout, intercept, _R2_STREAM_SLOPE)
    n_accept = int(round(acceptance_rate * n_applicants))
    n_accept = min(max(n_accept, 1), max(n_applicants - 1, 1))
    order = np.argsort(pool_latent, kind="stable")
    accepted, rejected = order[:n_accept], order[n_accept:]
    return {
        "X_accept": pool_features[accepted].tolist(),
        "y_accept": pool_labels[accepted].tolist(),
        "X_reject": pool_features[rejected].tolist(),
        "y_reject_latent": pool_labels[rejected].tolist(),
        "X_holdout": holdout_features.tolist(),
        "y_holdout_latent": holdout_labels.tolist(),
        "seed": seed,
    }


def _r2_study_stream_seeds(seed: int, n_trials: int) -> list[int]:
    """One distinct stream seed per trial, derived only from the study seed."""
    rng = np.random.default_rng([int(seed), 20260923])
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=max(int(n_trials), 0))]


def _r2_paired_bootstrap(differences: np.ndarray, level: float, n_boot: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean paired difference, resampling trials."""
    values = np.asarray(differences, dtype=float).reshape(-1)
    n = int(values.size)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng([int(seed), 57])
    draws = rng.integers(0, n, size=(max(int(n_boot), 1), n))
    means = values[draws].mean(axis=1)
    tail = 0.5 * (1.0 - float(level))
    low = float(np.percentile(means, 100.0 * tail))
    high = float(np.percentile(means, 100.0 * (1.0 - tail)))
    centre = float(values.mean())
    return min(low, centre), max(high, centre)


def _r2_algorithm1_step(metric: str, labels_accept: Any, scores_accept: Any, scores_reject: Any) -> Any:
    """A per-draw metric evaluator with the fixed score ranking hoisted out of the loop.

    Same arithmetic as :func:`_r2_metric_value`/``bayesian_metric`` but the scores (and
    therefore the ranks, the ROC boundaries and the ABR grid positions) are precomputed
    once; only the pseudo-labels change from draw to draw.
    """
    y_a = np.asarray(labels_accept, dtype=float).reshape(-1)
    s_a = np.asarray(scores_accept, dtype=float).reshape(-1)
    s_r = np.asarray(scores_reject, dtype=float).reshape(-1)
    n_a = int(y_a.size)
    n = n_a + int(s_r.size)
    if metric == "auc":
        ranks = rankdata(np.concatenate([s_a, s_r]), method="average")
        accept_positive_rank = float(np.sum(ranks[:n_a][y_a == 1.0]))
        n_pos_accept = float(np.sum(y_a == 1.0))
        n_neg_accept = float(np.sum(y_a == 0.0))
        ranks_reject = ranks[n_a:]

        def step(pseudo: np.ndarray) -> float:
            n_pos_reject = float(np.sum(pseudo == 1.0))
            n_pos = n_pos_accept + n_pos_reject
            n_neg = n_neg_accept + (ranks_reject.size - n_pos_reject)
            if n_pos <= 0.0 or n_neg <= 0.0:
                return 0.5
            positive_rank_sum = accept_positive_rank + float(np.sum(ranks_reject[pseudo == 1.0]))
            return (positive_rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

        return step
    if metric == "brier":
        accept_squared = float(np.sum((s_a - y_a) ** 2))

        def step(pseudo: np.ndarray) -> float:
            return (accept_squared + float(np.sum((s_r - pseudo) ** 2))) / float(n)

        return step
    if metric == "pauc":
        order = np.argsort(-np.concatenate([s_a, s_r]), kind="mergesort")
        sorted_scores = np.concatenate([s_a, s_r])[order]
        boundaries = np.concatenate([np.flatnonzero(np.diff(sorted_scores) != 0.0), [n - 1]])

        def step(pseudo: np.ndarray) -> float:
            labels_sorted = np.concatenate([y_a, pseudo])[order]
            n_pos = float(np.sum(labels_sorted == 1.0))
            n_neg = float(n - n_pos)
            if n_pos <= 0.0 or n_neg <= 0.0:
                return 0.5
            cumulative_positive = np.cumsum(labels_sorted == 1.0, dtype=float)
            cumulative_negative = np.cumsum(labels_sorted == 0.0, dtype=float)
            fpr = np.concatenate([[0.0], cumulative_negative[boundaries] / n_neg])
            tpr = np.concatenate([[0.0], cumulative_positive[boundaries] / n_pos])
            right = int(np.searchsorted(fpr, R2_PAUC_REGION, side="right"))
            area = 0.0
            if right >= 2:
                area = float(np.sum(0.5 * (fpr[1:right] - fpr[: right - 1]) * (tpr[1:right] + tpr[: right - 1])))
            if right < fpr.size:
                x0, y0 = float(fpr[right - 1]), float(tpr[right - 1])
                x1, y1 = float(fpr[right]), float(tpr[right])
                tpr_at_max = y0 if x1 <= x0 else y0 + (y1 - y0) * (R2_PAUC_REGION - x0) / (x1 - x0)
                area += 0.5 * (R2_PAUC_REGION - x0) * (y0 + tpr_at_max)
            return area / R2_PAUC_REGION

        return step
    if metric == "abr":
        order = np.argsort(np.concatenate([s_a, s_r]), kind="mergesort")
        ks = np.asarray([int(min(max(round(float(rate) * n), 1), n)) for rate in R2_ABR_GRID])

        def step(pseudo: np.ndarray) -> float:
            labels_sorted = np.concatenate([y_a, pseudo])[order]
            cumulative = np.cumsum(labels_sorted == 1.0)
            return float(np.mean(cumulative[ks - 1] / ks))

        return step
    raise ValueError(f"unknown metric {metric!r}; expected one of {R2_METRICS}")


def _r2_bayesian_metric_fast(
    labels_accept: Any,
    scores_accept: Any,
    scores_reject: Any,
    prior_reject: Any,
    metric: str,
    seed: int = 1,
    tolerance: float = 1e-6,
    min_draws: int = 20,
    max_draws: int = 60,
) -> dict[str, Any]:
    """``bayesian_metric`` with its fixed ranking hoisted out (identical draw sequence)."""
    prior = np.asarray(prior_reject, dtype=float).reshape(-1)
    n_reject = int(prior.size)
    min_draws = max(1, int(min_draws))
    max_draws = max(min_draws, int(max_draws))
    tolerance = abs(float(tolerance))
    step = _r2_algorithm1_step(metric, labels_accept, scores_accept, scores_reject)
    rng = np.random.default_rng([int(seed), 0])
    draw_values: list[float] = []
    running_mean: list[float] = []
    converged = False
    for draw_index in range(1, max_draws + 1):
        pseudo = (rng.random(n_reject) < prior).astype(float)
        draw_values.append(float(step(pseudo)))
        running_mean.append(float(np.mean(draw_values)))
        if draw_index >= min_draws and abs(running_mean[-1] - running_mean[-2]) <= tolerance:
            converged = True
            break
    return {"value": float(running_mean[-1]), "n_draws": int(len(draw_values)), "converged": bool(converged)}


def run_reject_inference_study(config: Mapping[str, Any]) -> dict[str, Any]:
    """C14: paired across-trials study of accepts-only vs Algorithm 1, plus a prior-flip sweep.

    Each trial draws its own stream from ``seed``; its row is reproducible by
    ``run_reject_inference_evaluation(stream, {**config, seed, prior_flip_fraction: 0.0})``
    (the in-loop ranking-hoisted Algorithm 1 agrees with that reference to ~1e-16), and
    ``truth`` is that scorecard's metric on the latent-labelled representative holdout.
    The corruption sweep reuses the same streams and seeds, so flip fraction 0 reproduces
    the paired errors exactly. Directions are recorded from the bootstrap interval and
    never treated as a claim.
    """
    seed = int(config.get("seed", 1))
    n_boot = int(config.get("n_boot", 1000))
    level = float(config.get("level", 0.95))
    tolerance = float(config.get("tolerance", 1e-6))
    min_draws = int(config.get("min_draws", 20))
    max_draws = int(config.get("max_draws", 60))
    fractions = [float(value) for value in config.get("corruption_fractions", [0.0, 0.25, 0.5, 0.75, 1.0])]
    stream_seeds = _r2_study_stream_seeds(seed, int(config.get("n_trials", 50)))

    trials: list[dict[str, Any]] = []
    sweep_bayesian = {fraction: {metric: [] for metric in R2_METRICS} for fraction in fractions}
    accepts_only_errors = {metric: [] for metric in R2_METRICS}
    for trial_index, stream_seed in enumerate(stream_seeds):
        stream = generate_reject_inference_stream({**config, "seed": stream_seed})
        y_accept = np.asarray(stream["y_accept"], dtype=float).reshape(-1)
        model, constant = _r2_fit_scorecard(stream["X_accept"], y_accept)
        scores_accept = np.clip(_r2_scorecard_predict(model, constant, stream["X_accept"]), 0.0, 1.0)
        scores_reject = np.clip(_r2_scorecard_predict(model, constant, stream["X_reject"]), 0.0, 1.0)
        scores_holdout = np.clip(_r2_scorecard_predict(model, constant, stream["X_holdout"]), 0.0, 1.0)
        prior_clean = reject_prior(stream)
        accepts_only = compute_metrics(y_accept, scores_accept)
        truth = compute_metrics(stream["y_holdout_latent"], scores_holdout)
        row = {
            "trial": int(trial_index),
            "stream_seed": int(stream_seed),
            "truth": {metric: float(truth[metric]) for metric in R2_METRICS},
            "accepts_only_estimate": {metric: float(accepts_only[metric]) for metric in R2_METRICS},
            "bayesian_estimate": {},
            "accepts_only_abs_error": {
                metric: float(abs(accepts_only[metric] - truth[metric])) for metric in R2_METRICS
            },
            "bayesian_abs_error": {},
        }
        clean = {
            metric: float(
                _r2_bayesian_metric_fast(
                    y_accept,
                    scores_accept,
                    scores_reject,
                    prior_clean,
                    metric,
                    seed=stream_seed,
                    tolerance=tolerance,
                    min_draws=min_draws,
                    max_draws=max_draws,
                )["value"]
            )
            for metric in R2_METRICS
        }
        row["bayesian_estimate"] = dict(clean)
        row["bayesian_abs_error"] = {metric: float(abs(clean[metric] - truth[metric])) for metric in R2_METRICS}
        for fraction in fractions:
            if fraction == 0.0:
                bayesian = clean
            else:
                prior = _r2_corrupt_prior(prior_clean, fraction, stream_seed)
                bayesian = {
                    metric: float(
                        _r2_bayesian_metric_fast(
                            y_accept,
                            scores_accept,
                            scores_reject,
                            prior,
                            metric,
                            seed=stream_seed,
                            tolerance=tolerance,
                            min_draws=min_draws,
                            max_draws=max_draws,
                        )["value"]
                    )
                    for metric in R2_METRICS
                }
            for metric in R2_METRICS:
                sweep_bayesian[fraction][metric].append(float(abs(bayesian[metric] - truth[metric])))
        trials.append(row)
        for metric in R2_METRICS:
            accepts_only_errors[metric].append(row["accepts_only_abs_error"][metric])

    sweep: list[dict[str, Any]] = []
    for fraction in fractions:
        sweep.append(
            {
                "flip_fraction": float(fraction),
                "mean_bayesian_abs_error": {
                    metric: float(np.mean(sweep_bayesian[fraction][metric])) for metric in R2_METRICS
                },
                "mean_accepts_only_abs_error": {
                    metric: float(np.mean(accepts_only_errors[metric])) for metric in R2_METRICS
                },
            }
        )

    paired: dict[str, dict[str, Any]] = {}
    for metric in R2_METRICS:
        accepts_only = np.asarray([row["accepts_only_abs_error"][metric] for row in trials], dtype=float)
        bayesian = np.asarray([row["bayesian_abs_error"][metric] for row in trials], dtype=float)
        difference = accepts_only - bayesian
        low, high = _r2_paired_bootstrap(difference, level, n_boot, seed)
        mean_difference = float(difference.mean()) if difference.size else 0.0
        paired[metric] = {
            "mean_accepts_only_abs_error": float(accepts_only.mean()) if accepts_only.size else 0.0,
            "mean_bayesian_abs_error": float(bayesian.mean()) if bayesian.size else 0.0,
            "mean_paired_difference": mean_difference,
            "ci_low": float(low),
            "ci_high": float(high),
            "level": float(level),
            "method": "paired_percentile_bootstrap",
            "direction": (
                "bayesian_closer" if low > 0.0 else "accepts_only_closer" if high < 0.0 else "indistinguishable"
            ),
        }

    crossover: dict[str, float | None] = {}
    for metric in R2_METRICS:
        found: float | None = None
        for sweep_row in sweep:
            if sweep_row["mean_bayesian_abs_error"][metric] >= sweep_row["mean_accepts_only_abs_error"][metric]:
                found = float(sweep_row["flip_fraction"])
                break
        crossover[metric] = found

    return {
        "n_trials": int(len(trials)),
        "metrics": list(R2_METRICS),
        "trials": trials,
        "paired": paired,
        "prior_corruption_sweep": sweep,
        "crossover": crossover,
        "claim_status": "derived_here",
    }


# ======================================================================================
# Round-2 (C15): BASL self-learning with a feature-only weak signal and a Bayesian
# early stop. Learning and stopping never read the latent reject/holdout outcomes; only
# the final comparison does.
# ======================================================================================


def _r2_is_better(metric: str, new_value: float, old_value: float) -> bool:
    return bool(new_value > old_value) if _R2_HIGHER_IS_BETTER[metric] else bool(new_value < old_value)


def run_basl(stream: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """C15: Kozodoi's BASL on ``stream`` with a feature-only weak signal and a Bayesian stop.

    Each iteration filters the reject pool to the middle of the weak learner's goodness
    band (``beta``), samples ``rho`` of it, labels the lowest-goodness ``gamma * theta``
    share bad and the highest-goodness ``gamma`` share good, and retrains the strong
    scorecard on accepts plus those pseudo-labelled rejects. The early-stop metric is the
    Bayesian metric of the strong model, with the accepts-trained prior; the loop stops
    at ``j_max``, on the first non-improving iteration, or when the filtered/sampled pool
    is empty. The selected iteration is the best one on the Bayesian metric; nothing in
    learning or stopping reads ``y_reject_latent`` or ``y_holdout_latent``.
    """
    seed = int(config.get("seed", 1))
    tolerance = float(config.get("tolerance", 1e-6))
    min_draws = int(config.get("min_draws", 20))
    max_draws = int(config.get("max_draws", 60))
    beta = [float(value) for value in config.get("beta", [0.05, 1.0])]
    rho = float(config.get("rho", 0.8))
    gamma = float(config.get("gamma", 0.01))
    theta = float(config.get("theta", 2.0))
    j_max = max(int(config.get("j_max", 3)), 0)
    early_stop_metric = str(config.get("early_stop_metric", "auc"))
    if early_stop_metric not in R2_METRICS:
        raise ValueError(f"unknown early_stop_metric {early_stop_metric!r}; expected one of {R2_METRICS}")

    y_accept = np.asarray(stream["y_accept"], dtype=float).reshape(-1)
    x_accept = np.asarray(stream["X_accept"], dtype=float)
    x_reject = np.asarray(stream["X_reject"], dtype=float)
    x_holdout = np.asarray(stream["X_holdout"], dtype=float)
    prior = np.asarray(reject_prior(stream), dtype=float)

    base_model, base_constant = _r2_fit_scorecard(x_accept, y_accept)
    accept_scores = np.clip(_r2_scorecard_predict(base_model, base_constant, x_accept), 0.0, 1.0)
    reject_scores = np.clip(_r2_scorecard_predict(base_model, base_constant, x_reject), 0.0, 1.0)
    holdout_scores_accepts_only = [
        float(value) for value in np.clip(_r2_scorecard_predict(base_model, base_constant, x_holdout), 0.0, 1.0)
    ]

    def bayesian_value(accept_scores_in: np.ndarray, reject_scores_in: np.ndarray) -> float:
        draw = bayesian_metric(
            labels_accept=y_accept,
            scores_accept=accept_scores_in,
            scores_reject=reject_scores_in,
            prior_reject=prior,
            metric=early_stop_metric,
            seed=seed,
            tolerance=tolerance,
            min_draws=min_draws,
            max_draws=max_draws,
        )
        return float(draw["value"])

    values = [bayesian_value(accept_scores, reject_scores)]
    holdout_scores = [holdout_scores_accepts_only]
    labeling: list[dict[str, int]] = []
    stop_reason = "j_max"
    goodness = 1.0 - reject_scores
    for iteration in range(1, j_max + 1):
        low = float(np.quantile(goodness, beta[0]))
        high = float(np.quantile(goodness, beta[1]))
        kept = np.flatnonzero((goodness >= low) & (goodness <= high))
        n_filtered = int(kept.size)
        n_sampled = min(n_filtered, int(round(rho * n_filtered)))
        if n_filtered == 0 or n_sampled == 0:
            stop_reason = "empty_reject_pool"
            break
        rng = np.random.default_rng([seed, 101 + iteration])
        picked = kept[rng.permutation(n_filtered)[:n_sampled]]
        n_good = int(round(gamma * n_sampled))
        n_bad = min(int(round(gamma * theta * n_sampled)), n_sampled - min(n_good, n_sampled))
        n_good = min(n_good, n_sampled - n_bad)
        ranked = picked[np.argsort(goodness[picked], kind="stable")]
        bad_index = ranked[:n_bad]
        good_index = ranked[n_sampled - n_good:]
        features = np.vstack([x_accept, x_reject[bad_index], x_reject[good_index]])
        labels = np.concatenate([y_accept, np.ones(n_bad), np.zeros(n_good)])
        model, constant = _r2_fit_scorecard(features, labels)
        strong_accept = np.clip(_r2_scorecard_predict(model, constant, x_accept), 0.0, 1.0)
        strong_reject = np.clip(_r2_scorecard_predict(model, constant, x_reject), 0.0, 1.0)
        strong_holdout = np.clip(_r2_scorecard_predict(model, constant, x_holdout), 0.0, 1.0)
        value = bayesian_value(strong_accept, strong_reject)
        improved = _r2_is_better(early_stop_metric, value, values[-1])
        labeling.append({"n_filtered": n_filtered, "n_sampled": n_sampled, "n_good": n_good, "n_bad": n_bad})
        values.append(value)
        holdout_scores.append([float(entry) for entry in strong_holdout])
        if iteration == j_max:
            stop_reason = "j_max"
            break
        if not improved:
            stop_reason = "no_improvement"
            break

    selected_iteration = values.index(
        max(values) if _R2_HIGHER_IS_BETTER[early_stop_metric] else min(values)
    )
    holdout_scores_basl = (
        list(holdout_scores_accepts_only)
        if selected_iteration == 0
        else list(holdout_scores[selected_iteration])
    )
    accepts_only_metrics = compute_metrics(stream["y_holdout_latent"], holdout_scores_accepts_only)
    basl_metrics = compute_metrics(stream["y_holdout_latent"], holdout_scores_basl)
    comparison = {
        metric: {
            "accepts_only": float(accepts_only_metrics[metric]),
            "basl": float(basl_metrics[metric]),
            "difference": float(basl_metrics[metric] - accepts_only_metrics[metric]),
        }
        for metric in R2_METRICS
    }

    return {
        "parameters": {
            "beta": beta,
            "rho": rho,
            "gamma": gamma,
            "theta": theta,
            "j_max": int(j_max),
            "early_stop_metric": early_stop_metric,
        },
        "iterations_run": int(len(values) - 1),
        "stop_reason": stop_reason,
        "early_stop_metric": early_stop_metric,
        "metric_by_iteration": [float(value) for value in values],
        "selected_iteration": int(selected_iteration),
        "labeling": labeling,
        "holdout_scores_accepts_only": holdout_scores_accepts_only,
        "holdout_scores_basl": holdout_scores_basl,
        "comparison": comparison,
    }
