"""recovery experiment: a pure-PU oracle and its measured practical variants.

Synthetic mechanism (one replicate, one propensity floor ``e_min``):

* ``x ~ N(0, 1)`` for each of the ``n_transactions`` transactions;
* risk score ``s = sigmoid(2x)``; latent outcome ``Y ~ Bernoulli(s)``;
* prediction ``pred = 1[s >= 0.5]``;
* propensity ``e = e_min + (1 - e_min) * u`` with ``u ~ U(0, 1)`` drawn
  independently of ``x`` and ``Y``;
* observed positive ``S = Y * Bernoulli(e)``.

The latent 0--1 risk of a replicate is ``mean(loss(Y, pred))``. The ``oracle``
variant scores ``S`` with the *true* propensities via the pure-PU SAR formula;
``estimated`` uses a constant, data-derived propensity (observed-positive
frequency in the replicate's top risk-score decile, clipped to (0.02, 0.98));
``misspecified`` uses the constant 0.5. Only the oracle inherits the pure-PU
guarantee; the practical variants are ``derived_here`` stress tests reported as
a measured bias and variance, with no asserted sign.

The oracle report is evaluated at the largest configured ``e_min`` (the best
identified floor). Its Monte-Carlo error interval is ``mean +/- 1.96 * sem`` of
the per-replicate ``SAR - latent`` gaps. ``uncertainty_by_e_min`` reports the
same oracle interval at every ``e_min``; a lower floor must widen it, and that
monotonicity is asserted here so a regression is loud.

``latent_risk_stratum`` is per transaction in the pre-panel aggregation
(``low`` iff ``s < 0.5``). A per-replicate row carries the tag of the
replicate's mean-score side -- a documented convention, because one replicate's
population spans both strata.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from label_delay.config import config_hash, config_key, run_identity, seed_of

VARIANTS: tuple[str, ...] = ("oracle", "estimated", "misspecified")
STRATA: tuple[str, ...] = ("low", "high")
LEVEL = 0.95
_Z95 = 1.959963984540054
ESTIMATED_PROPENSITY_CLIP: tuple[float, float] = (0.02, 0.98)
ESTIMATED_PROPENSITY_RULE = (
    "constant propensity = observed-positive frequency in the replicate's top "
    "risk-score decile, clipped to (0.02, 0.98)"
)
MISSPECIFIED_PROPENSITY_RULE = "constant propensity fixed at 0.50 for every transaction"
ORACLE_INTERVAL_METHOD = (
    "mean +/- 1.96 * sem of per-replicate SAR-minus-latent truth gaps "
    "(normal approximation, Monte-Carlo error interval)"
)
CLAIM_STATUS_BY_VARIANT: dict[str, str] = {
    "oracle": "source_backed_pure_pu_oracle",
    "estimated": "derived_here",
    "misspecified": "derived_here",
}

_CACHE: dict[str, dict[str, Any]] = {}


def sar_zero_one_risk(
    observed_positive: Sequence[float],
    propensity: Sequence[float],
    prediction: Sequence[float],
) -> float:
    """Pure-PU SAR 0--1 empirical risk (latent ``Y`` is never an argument).

    ``risk = mean_i ((S_i / e_i) * loss(1, p_i) + (1 - S_i / e_i) * loss(0, p_i))``
    with ``loss(y, p) = 1 if y != p else 0``. Plain-Python arithmetic so the
    exact reference ``sar_zero_one_risk([1,0,1,0],[0.5,0.5,1.0,0.25],[0,0,0,0])``
    returns exactly ``0.75``.
    """
    observed = [float(value) for value in observed_positive]
    weights = [float(value) for value in propensity]
    guesses = [float(value) for value in prediction]
    if not observed or len(observed) != len(weights) or len(observed) != len(guesses):
        raise ValueError(
            "sar_zero_one_risk needs equally sized, non-empty observed_positive, "
            "propensity and prediction sequences"
        )
    total = 0.0
    for s_i, e_i, p_i in zip(observed, weights, guesses):
        if e_i <= 0.0:
            raise ValueError("propensity must be strictly positive")
        if p_i == 1.0:
            total += 1.0 - s_i / e_i
        elif p_i == 0.0:
            total += s_i / e_i
        else:
            raise ValueError("prediction must be 0 or 1 (0--1 loss)")
    return total / float(len(observed))


def _sar_risk_array(
    observed_positive: np.ndarray,
    propensity: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Vectorised internal twin of :func:`sar_zero_one_risk`."""
    ratio = observed_positive / propensity
    term = np.where(prediction >= 0.5, 1.0 - ratio, ratio)
    return float(term.mean())


def run_recovery_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic recovery experiment for ``config``."""
    key = config_key(config)
    if key not in _CACHE:
        _CACHE[key] = _run(config)
    return copy.deepcopy(_CACHE[key])


def _run(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = seed_of(config)
    n_transactions = int(config["n_transactions"])
    n_replicates = int(config["n_replicates"])
    e_min_values = [float(value) for value in config["e_min_values"]]
    if not e_min_values:
        raise ValueError("config['e_min_values'] must contain at least one value")
    if n_replicates < 2:
        raise ValueError("config['n_replicates'] must be at least 2 for a sem interval")

    sar_by: dict[tuple[float, str], list[float]] = {}
    gap_by: dict[tuple[float, str], list[float]] = {}
    latent_by_e: dict[float, list[float]] = {}
    pre_sums: dict[tuple[float, str], tuple[float, int]] = {}
    raw_rows: list[dict[str, Any]] = []

    for e_index, e_min in enumerate(e_min_values):
        for variant in VARIANTS:
            sar_by[(e_min, variant)] = []
            gap_by[(e_min, variant)] = []
        latent_by_e[e_min] = []
        for replicate in range(n_replicates):
            rng = np.random.default_rng([int(seed), 101 + e_index, int(replicate)])
            x = rng.standard_normal(n_transactions)
            score = 1.0 / (1.0 + np.exp(-2.0 * x))
            latent = (rng.random(n_transactions) < score).astype(np.float64)
            prediction = (score >= 0.5).astype(np.float64)
            propensity = e_min + (1.0 - e_min) * rng.random(n_transactions)
            observed_positive = latent * (
                rng.random(n_transactions) < propensity
            ).astype(np.float64)

            latent_risk = float(np.mean(latent != prediction))
            latent_by_e[e_min].append(latent_risk)

            top_decile = score >= float(np.quantile(score, 0.9))
            estimated_propensity = float(
                np.clip(
                    float(observed_positive[top_decile].mean()),
                    ESTIMATED_PROPENSITY_CLIP[0],
                    ESTIMATED_PROPENSITY_CLIP[1],
                )
            )
            propensities = {
                "oracle": propensity,
                "estimated": np.full(n_transactions, estimated_propensity),
                "misspecified": np.full(n_transactions, 0.5),
            }
            stratum = "high" if float(score.mean()) >= 0.5 else "low"
            observed_positive_rate = float(observed_positive.mean())

            for stratum_name, mask in (("low", score < 0.5), ("high", score >= 0.5)):
                running_sum, running_count = pre_sums.get((e_min, stratum_name), (0.0, 0))
                pre_sums[(e_min, stratum_name)] = (
                    running_sum + float(observed_positive[mask].sum()),
                    running_count + int(mask.sum()),
                )

            for variant in VARIANTS:
                sar_risk = _sar_risk_array(
                    observed_positive, propensities[variant], prediction
                )
                gap = sar_risk - latent_risk
                sar_by[(e_min, variant)].append(sar_risk)
                gap_by[(e_min, variant)].append(gap)
                raw_rows.append(
                    {
                        "e_min": e_min,
                        "estimator_variant": variant,
                        "replicate": int(replicate + 1),
                        "latent_risk_stratum": stratum,
                        "observed_positive_rate": observed_positive_rate,
                        "sar_risk": sar_risk,
                        "latent_risk": latent_risk,
                        "sar_minus_truth_risk": gap,
                        "estimated_propensity": estimated_propensity,
                    }
                )

    interval_width_by: dict[tuple[float, str], float] = {}
    for e_min in e_min_values:
        for variant in VARIANTS:
            gaps = np.asarray(gap_by[(e_min, variant)], dtype=np.float64)
            sem = float(gaps.std(ddof=1) / math.sqrt(gaps.size))
            interval_width_by[(e_min, variant)] = float(2.0 * _Z95 * sem)
    for row in raw_rows:
        row["interval_width"] = interval_width_by[
            (row["e_min"], row["estimator_variant"])
        ]

    observed_rows: list[dict[str, Any]] = []
    for e_min in e_min_values:
        for stratum in STRATA:
            total, count = pre_sums[(e_min, stratum)]
            observed_rows.append(
                {
                    "e_min": e_min,
                    "latent_risk_stratum": stratum,
                    "observed_positive_rate": total / count if count else 0.0,
                    "n": int(count),
                }
            )

    uncertainty_rows: list[dict[str, Any]] = []
    for e_min in e_min_values:
        gaps = np.asarray(gap_by[(e_min, "oracle")], dtype=np.float64)
        mean_gap = float(gaps.mean())
        sem = float(gaps.std(ddof=1) / math.sqrt(gaps.size))
        uncertainty_rows.append(
            {
                "e_min": e_min,
                "interval_width": float(2.0 * _Z95 * sem),
                "interval_low": float(mean_gap - _Z95 * sem),
                "interval_high": float(mean_gap + _Z95 * sem),
                "method": ORACLE_INTERVAL_METHOD,
                "n_replicates": int(n_replicates),
                "level": LEVEL,
                "mean_sar_minus_latent": mean_gap,
            }
        )

    ordered = sorted(uncertainty_rows, key=lambda row: row["e_min"])
    widths = [row["interval_width"] for row in ordered]
    assert all(a > b for a, b in zip(widths, widths[1:])), (
        " regression: smaller e_min must produce a strictly wider oracle "
        f"Monte-Carlo interval, got widths {widths} for e_min "
        f"{[row['e_min'] for row in ordered]}."
    )

    e_max = max(e_min_values)
    oracle_sar = np.asarray(sar_by[(e_max, "oracle")], dtype=np.float64)
    oracle_latent = np.asarray(latent_by_e[e_max], dtype=np.float64)
    oracle_gaps = np.asarray(gap_by[(e_max, "oracle")], dtype=np.float64)
    oracle_half_width = float(_Z95 * oracle_gaps.std(ddof=1) / math.sqrt(oracle_gaps.size))
    oracle_center = float(oracle_sar.mean() - oracle_latent.mean())
    oracle = {
        "e_min": e_max,
        "mean_sar_risk": float(oracle_sar.mean()),
        "mean_latent_zero_one_risk": float(oracle_latent.mean()),
        "monte_carlo_error_interval": {
            "low": float(oracle_center - oracle_half_width),
            "high": float(oracle_center + oracle_half_width),
            "method": ORACLE_INTERVAL_METHOD,
            "level": LEVEL,
            "n_replicates": int(n_replicates),
        },
        "n_replicates": int(n_replicates),
        "method": ORACLE_INTERVAL_METHOD,
    }

    pooled_latent = np.concatenate(
        [np.asarray(latent_by_e[e_min], dtype=np.float64) for e_min in e_min_values]
    )
    practical_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        if variant == "oracle":
            continue
        pooled_sar = np.concatenate(
            [np.asarray(sar_by[(e_min, variant)], dtype=np.float64) for e_min in e_min_values]
        )
        pooled_gaps = np.concatenate(
            [np.asarray(gap_by[(e_min, variant)], dtype=np.float64) for e_min in e_min_values]
        )
        practical_rows.append(
            {
                "variant": variant,
                "bias": float(pooled_gaps.mean()),
                "variance": float(pooled_sar.var(ddof=1)),
                "claim_status": "derived_here",
                "propensity_rule": (
                    ESTIMATED_PROPENSITY_RULE
                    if variant == "estimated"
                    else MISSPECIFIED_PROPENSITY_RULE
                ),
                "n_replicates": int(n_replicates),
                "n_transactions": int(n_transactions),
                "e_min_values": [float(value) for value in e_min_values],
                "mean_sar_risk": float(pooled_sar.mean()),
                "mean_latent_zero_one_risk": float(pooled_latent.mean()),
            }
        )

    post_rows: list[dict[str, Any]] = []
    for e_min in e_min_values:
        for variant in VARIANTS:
            gaps = np.asarray(gap_by[(e_min, variant)], dtype=np.float64)
            post_rows.append(
                {
                    "e_min": e_min,
                    "estimator_variant": variant,
                    "absolute_bias": float(abs(float(gaps.mean()))),
                    "interval_width": interval_width_by[(e_min, variant)],
                    "claim_status": CLAIM_STATUS_BY_VARIANT[variant],
                }
            )

    return {
        "oracle": oracle,
        "uncertainty_by_e_min": uncertainty_rows,
        "practical_propensities": practical_rows,
        "replicates": raw_rows,
        "observed_positive_by_stratum": observed_rows,
        "post_summary": post_rows,
        "claim_status_by_variant": dict(CLAIM_STATUS_BY_VARIANT),
        "seed": int(seed),
        "config_hash": config_hash(config),
        "n_transactions": int(n_transactions),
        "n_replicates": int(n_replicates),
        "e_min_values": [float(value) for value in e_min_values],
        "oracle_interval_method": ORACLE_INTERVAL_METHOD,
    }


# ===========================================================================
# the excess-risk study against the Coudray Eq. 15 form, with
# kappa_1 left free. The study is a genuine SAR-ERM over a class whose VC
# dimension and Massart margin are stated in advance, and every random number
# is drawn here from the config's seed.
#
# Population (design constants, no data-dependent tuning):
#   x = (x1, x2) ~ N(0, I_2); the class G is the set of thresholds
#   g_t(x) = 1[x1 > t] on the univariate score x1 (VC dimension V = 1);
#   the conditional label probability is
#       eta(x) = 1/2 + 1/2 * (h + (1 - h) |u(x1)|) * sign(x1),
#       u(x1) = x1 / (1 + |x1|),
#   so |2 eta(x) - 1| = h + (1 - h)|u| >= h for every x: the Massart margin
#   h = 0.2 holds exactly on the population (A2 of the paper) and the Bayes
#   classifier is the threshold at 0, which is inside G (so g* is both the
#   Bayes and the in-class risk minimiser and the identity
#   R(g) - R(g*) = E[|2 eta - 1| 1{g != g*}] holds);
#   the known, instance-dependent propensity is e(x) = e_m + (1 - e_m)
#   sigmoid(x2) >= e_m with an observed floor inside [e_m, e_m + 0.1].
# The estimator under test is the exact empirical SAR risk minimiser over G
# (Eq. 14 of the paper), obtained by sweeping the n+1 candidate thresholds of
# the sorted score. No numeric kappa_1 is used anywhere: the returned
# ``bound_kappa1_1`` is Eq. 15/16 evaluated at kappa_1 = 1 (a reference scale
# only) and ``kappa_hat`` is measured from the cell ratios.
# ===========================================================================

C9_VC_DIM = 1
C9_MARGIN_H = 0.2
C9_POPULATION_SAMPLE = 200_000

C9_CLAIM_STATUS = "derived_here"
C9_CLASS_RULE = (
    "G = thresholds g_t(x) = 1[x1 > t] on the univariate score x1 "
    "(VC dimension V = 1); the SAR-ERM of Eq. 14 is evaluated exactly over "
    "all n+1 sorted candidate thresholds"
)
C9_PROPENSITY_RULE = (
    "known instance-dependent e(x) = e_m + (1 - e_m) * sigmoid(x2), "
    "independent of the label and of the score direction"
)
C9_EXCESS_RULE = (
    "R(g_hat) - R(g*) = E[|2 eta(x) - 1| 1{g_hat != g*}] estimated on an "
    "independent 200,000-point sample of the same population"
)
C9_BOUND_RULE = (
    "Eq. 15/16 at kappa_1 = 1: min(V/(n e_m h) * (1 + log(max(n h^2 / V, 1))), "
    "sqrt(V/(n e_m))); the returned bound is that reference scale, not a fitted kappa_1"
)
C9_KAPPA_RULE = (
    "kappa_hat = max over cells of mean_excess_risk / bound_kappa1_1: the "
    "smallest kappa_1 >= 1 consistent with the measured mean excess risks on "
    "this grid (a measured constant, never hard-coded)"
)
C9_TREND_RULE = (
    "joint OLS of the per-cell mean ratios on [1, log n, 1/e_m]; interval = "
    "bootstrap normal interval, slope +/- z * bootstrap SD, with replicates "
    "resampled within each cell and the OLS refit per bootstrap draw"
)

_C9_CACHE: dict[str, dict[str, Any]] = {}


def _c9_margin_term(x1: np.ndarray) -> np.ndarray:
    """|2 eta(x) - 1| for the study population: >= C9_MARGIN_H, = C9_MARGIN_H at x1 = 0."""
    return C9_MARGIN_H + (1.0 - C9_MARGIN_H) * np.abs(x1 / (1.0 + np.abs(x1)))


def _c9_propensity(x2: np.ndarray, e_m: float) -> np.ndarray:
    """Known instance-dependent propensity in [e_m, 1) with min observed near e_m."""
    return e_m + (1.0 - e_m) / (1.0 + np.exp(-x2))


def _eq15_bound_kappa1_one(n: int, e_m: float, v: float, h: float) -> float:
    """Eq. 15/16 of Coudray et al. at kappa_1 = 1 (the reference scale).

    Implements the Eq. 15 expression directly.
    """
    first = v / (n * e_m * h) * (1.0 + math.log(max(n * h * h / v, 1.0)))
    second = math.sqrt(v / (n * e_m))
    return float(min(first, second))


def _c9_sar_erm_threshold(
    score: np.ndarray, propensity: np.ndarray, observed_positive: np.ndarray
) -> float:
    """Exact SAR-ERM over the threshold class G on the sorted score.

    For a split after the j-th sorted score the empirical SAR risk is
    ``(1/n) [ W + (n - j) - 2 suffix_j ]`` (W = sum of S/e, a constant for the
    argmin), so the sweep is exact and O(n log n).
    """
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    weights = observed_positive[order] / propensity[order]
    suffix = np.cumsum(weights[::-1])[::-1]
    split_index = np.arange(sorted_score.size + 1, dtype=float)
    objective = (sorted_score.size - split_index) - 2.0 * np.concatenate([suffix, [0.0]])
    split = int(np.argmin(objective))
    if split == 0:
        return float(sorted_score[0] - 1.0)
    if split == sorted_score.size:
        return float(sorted_score[-1] + 1.0)
    return float(0.5 * (sorted_score[split - 1] + sorted_score[split]))


def _c9_excess_mass(
    sorted_score: np.ndarray, prefix_mass: np.ndarray, sample_size: int, threshold: float
) -> float:
    """E[|2 eta - 1| 1{g_hat != g*}] on the pre-sorted population sample."""
    low = min(0.0, threshold)
    high = max(0.0, threshold)
    left = int(np.searchsorted(sorted_score, low, side="right"))
    right = int(np.searchsorted(sorted_score, high, side="right"))
    return float((prefix_mass[right] - prefix_mass[left]) / sample_size)


def fit_excess_ratio_trend(
    cells: Sequence[Mapping[str, Any]],
    level: float = 0.95,
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Joint OLS of per-cell mean ratios on ``[1, log n, 1/e_m]`` + bootstrap interval."""
    ordered = list(cells)
    if not ordered:
        raise ValueError("fit_excess_ratio_trend needs at least one cell")
    design = np.asarray(
        [[1.0, math.log(float(cell["n"])), 1.0 / float(cell["e_m"])] for cell in ordered],
        dtype=float,
    )
    ratios = np.asarray(
        [
            float(np.mean(np.asarray(cell["excess_risks"], dtype=float)))
            / float(cell["bound_kappa1_1"])
            for cell in ordered
        ],
        dtype=float,
    )
    coefficients = np.linalg.lstsq(design, ratios, rcond=None)[0]
    draws = np.empty((max(int(n_boot), 0), 3), dtype=float)
    rng = np.random.default_rng(int(seed))
    for draw in range(draws.shape[0]):
        sampled = np.empty(len(ordered), dtype=float)
        for index, cell in enumerate(ordered):
            values = np.asarray(cell["excess_risks"], dtype=float)
            if values.size == 0:
                raise ValueError("every cell needs at least one replicate excess risk")
            resample = values[rng.integers(0, values.size, values.size)]
            sampled[index] = float(resample.mean()) / float(cell["bound_kappa1_1"])
        draws[draw] = np.linalg.lstsq(design, sampled, rcond=None)[0]
    if draws.shape[0] >= 2:
        spread = draws.std(axis=0, ddof=1)
    else:
        spread = np.zeros(3, dtype=float)
    # A resampling spread of exactly zero means the bootstrap cannot see any
    # uncertainty (e.g. one replicate per cell); keep the interval numerically
    # non-degenerate rather than reporting low == high.
    floor = 1e-12 * np.maximum(1.0, np.abs(coefficients))
    spread = np.maximum(spread, floor)
    z_value = float(stats.norm.ppf(0.5 + 0.5 * float(level)))
    trend: dict[str, Any] = {}
    for index, key in ((1, "log_n"), (2, "inv_e_m")):
        slope = float(coefficients[index])
        half_width = z_value * float(spread[index])
        trend[key] = {"slope": slope, "low": slope - half_width, "high": slope + half_width}
    trend["level"] = float(level)
    trend["n_boot"] = int(n_boot)
    trend["method"] = "bootstrap"
    trend["interval_rule"] = C9_TREND_RULE
    trend["design"] = "[1, log n, 1/e_m]"
    trend["cell_count"] = len(ordered)
    return trend


def run_excess_risk_study(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic excess-risk study for ``config``."""
    key = config_key(config)
    if key not in _C9_CACHE:
        _C9_CACHE[key] = _c9_run(config)
    return copy.deepcopy(_C9_CACHE[key])


def _c9_run(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = seed_of(config)
    n_values = [int(value) for value in config["n_values"]]
    e_m_values = [float(value) for value in config["e_m_values"]]
    n_replicates = int(config["n_replicates"])
    level = float(config.get("level", 0.95))
    n_boot = int(config.get("n_boot", 500))
    if not n_values or not e_m_values:
        raise ValueError("config['n_values'] and config['e_m_values'] must be non-empty")
    if n_replicates < 2:
        raise ValueError("config['n_replicates'] must be at least 2 for a replicate SE")

    vc_dim = int(C9_VC_DIM)
    margin_h = float(C9_MARGIN_H)
    population_rng = np.random.default_rng([int(seed), 777])
    score_truth = population_rng.standard_normal(C9_POPULATION_SAMPLE)
    nuisance_truth = population_rng.standard_normal(C9_POPULATION_SAMPLE)
    margin_truth = _c9_margin_term(score_truth)
    margin_min_observed = float(margin_truth.min())

    order = np.argsort(score_truth, kind="stable")
    sorted_score = score_truth[order]
    prefix_mass = np.concatenate([[0.0], np.cumsum(margin_truth[order])])

    cells: list[dict[str, Any]] = []
    for e_index, e_m in enumerate(e_m_values):
        propensity_truth = _c9_propensity(nuisance_truth, e_m)
        e_min_observed = float(propensity_truth.min())
        e_max_observed = float(propensity_truth.max())
        for n in n_values:
            excess_risks: list[float] = []
            for replicate in range(n_replicates):
                rng = np.random.default_rng([int(seed), int(n), int(e_index), int(replicate)])
                score = rng.standard_normal(n)
                nuisance = rng.standard_normal(n)
                margin = _c9_margin_term(score)
                eta = 0.5 + 0.5 * margin * np.sign(score)
                latent = (rng.random(n) < eta).astype(np.float64)
                propensity = _c9_propensity(nuisance, e_m)
                observed_positive = latent * (rng.random(n) < propensity).astype(np.float64)
                threshold = _c9_sar_erm_threshold(score, propensity, observed_positive)
                excess_risks.append(
                    _c9_excess_mass(sorted_score, prefix_mass, C9_POPULATION_SAMPLE, threshold)
                )
            values = np.asarray(excess_risks, dtype=float)
            bound = _eq15_bound_kappa1_one(int(n), float(e_m), float(vc_dim), margin_h)
            mean_excess = float(values.mean())
            cells.append(
                {
                    "n": int(n),
                    "e_m": float(e_m),
                    "excess_risks": [float(value) for value in values],
                    "mean_excess_risk": mean_excess,
                    "se_excess_risk": float(values.std(ddof=1) / math.sqrt(values.size)),
                    "bound_kappa1_1": bound,
                    "ratio": mean_excess / bound,
                    "e_min_observed": e_min_observed,
                    "e_max_observed": e_max_observed,
                    "n_replicates": int(n_replicates),
                }
            )

    ratios = [float(cell["ratio"]) for cell in cells]
    kappa_hat = float(max(ratios))
    trend = fit_excess_ratio_trend(cells, level=level, n_boot=n_boot, seed=int(seed))
    return {
        "vc_dim": int(vc_dim),
        "margin_h": margin_h,
        "margin_min_observed": margin_min_observed,
        "cells": cells,
        "kappa_hat": kappa_hat,
        "kappa_hat_rule": C9_KAPPA_RULE,
        "trend": trend,
        "seed": int(seed),
        "config_hash": config_hash(config),
        "n_replicates": int(n_replicates),
        "n_values": [int(value) for value in n_values],
        "e_m_values": [float(value) for value in e_m_values],
        "level": level,
        "n_boot": int(n_boot),
        "claim_status": C9_CLAIM_STATUS,
        "class_rule": C9_CLASS_RULE,
        "propensity_rule": C9_PROPENSITY_RULE,
        "excess_risk_rule": C9_EXCESS_RULE,
        "bound_rule": C9_BOUND_RULE,
        "population_sample": int(C9_POPULATION_SAMPLE),
    }


# ===========================================================================
# Three genuinely different
# estimators run on ONE shared realistic stream.
#
#   sar       a pure-PU SAR risk on the MATURED labels: the PU positive is a
#             matured confirmed fraud (S = Y * label_observed, so
#             P(S=1 | Y=0, x) = 0), the propensity is the design-known
#             e(x) = P(label observed | Y=1, x) = approval_probability(score) *
#             fraud-mix maturation probability(age), and bias/variance are
#             reported per propensity stratum (never pooled into one number).
#   bayesian  the accept + pseudo-label metric of Kozodoi Algorithm 1: true
#             labels on the observed (accept) records, prior-matched pseudo
#             labels drawn for every unlabelled record, AUC/Brier averaged over
#             posterior draws.
#   wctm      a monitoring statistic: windowed matured-label scores turned into
#             online conformal p-values and a WCTM test-martingale wealth path
#             (Prinster Eq. 6) with alarm threshold c.
#
# All three read the same stream and stamp the same data hash; the config
# identity is computed on the config with the ledger/cache infrastructure keys
# removed.
# ===========================================================================

STREAM_ESTIMATORS: tuple[str, ...] = ("sar", "bayesian", "wctm")
STREAM_CLAIM_STATUS = "derived_here"
STREAM_INFRA_KEYS: tuple[str, ...] = ("ledger_path", "cache_dir")
STREAM_CLASSIFIER_QUANTILE_FALLBACK = 0.85
DEFAULT_BAYESIAN_DRAWS = 32
DEFAULT_MONITOR_WINDOWS = 8
DEFAULT_WCTM_THRESHOLD_C = 20.0

STREAM_PROPENSITY_RULE = (
    "known-by-design one-sided propensity e(x) = P(label observed | Y=1, x) = "
    "approval_probability(model_score) * fraud-mix maturation probability(age), "
    "using the generator's published approval constants and maturation curves"
)
STREAM_CLASSIFIER_RULE = (
    "g(x) = 1 iff the deployed score is at or above the operational approval cut "
    "(the score where the published approval probability crosses 0.5)"
)
STREAM_SAR_RULE = (
    "mean over rows of S/e (2*1{g != 1} - 1) + 1{g != 0} (Coudray Eq. 12) with "
    "S = Y * label_observed; bias = bootstrap-mean SAR - stratum truth risk, "
    "variance = bootstrap variance of the SAR estimate"
)
STREAM_BAYESIAN_RULE = (
    "Kozodoi Algorithm 1 style: true labels on the label-observed records, "
    "Bernoulli(prior) pseudo labels on every unlabelled record, prior = the "
    "observed bad rate of the labelled records, AUC (half credit for ties) and "
    "Brier averaged over posterior draws"
)
STREAM_WCTM_RULE = (
    "windows of matured-label mean model score -> online conformal p-values "
    "(p_t = (K_t + 0.5)/t) -> WCTM wealth path (equal mixture over bets "
    "eps in {-1, 0, 1} of prod (1 + eps (p - 0.5))), alarm when wealth >= c"
)

_STREAM_STUDY_CACHE: dict[str, dict[str, Any]] = {}


def effective_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the ledger/cache infrastructure keys before hashing or generating."""
    return {key: value for key, value in dict(config).items() if key not in STREAM_INFRA_KEYS}


def _stream_arrays(stream: Mapping[str, Any]) -> dict[str, np.ndarray]:
    transactions = stream["transactions"]
    return {
        "score": np.asarray([float(t["model_score"]) for t in transactions], dtype=float),
        "arrival": np.asarray([int(t["arrival_index"]) for t in transactions], dtype=float),
        "latent": np.asarray([int(t["Y"]) for t in transactions], dtype=float),
        "observed": np.asarray([bool(t["label_observed"]) for t in transactions], dtype=float),
        "approved": np.asarray([bool(t["approved"]) for t in transactions], dtype=float),
        "typology": np.asarray([str(t["typology"]) for t in transactions], dtype=object),
    }


def _maturity_probability(typology: str, age: float) -> float:
    """Exact sampler CDF of the typology maturation curve at an integer age."""
    from label_delay.data import STUDY_MATURATION_CURVES

    points = STUDY_MATURATION_CURVES[typology]
    ages = np.asarray([point[0] for point in points], dtype=float)
    fractions = np.asarray([point[1] for point in points], dtype=float)
    if age < ages[0]:
        return 0.0
    if age >= ages[-1]:
        scale = max(1.0, float(ages[-1]) * 0.35)
        tail = 1.0 - math.exp(-max(0.0, (age - ages[-1]) - 1.0) / scale)
        return float(fractions[-1] + (1.0 - fractions[-1]) * tail)
    index = int(np.searchsorted(ages, age, side="right")) - 1
    return float(fractions[index])


def _stream_propensity(stream: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Design-known one-sided propensity e(x) = P(label observed | Y=1, x)."""
    from label_delay.data import STUDY_APPROVAL_ALPHA, STUDY_APPROVAL_BETA, STUDY_FRAUD_TYPOLOGIES

    n = int(stream["n"])
    score = arrays["score"]
    log_odds = np.log(score / (1.0 - score))
    standardised = (log_odds - log_odds.mean()) / (log_odds.std() + 1e-12)
    approval = 1.0 / (1.0 + np.exp(-(STUDY_APPROVAL_ALPHA - STUDY_APPROVAL_BETA * standardised)))
    fraud_weights = {"first_party_fraud": 0.30, "synthetic_identity": 0.30, "account_takeover": 0.40}
    total_weight = float(sum(fraud_weights[typ] for typ in STUDY_FRAUD_TYPOLOGIES))
    maturity = np.empty(n, dtype=float)
    for index in range(n):
        age = n - 1 - int(arrays["arrival"][index])
        maturity[index] = (
            sum(
                fraud_weights[typ] * _maturity_probability(typ, age)
                for typ in STUDY_FRAUD_TYPOLOGIES
            )
            / total_weight
        )
    return approval * maturity


def _stream_classifier(stream: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """The deployed score's operational cut: p_approve == 0.5 under the published rule."""
    from label_delay.data import STUDY_APPROVAL_ALPHA, STUDY_APPROVAL_BETA

    score = arrays["score"]
    log_odds = np.log(score / (1.0 - score))
    mean, std = log_odds.mean(), log_odds.std() + 1e-12
    cut_standardised = STUDY_APPROVAL_ALPHA / STUDY_APPROVAL_BETA
    cut = 1.0 / (1.0 + math.exp(-(cut_standardised * std + mean)))
    if not np.isfinite(cut) or cut <= 0.0 or cut >= 1.0:
        cut = float(np.quantile(score, STREAM_CLASSIFIER_QUANTILE_FALLBACK))
    return (score >= cut).astype(float)


def _auc_half_credit(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUC with average ranks, so tied scores get half credit."""
    positive = labels > 0.5
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = stats.rankdata(scores, method="average")
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def _bootstrap_risk_means(values: np.ndarray, replicates: int, seed: list[int]) -> np.ndarray:
    """Bootstrap means of a per-record risk vector (the SAR estimator's resampling)."""
    size = int(values.size)
    rng = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=float)
    for draw in range(int(replicates)):
        draws[draw] = float(values[rng.integers(0, size, size)].mean())
    return draws


def _estimate_sar(
    config: Mapping[str, Any], stream: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    n = int(stream["n"])
    seed = seed_of(config)
    strata_count = int(config.get("n_propensity_strata", 4))
    replicates = int(config.get("n_replicates", 60))
    propensity = _stream_propensity(stream, arrays)
    classifier = _stream_classifier(stream, arrays)
    latent, observed = arrays["latent"], arrays["observed"]
    positive = latent * observed  # PU positive: a matured, confirmed fraud
    risk_terms = positive / propensity * (2.0 * (classifier != 1.0) - 1.0) + (classifier != 0.0)
    truth_terms = (classifier != latent).astype(float)

    edges = np.quantile(propensity, np.linspace(0.0, 1.0, strata_count + 1))
    strata_index = np.clip(np.searchsorted(edges, propensity, side="right") - 1, 0, strata_count - 1)
    strata: list[dict[str, Any]] = []
    for stratum in range(strata_count):
        mask = strata_index == stratum
        count = int(mask.sum())
        if count < 1:
            raise ValueError("a propensity stratum is empty; lower n_propensity_strata")
        truth_risk = float(truth_terms[mask].mean())
        draws = _bootstrap_risk_means(risk_terms[mask], replicates, [int(seed), 555, stratum])
        strata.append(
            {
                "stratum": stratum,
                "propensity_low": float(propensity[mask].min()),
                "propensity_high": float(propensity[mask].max()),
                "n": count,
                "bias": float(draws.mean()) - truth_risk,
                "variance": float(draws.var(ddof=1)) if draws.size > 1 else 0.0,
                "truth_risk": truth_risk,
                "sar_risk": float(draws.mean()),
                "n_confirmed_fraud": int(positive[mask].sum()),
                "flagged_share": float(classifier[mask].mean()),
            }
        )
    pooled_truth = float(truth_terms.mean())
    pooled_draws = _bootstrap_risk_means(risk_terms, replicates, [int(seed), 555, 99])
    pooled = {
        "stratum": "pooled",
        "n": n,
        "bias": float(pooled_draws.mean()) - pooled_truth,
        "variance": float(pooled_draws.var(ddof=1)) if pooled_draws.size > 1 else 0.0,
        "truth_risk": pooled_truth,
        "sar_risk": float(pooled_draws.mean()),
        "n_confirmed_fraud": int(positive.sum()),
        "flagged_share": float(classifier.mean()),
    }
    return {
        "estimator": "sar",
        "claim_status": STREAM_CLAIM_STATUS,
        "population": "all stream transactions; the PU positive is a matured confirmed fraud",
        "n_population": n,
        "n_strata": strata_count,
        "n_replicates": replicates,
        "n_confirmed_fraud": int(positive.sum()),
        "n_observed_labels": int(observed.sum()),
        "label_cost": int(observed.sum()),
        "label_cost_rule": "observed labels consumed (both outcomes; the PU positive is the matured confirmed fraud)",
        "propensity_rule": STREAM_PROPENSITY_RULE,
        "classifier_rule": STREAM_CLASSIFIER_RULE,
        "sar_rule": STREAM_SAR_RULE,
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "propensity_mean": float(propensity.mean()),
        "sar_risk": float(risk_terms.mean()),
        "truth_risk": pooled_truth,
        "strata": strata,
        "pooled": pooled,
    }


def _estimate_bayesian(
    config: Mapping[str, Any], stream: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    seed = seed_of(config)
    draws_count = int(config.get("bayesian_draws", DEFAULT_BAYESIAN_DRAWS))
    score, latent, observed = arrays["score"], arrays["latent"], arrays["observed"]
    labelled = observed > 0.5
    unlabelled = ~labelled
    n_labelled = int(labelled.sum())
    n_unlabelled = int(unlabelled.sum())
    if n_labelled == 0 or n_unlabelled == 0:
        raise ValueError("the Bayesian estimator needs both labelled and unlabelled records")
    labelled_labels = latent[labelled]
    if len(np.unique(labelled_labels)) < 2:
        raise ValueError("the Bayesian estimator needs both outcomes among the labelled records")
    prior = float(labelled_labels.mean())
    labelled_scores, labelled_labels = score[labelled], latent[labelled]
    unlabelled_scores = score[unlabelled]
    rng = np.random.default_rng([int(seed), 909])
    aucs = np.empty(draws_count, dtype=float)
    briers = np.empty(draws_count, dtype=float)
    for draw in range(draws_count):
        pseudo = (rng.random(n_unlabelled) < prior).astype(float)
        completed_labels = np.concatenate([labelled_labels, pseudo])
        completed_scores = np.concatenate([labelled_scores, unlabelled_scores])
        aucs[draw] = _auc_half_credit(completed_labels, completed_scores)
        briers[draw] = float(np.mean((completed_scores - completed_labels) ** 2))
    return {
        "estimator": "bayesian",
        "claim_status": STREAM_CLAIM_STATUS,
        "n_labelled": n_labelled,
        "n_unlabelled": n_unlabelled,
        "n_draws": draws_count,
        "prior_rule": "observed bad rate of the labelled records (no unlabelled outcome is read)",
        "prior": prior,
        "auc_mean": float(aucs.mean()),
        "auc_sd": float(aucs.std(ddof=1)) if draws_count > 1 else 0.0,
        "brier_mean": float(briers.mean()),
        "brier_sd": float(briers.std(ddof=1)) if draws_count > 1 else 0.0,
        "metric_rule": STREAM_BAYESIAN_RULE,
        "label_cost": n_labelled,
        "label_cost_rule": "true (observed) labels consumed; pseudo labels are not observed labels",
    }


def _wctm_wealth_path(p_values: Sequence[float]) -> list[float]:
    """Eq. 6/7 WCTM wealth: equal mixture over eps in {-1, 0, 1}, started at 1."""
    wealth = np.ones(len(p_values), dtype=float)
    running = np.ones(3, dtype=float)
    for index, value in enumerate(p_values):
        centred = float(value) - 0.5
        running = running * (1.0 + np.asarray([-1.0, 0.0, 1.0]) * centred)
        wealth[index] = float(running.mean())
    return [float(value) for value in wealth]


def _estimate_wctm(
    config: Mapping[str, Any], stream: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    n = int(stream["n"])
    windows_count = int(config.get("n_windows", DEFAULT_MONITOR_WINDOWS))
    threshold_c = float(config.get("wctm_threshold_c", DEFAULT_WCTM_THRESHOLD_C))
    score, observed = arrays["score"], arrays["observed"]
    windows = np.array_split(np.arange(n), windows_count)
    window_rows: list[dict[str, Any]] = []
    values = np.empty(windows_count, dtype=float)
    for index, window in enumerate(windows):
        matured = observed[window] > 0.5
        matured_count = int(matured.sum())
        values[index] = float(score[window][matured].mean()) if matured_count else 0.5
        window_rows.append(
            {
                "window": index,
                "start": int(window[0]),
                "end": int(window[-1]) + 1,
                "n": int(window.size),
                "n_matured_labels": matured_count,
                "mean_matured_score": float(values[index]),
            }
        )
    p_values: list[float] = []
    for index, value in enumerate(values):
        if index == 0:
            p_values.append(0.5)
        else:
            greater = int(np.sum(values[:index] > value))
            p_values.append(float(min(1.0, max(0.0, (greater + 0.5) / index))))
    wealth = _wctm_wealth_path(p_values)
    peak = float(max(wealth)) if wealth else 1.0
    alarm_window = next((index for index, value in enumerate(wealth) if value >= threshold_c), None)
    half = windows_count // 2
    return {
        "estimator": "wctm",
        "claim_status": STREAM_CLAIM_STATUS,
        "statistic": "windowed mean model score of matured-label records",
        "n_windows": windows_count,
        "windows": window_rows,
        "p_values": p_values,
        "wealth_path": wealth,
        "max_wealth": peak,
        "final_wealth": float(wealth[-1]) if wealth else 1.0,
        "threshold_c": threshold_c,
        "alarm": bool(alarm_window is not None),
        "alarm_window": alarm_window,
        "pre_post_mean_difference": float(values[half:].mean() - values[:half].mean()) if windows_count > 1 else 0.0,
        "wctm_rule": STREAM_WCTM_RULE,
        "delay_note": (
            "labels mature with a typology-specific delay, so a record affected "
            "late in the stream may not be visible to a matured-label monitor "
            "before the stream ends; the wealth path is reported as measured, "
            "with no validity claim"
        ),
        "validity_claim": False,
        "label_cost": int(observed.sum()),
        "label_cost_rule": "matured (observed) labels consumed by the monitoring windows",
    }


_ESTIMATORS = {
    "sar": _estimate_sar,
    "bayesian": _estimate_bayesian,
    "wctm": _estimate_wctm,
}


def estimator_sidecar(
    name: str, config: Mapping[str, Any], stream: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Compute one estimator's sidecar on the shared stream, with the run identity."""
    if name not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; expected one of {STREAM_ESTIMATORS}")
    effective = effective_config(config)
    if stream is None:
        from label_delay.data import generate_realistic_stream

        stream = generate_realistic_stream(effective)
    from label_delay.data import audit_stream

    audit = audit_stream(stream)
    identity = run_identity(effective)
    payload = _ESTIMATORS[name](effective, stream, _stream_arrays(stream))
    sidecar = {
        **payload,
        "seed": seed_of(effective),
        "config_hash": identity["config_hash"],
        "data_hash": audit["data_hash"],
        "run_id": identity["run_id"],
    }
    if int(sidecar.get("label_cost", 0)) <= 0:
        raise ValueError(f"estimator {name!r} consumed no observed labels")
    return sidecar


def run_shared_stream_study(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic shared-stream study for ``config``."""
    key = config_key(config)
    if key not in _STREAM_STUDY_CACHE:
        _STREAM_STUDY_CACHE[key] = _shared_stream_study(config)
    return copy.deepcopy(_STREAM_STUDY_CACHE[key])


def _shared_stream_study(config: Mapping[str, Any]) -> dict[str, Any]:
    from label_delay.data import audit_stream, generate_realistic_stream

    effective = effective_config(config)
    stream = generate_realistic_stream(effective)
    audit = audit_stream(stream)
    identity = run_identity(effective)
    sidecars = {name: estimator_sidecar(name, effective, stream) for name in STREAM_ESTIMATORS}
    sar = sidecars["sar"]
    return {
        "data_hash": audit["data_hash"],
        "sidecars": sidecars,
        "sar_by_propensity_stratum": [dict(row) for row in sar["strata"]],
        "sar_pooled": dict(sar["pooled"]),
        "n_population": int(sar["n_population"]),
        "n_propensity_strata": int(sar["n_strata"]),
        "n_replicates": int(sar["n_replicates"]),
        "n_observed_labels": int(audit["n_observed_labels"]),
        "n_confirmed_fraud": int(sar["n_confirmed_fraud"]),
        "seed": seed_of(effective),
        "config_hash": identity["config_hash"],
        "run_id": identity["run_id"],
        "estimators": list(STREAM_ESTIMATORS),
        "propensity_rule": STREAM_PROPENSITY_RULE,
        "classifier_rule": STREAM_CLASSIFIER_RULE,
        "sar_rule": STREAM_SAR_RULE,
        "claim_status": STREAM_CLAIM_STATUS,
    }
