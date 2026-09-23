"""C5--C6 monitoring-validity experiment: ready-label null and measured breaks.

The ready-label null is a pre-registered betting-martingale WATCH engine. Each
stream is a sequence of fair signs and the wealth is a mixture over a fixed
lambda grid, which is a non-negative martingale with expectation one at every
update, so Ville's inequality bounds the true alarm probability by ``1/c``. A
single seeded run therefore cannot be judged by an exact rate; it is judged by
binomial compatibility (Wilson interval) with ``1/c``.

The delayed (stale) and selective (one-sided reveal) variants run the same
engine on labels that violate the ready-label assumptions. They are reported as
observations only: ``validity_claim`` is ``False``, with no required direction.

Round 2 (C11--C13) adds the weighted-conformal test martingale (WCTM) of
Prinster et al. (corrected version, as ingested): the Eq. 6 betting form
``h_eps(p) = 1 + eps (p - 0.5)`` with ``eps in {-1, 0, 1}``, the Eq. 9 weighted
conformal p-value with a uniform tie-breaker ``u ~ Unif[0, 1]``, and the
ONE-SIDED guarantee ``P(ever alarm) <= 1/c``. A valid conservative monitor may
alarm rarely, including never; validity is therefore decided by an exact
one-sided Clopper-Pearson criterion, never by a two-sided interval. The
out-of-guarantee variants (selective observation, drift during the label lag)
are measured observations with ``validity_claim = False`` and no direction
required.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
from scipy import stats

from exec.config import config_hash, config_key, seed_of

WATCH_LAMBDAS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
NULL_UPDATES = 160
ALTERNATE_UPDATES = 200
CHANGE_UPDATE = 80
POST_CHANGE_POSITIVE_PROBABILITY = 0.65
DELAY_UPDATES = 8
SELECTIVE_POSITIVE_REVEAL = 0.70
SELECTIVE_NEGATIVE_REVEAL = 0.50
WEALTH_FLOOR = 1e-12
WEALTH_STREAMS_SHOWN = 20
LABEL_EVENTS_PER_VARIANT = 10
INTERVAL_LEVEL = 0.95
_Z95 = 1.959963984540054

DETECTION_DELAY_RULE = (
    "mean over streams of max(0, first_alarm_update - 80); streams that never "
    "alarm contribute N - 80"
)

_CACHE: dict[str, dict[str, Any]] = {}


def run_monitoring_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic C5--C6 monitoring experiment for ``config``."""
    key = config_key(config)
    if key not in _CACHE:
        _CACHE[key] = _run(config)
    return copy.deepcopy(_CACHE[key])


# --------------------------------------------------------------------------- #
# pre-registered ready-label null engine


def _null_signs(seed: int, n_streams: int, n_updates: int) -> np.ndarray:
    """Fair +/-1 signs drawn once for all streams in a fixed order."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, 2, size=(n_streams, n_updates)) * 2.0 - 1.0


def _wealth(signs: np.ndarray) -> np.ndarray:
    """Mixture wealth path W_n = mean_lambda prod_{k<=n}(1 + lambda*X_k)."""
    wealth = np.zeros(signs.shape, dtype=float)
    for lam in WATCH_LAMBDAS:
        wealth += np.cumprod(1.0 + lam * signs, axis=1)
    return wealth / float(len(WATCH_LAMBDAS))


def _lag_signs(signs: np.ndarray, delay: int) -> np.ndarray:
    """Stale-label detector: update n uses the sign observed at n - delay."""
    used = np.zeros_like(signs)
    if delay > 0:
        used[:, delay:] = signs[:, :-delay]
    else:
        used[:] = signs
    return used


def _selective_signs(
    signs: np.ndarray,
    seed: int,
    tag: int,
    p_positive: float = SELECTIVE_POSITIVE_REVEAL,
    p_negative: float = SELECTIVE_NEGATIVE_REVEAL,
) -> np.ndarray:
    """One-sided reveal: positive evidence is revealed more often than negative."""
    rng = np.random.default_rng([int(seed), int(tag)])
    reveal_probability = np.where(signs > 0, p_positive, p_negative)
    revealed = rng.random(signs.shape) < reveal_probability
    return signs * revealed


def _change_point_signs(seed: int, n_streams: int, n_updates: int) -> np.ndarray:
    """Fair signs before CHANGE_UPDATE, P(+1)=0.65 after it."""
    rng = np.random.default_rng([int(seed), 2])
    pre = (rng.random((n_streams, n_updates)) < 0.5) * 2.0 - 1.0
    post = (
        rng.random((n_streams, n_updates)) < POST_CHANGE_POSITIVE_PROBABILITY
    ) * 2.0 - 1.0
    return np.where(np.arange(n_updates)[None, :] >= CHANGE_UPDATE, post, pre)


def _wilson_interval(count: int, n: int, z: float = _Z95) -> dict[str, Any]:
    """Named Wilson score interval; contains the point estimate for every count."""
    if n <= 0:
        return {"method": "wilson", "low": 0.0, "high": 1.0, "level": INTERVAL_LEVEL}
    rate = count / n
    denominator = 1.0 + z * z / n
    center = (rate + z * z / (2.0 * n)) / denominator
    half_width = (z / denominator) * math.sqrt(
        rate * (1.0 - rate) / n + z * z / (4.0 * n * n)
    )
    return {
        "method": "wilson",
        "low": float(max(0.0, center - half_width)),
        "high": float(min(1.0, center + half_width)),
        "level": INTERVAL_LEVEL,
    }


def _interval_contains(interval: Mapping[str, Any], target: float) -> bool:
    return interval["low"] <= target <= interval["high"]


def _alarm_count(wealth: np.ndarray, threshold_c: float) -> int:
    return int((wealth.max(axis=1) >= threshold_c).sum())


def _detection_delay(
    wealth: np.ndarray, threshold_c: float, n_updates: int
) -> tuple[float, int]:
    peak = wealth.max(axis=1)
    alarmed = peak >= threshold_c
    delays = np.full(wealth.shape[0], float(n_updates - CHANGE_UPDATE))
    for stream in range(wealth.shape[0]):
        if not alarmed[stream]:
            continue
        first_alarm = int(np.nonzero(wealth[stream] >= threshold_c)[0][0]) + 1
        delays[stream] = float(max(0, first_alarm - CHANGE_UPDATE))
    return float(delays.mean()), int(alarmed.sum())


def _wealth_rows(variant: str, wealth: np.ndarray, threshold_c: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shown = min(WEALTH_STREAMS_SHOWN, wealth.shape[0])
    for stream in range(shown):
        logs = np.log(np.maximum(wealth[stream], WEALTH_FLOOR))
        for n in range(wealth.shape[1]):
            rows.append(
                {
                    "variant": variant,
                    "stream": int(stream),
                    "update_index": int(n + 1),
                    "log_wealth": float(logs[n]),
                    "threshold_c": float(threshold_c),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# experiment


def _run(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = seed_of(config)
    n_streams = int(config["watch_streams"])
    threshold_c = float(config["watch_threshold"])
    target = 1.0 / threshold_c

    # Each pre-registered horizon is its own seeded run: numpy's bounded
    # integer draws are not prefix-consistent, so the 200-update fallback must
    # be drawn as a fresh run of the same design rather than as an extension.
    counts: dict[str, int] = {}
    wealth_by_horizon: dict[int, np.ndarray] = {}
    signs_by_horizon: dict[int, np.ndarray] = {}
    intervals: dict[int, dict[str, Any]] = {}
    for horizon in (NULL_UPDATES, ALTERNATE_UPDATES):
        signs = _null_signs(seed, n_streams, horizon)
        wealth = _wealth(signs)
        wealth_by_horizon[horizon] = wealth
        signs_by_horizon[horizon] = signs
        counts[str(horizon)] = _alarm_count(wealth, threshold_c)
        intervals[horizon] = _wilson_interval(counts[str(horizon)], n_streams)

    if _interval_contains(intervals[NULL_UPDATES], target):
        selected_horizon = NULL_UPDATES
        horizon_rule = (
            "160-update Wilson interval contains 1/c; the pre-registered default "
            "horizon is used"
        )
    elif _interval_contains(intervals[ALTERNATE_UPDATES], target):
        selected_horizon = ALTERNATE_UPDATES
        horizon_rule = (
            "160-update Wilson interval excluded 1/c, so the pre-registered "
            "alternative horizon of 200 updates is used"
        )
    else:
        # Should not occur for the frozen seed; report the closer interval.
        def _distance(horizon: int) -> float:
            interval = intervals[horizon]
            return max(interval["low"] - target, target - interval["high"], 0.0)

        selected_horizon = min(
            (NULL_UPDATES, ALTERNATE_UPDATES), key=_distance
        )
        horizon_rule = (
            "neither pre-registered horizon was interval-compatible with 1/c; "
            "the closer interval is reported as an observation"
        )

    selected_wealth = wealth_by_horizon[selected_horizon]
    selected_signs = signs_by_horizon[selected_horizon]
    selected_interval = intervals[selected_horizon]
    selected_count = counts[str(selected_horizon)]
    ready_rate = selected_count / n_streams if n_streams else 0.0

    ready = {
        "variant": "ready",
        "threshold_c": threshold_c,
        "target_false_alarm_rate": float(target),
        "n_streams": int(n_streams),
        "updates_per_stream": int(selected_horizon),
        "false_alarm_count": int(selected_count),
        "empirical_false_alarm_rate": float(ready_rate),
        "false_alarm_binomial_interval": selected_interval,
        "validity_claim": "source_backed_ready_label_reference",
        "updates_per_stream_documentation": {
            "pre_registered_horizons": [NULL_UPDATES, ALTERNATE_UPDATES],
            "false_alarm_count_by_horizon": {
                str(NULL_UPDATES): int(counts[str(NULL_UPDATES)]),
                str(ALTERNATE_UPDATES): int(counts[str(ALTERNATE_UPDATES)]),
            },
            "wilson_interval_by_horizon": {
                str(NULL_UPDATES): intervals[NULL_UPDATES],
                str(ALTERNATE_UPDATES): intervals[ALTERNATE_UPDATES],
            },
            "selected_horizon": int(selected_horizon),
            "horizon_selection_rule": horizon_rule,
        },
        "wealth_paths": _wealth_rows("ready", selected_wealth, threshold_c),
    }

    delayed_null_signs = _lag_signs(selected_signs, DELAY_UPDATES)
    selective_null_signs = _selective_signs(selected_signs, seed, 11)
    delayed_null_wealth = _wealth(delayed_null_signs)
    selective_null_wealth = _wealth(selective_null_signs)
    delayed_null_count = _alarm_count(delayed_null_wealth, threshold_c)
    selective_null_count = _alarm_count(selective_null_wealth, threshold_c)

    change_point_signs = _change_point_signs(seed, n_streams, NULL_UPDATES)
    change_point_variants = {
        "ready": change_point_signs,
        "delayed": _lag_signs(change_point_signs, DELAY_UPDATES),
        "selective": _selective_signs(change_point_signs, seed, 12),
    }
    detection_rows: list[dict[str, Any]] = []
    delays_by_variant: dict[str, float] = {}
    for variant in ("ready", "delayed", "selective"):
        wealth = _wealth(change_point_variants[variant])
        delay, alarm_count = _detection_delay(wealth, threshold_c, NULL_UPDATES)
        delays_by_variant[variant] = delay
        detection_rows.append(
            {
                "variant": variant,
                "detection_delay": delay,
                "alarm_count": int(alarm_count),
                "n_streams": int(n_streams),
                "updates_per_stream": NULL_UPDATES,
                "change_point_update": CHANGE_UPDATE,
                "post_change_positive_probability": POST_CHANGE_POSITIVE_PROBABILITY,
                "detection_delay_rule": DETECTION_DELAY_RULE,
            }
        )

    delayed_interval = _wilson_interval(delayed_null_count, n_streams)
    selective_interval = _wilson_interval(selective_null_count, n_streams)
    delayed_rate = delayed_null_count / n_streams if n_streams else 0.0
    selective_rate = selective_null_count / n_streams if n_streams else 0.0
    variants = [
        {
            "variant": "delayed",
            "threshold_c": threshold_c,
            "n_streams": int(n_streams),
            "updates_per_stream": int(selected_horizon),
            "false_alarm_count": int(delayed_null_count),
            "empirical_false_alarm_rate": float(delayed_rate),
            "false_alarm_binomial_interval": delayed_interval,
            "validity_claim": False,
            "detection_delay": float(delays_by_variant["delayed"]),
            "detection_delay_rule": DETECTION_DELAY_RULE,
            "alarm_count": int(
                next(r["alarm_count"] for r in detection_rows if r["variant"] == "delayed")
            ),
            "note": (
                f"break: labels are stale by {DELAY_UPDATES} updates (the detector "
                f"at update n uses the sign from n - {DELAY_UPDATES}), so the "
                "ready-label martingale assumptions fail; measured rate "
                f"{delayed_rate:.3f} vs ready target {target:.3f}"
            ),
        },
        {
            "variant": "selective",
            "threshold_c": threshold_c,
            "n_streams": int(n_streams),
            "updates_per_stream": int(selected_horizon),
            "false_alarm_count": int(selective_null_count),
            "empirical_false_alarm_rate": float(selective_rate),
            "false_alarm_binomial_interval": selective_interval,
            "validity_claim": False,
            "detection_delay": float(delays_by_variant["selective"]),
            "detection_delay_rule": DETECTION_DELAY_RULE,
            "alarm_count": int(
                next(r["alarm_count"] for r in detection_rows if r["variant"] == "selective")
            ),
            "note": (
                "break: labels are revealed preferentially on the positive (fraud) "
                f"evidence side (p={SELECTIVE_POSITIVE_REVEAL:.2f} vs "
                f"{SELECTIVE_NEGATIVE_REVEAL:.2f}), which breaks the fair-sign null "
                f"expectation and inflates the measured rate to {selective_rate:.3f} "
                f"against the ready target {target:.3f}"
            ),
        },
    ]

    label_status_sequence = _label_status_sequence(seed, selected_signs)

    wealth_paths = (
        ready["wealth_paths"]
        + _wealth_rows("selective", selective_null_wealth, threshold_c)
        + _wealth_rows("delayed", delayed_null_wealth, threshold_c)
    )

    return {
        "ready_label_null": ready,
        "out_of_guarantee_variants": variants,
        "detection_delay_by_variant": detection_rows,
        "label_status_sequence": label_status_sequence,
        "wealth_paths": wealth_paths,
        "seed": int(seed),
        "config_hash": config_hash(config),
    }


def _label_status_sequence(
    seed: int, signs: np.ndarray
) -> list[dict[str, Any]]:
    """Update-level label arrivals: on-time, selectively revealed, or stale."""
    events: list[dict[str, Any]] = []
    n_events = LABEL_EVENTS_PER_VARIANT
    for event in range(1, n_events + 1):
        events.append(
            {
                "variant": "ready",
                "event_index": event,
                "update_index": event,
                "label_status": "ready",
                "label_delay_updates": 0,
            }
        )
    reveal_rng = np.random.default_rng([int(seed), 13])
    stream_signs = signs[0, :n_events]
    reveal_probability = np.where(
        stream_signs > 0, SELECTIVE_POSITIVE_REVEAL, SELECTIVE_NEGATIVE_REVEAL
    )
    revealed = reveal_rng.random(n_events) < reveal_probability
    for event in range(1, n_events + 1):
        events.append(
            {
                "variant": "selective",
                "event_index": event,
                "update_index": event,
                "label_status": "selected" if revealed[event - 1] else "censored",
                "label_delay_updates": 0,
            }
        )
    for event in range(1, n_events + 1):
        pending = event <= DELAY_UPDATES
        events.append(
            {
                "variant": "delayed",
                "event_index": event,
                "update_index": event,
                "label_status": "censored" if pending else "ready",
                "label_delay_updates": DELAY_UPDATES,
            }
        )
    return events


# --------------------------------------------------------------------------- #
# Round 2 (C11--C13): weighted-conformal test martingale (WCTM), Prinster et al.
#
# Grounded design (research/verification-r2.md, Prinster row; corrected version,
# as ingested): Eq. 6 betting form h_eps(p) = 1 + eps (p - 0.5) with
# eps in {-1, 0, 1}, mixed as a Composite Jumper; Eq. 9 weighted conformal
# p-value with an independent u ~ Unif[0, 1] tie-breaker; ONE-SIDED guarantee
# P(ever alarm) <= 1/c (an upper bound), so a valid conservative monitor may
# alarm rarely or never. The homogeneous null with uniform weights makes the
# online p-values exactly iid Uniform(0, 1) under iid continuous scores.

WCTM_BET_EPSILONS: tuple[float, ...] = (-1.0, 0.0, 1.0)
WCTM_BET_WEIGHTS: tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
WCTM_BOUND_LEVEL = 0.99
WCTM_DELAY_LEVEL = 0.95
WCTM_OOC_HORIZON = 120
WCTM_OOC_APPROVAL_QUANTILE = 0.70
WCTM_OOC_DRIFT_QUANTILE = 0.25
WCTM_OOC_DRIFT_SIZE = 1.5
WCTM_ALARM_RULE = "first step t with wealth M_t >= c (M_0 = 1)"
WCTM_SOURCE = (
    "Prinster et al. 2505.04608 v4 (corrected version, as ingested): Eq. 6 "
    "betting form h_eps(p) = 1 + eps (p - 0.5), eps in {-1, 0, 1}; Eq. 9 "
    "weighted conformal p-value with u ~ Unif[0, 1]; one-sided guarantee "
    "P(ever alarm) <= 1/c"
)

_WCTM_CACHE: dict[str, dict[str, Any]] = {}


def weighted_conformal_pvalue(
    scores: Any, weights: Any = None, u: float = 0.5
) -> float:
    """Eq. 9 weighted conformal p-value with the uniform tie-breaker.

    ``scores`` are ``v_1..v_n`` followed by the TEST score ``v_{n+1}`` last, so
    ``p = sum_i w_i [1{v_i > v_test} + u 1{v_i = v_test}]`` over all ``n+1``
    points (the test contributes its own ``u w_{n+1}`` tie mass). ``weights``
    ``None`` means uniform ``1/(n+1)``; provided weights are normalised to unit
    mass. ``u`` is the tie-breaker, drawn ``Unif[0, 1]`` by callers that need
    exact uniformity under exchangeability.
    """
    values = [float(v) for v in scores]
    if not values:
        raise ValueError("weighted_conformal_pvalue needs at least one score")
    if weights is None:
        mass = [1.0 / len(values)] * len(values)
    else:
        mass = [max(float(w), 0.0) for w in weights]
        if len(mass) != len(values):
            raise ValueError("weights must have one entry per score")
        total = float(sum(mass))
        if total <= 0.0:
            raise ValueError("weights must carry positive mass")
        mass = [w / total for w in mass]
    test = values[-1]
    u_value = float(u)
    p_value = 0.0
    for value, weight in zip(values, mass):
        if value > test:
            p_value += weight
        elif value == test:
            p_value += u_value * weight
    return float(min(1.0, max(0.0, p_value)))


def wctm_wealth_path(p_values: Any) -> list[float]:
    """Eq. 6/7 test-martingale wealth path for one stream of p-values.

    The wealth is the equal mixture over the bets ``eps in {-1, 0, 1}`` of
    ``prod_{k<=t} (1 + eps (p_k - 0.5))``: non-negative, started at ``M_0 = 1``,
    and a martingale with ``E[M_t] = 1`` for iid Uniform(0, 1) p-values because
    every bet is centred at ``p = 0.5`` (``E[h_eps(p)] = 1``).
    """
    values = np.asarray([float(p) for p in p_values], dtype=float)
    if values.size == 0:
        return []
    return [float(m) for m in _wctm_wealth_matrix(values)[0]]


def run_wctm_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic C11--C13 WCTM experiment for ``config``.

    The ready-label null is the exact one-sided validity test at
    ``WCTM_BOUND_LEVEL`` against ``1/c``; the shift experiment reports detection
    delay and miss rate per configured label lag; the out-of-guarantee variants
    report measured alarm rates with exact Clopper-Pearson bounds and no
    validity claim.
    """
    key = config_key(config)
    if key not in _WCTM_CACHE:
        _WCTM_CACHE[key] = _run_wctm(config)
    return copy.deepcopy(_WCTM_CACHE[key])


def _run_wctm(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = seed_of(config)
    threshold_c = float(config.get("wctm_threshold_c", 20.0))
    return {
        "null": _wctm_null_block(seed, config, threshold_c),
        "shift": _wctm_shift_block(seed, config, threshold_c),
        "out_of_guarantee": _wctm_ooc_block(seed, config, threshold_c),
        "threshold_c": threshold_c,
        "one_sided_bound": 1.0 / threshold_c if threshold_c else float("inf"),
        "bound_level": WCTM_BOUND_LEVEL,
        "alarm_rule": WCTM_ALARM_RULE,
        "source": WCTM_SOURCE,
        "seed": int(seed),
        "config_hash": config_hash(config),
    }


def _online_conformal_pvalues(scores: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Online Eq. 9 p-values: ``p_t = (K_t + u_t) / t`` with uniform weights.

    ``K_t`` counts the earlier scores strictly greater than the test score
    ``v_t``. For iid continuous scores and independent ``u_t ~ Unif[0, 1]`` the
    rows are exactly iid Uniform(0, 1) (exchangeability of the online bag).
    """
    scores = np.asarray(scores, dtype=float)
    uniforms = np.asarray(uniforms, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
        uniforms = uniforms.reshape(1, -1)
    n_streams, horizon = scores.shape
    greater = np.zeros((n_streams, horizon), dtype=float)
    for t in range(1, horizon):
        greater[:, t] = (scores[:, :t] > scores[:, t : t + 1]).sum(axis=1)
    steps = np.arange(1, horizon + 1, dtype=float)[None, :]
    return (greater + uniforms) / steps


def _wctm_wealth_matrix(p_values: np.ndarray) -> np.ndarray:
    """Vectorised wealth paths for a matrix of p-values (one row per stream)."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    centered = p - 0.5
    epsilons = np.asarray(WCTM_BET_EPSILONS, dtype=float)[:, None, None]
    factors = 1.0 + epsilons * centered[None, :, :]
    paths = np.cumprod(factors, axis=2)
    weights = np.asarray(WCTM_BET_WEIGHTS, dtype=float)
    return np.tensordot(weights, paths, axes=(0, 0))


def _apply_label_lag(p_values: np.ndarray, lag: int) -> np.ndarray:
    """Matured-label view: step ``t`` sees the p-value of step ``t - lag``.

    The first ``lag`` steps carry no matured evidence, so their update is the
    neutral ``p = 0.5`` (every bet factor is exactly 1). A lag of an iid null is
    null-invariant: the wealth path is the ready-label path delayed by ``lag``.
    """
    if lag <= 0:
        return np.array(p_values, dtype=float, copy=True)
    out = np.full(p_values.shape, 0.5, dtype=float)
    if lag < p_values.shape[1]:
        out[:, lag:] = p_values[:, : p_values.shape[1] - lag]
    return out


def _wctm_null_block(
    seed: int, config: Mapping[str, Any], threshold_c: float
) -> dict[str, Any]:
    n_streams = int(config.get("wctm_streams", 1000))
    horizon = int(config.get("wctm_horizon", 100))
    scores = np.random.default_rng([int(seed), 101]).standard_normal(
        (n_streams, horizon)
    )
    uniforms = np.random.default_rng([int(seed), 102]).uniform(
        size=(n_streams, horizon)
    )
    pvalues = _online_conformal_pvalues(scores, uniforms)
    max_wealth = _wctm_wealth_matrix(pvalues).max(axis=1)
    alarms = [bool(m >= threshold_c) for m in max_wealth]
    alarm_count = int(sum(alarms))
    return {
        "n_streams": n_streams,
        "horizon": horizon,
        "threshold_c": float(threshold_c),
        "pvalues": pvalues.tolist(),
        "max_wealth": [float(m) for m in max_wealth],
        "alarms": alarms,
        "alarm_count": alarm_count,
        "alarm_rate": alarm_count / n_streams if n_streams else 0.0,
        "alarm_rate_bounds": _one_sided_bounds(alarm_count, n_streams),
        "bet_epsilons": [float(e) for e in WCTM_BET_EPSILONS],
        "bet_weights": [float(w) for w in WCTM_BET_WEIGHTS],
        "alarm_rule": WCTM_ALARM_RULE,
        "validity_claim": "source_backed_ready_label_reference",
        "source": WCTM_SOURCE,
    }


def _wctm_shift_block(
    seed: int, config: Mapping[str, Any], threshold_c: float
) -> dict[str, Any]:
    shift_config = config.get("wctm_shift", {}) or {}
    t_change = int(shift_config.get("t_change", 60))
    shift_size = float(shift_config.get("shift_size", 2.0))
    n_runs = int(config.get("wctm_shift_streams", 300))
    horizon = int(config.get("wctm_shift_horizon", 240))
    lags = [int(lag) for lag in config.get("wctm_label_lags", [0, 10, 30])]
    scores = np.random.default_rng([int(seed), 201]).standard_normal((n_runs, horizon))
    if 0 <= t_change < horizon:
        scores[:, t_change:] += shift_size
    uniforms = np.random.default_rng([int(seed), 202]).uniform(size=(n_runs, horizon))
    ready_pvalues = _online_conformal_pvalues(scores, uniforms)
    rows = []
    for lag in lags:
        matured = _apply_label_lag(ready_pvalues, lag)
        wealth = _wctm_wealth_matrix(matured)
        rows.append(_delay_summary(wealth, threshold_c, t_change, lag, n_runs))
    return {
        "t_change": t_change,
        "shift_size": shift_size,
        "n_runs": n_runs,
        "horizon": horizon,
        "label_lags": lags,
        "rows": rows,
        "alarm_rule": WCTM_ALARM_RULE,
    }


def _delay_summary(
    wealth: np.ndarray,
    threshold_c: float,
    t_change: int,
    lag: int,
    n_runs: int,
) -> dict[str, Any]:
    crossed = wealth >= threshold_c
    ever_alarmed = crossed.any(axis=1)
    first_alarm = crossed.argmax(axis=1)
    delays: list[int] = []
    pre_change = 0
    for stream in range(n_runs):
        if not bool(ever_alarmed[stream]):
            continue
        index = int(first_alarm[stream])
        if index < t_change:
            pre_change += 1
        else:
            delays.append(int(index - t_change))
    eligible = n_runs - pre_change
    detected = len(delays)
    misses = eligible - detected
    return {
        "label_lag": int(lag),
        "n_runs": int(n_runs),
        "n_pre_change_false_alarms": int(pre_change),
        "n_detected": int(detected),
        "delays": delays,
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "median_delay_interval": _median_delay_interval(delays),
        "miss_rate": float(misses / eligible) if eligible else 1.0,
        "miss_rate_interval": _miss_rate_interval(misses, eligible),
        "threshold_c": float(threshold_c),
        "t_change": int(t_change),
    }


def _wctm_ooc_block(
    seed: int, config: Mapping[str, Any], threshold_c: float
) -> list[dict[str, Any]]:
    n_streams = int(config.get("wctm_ooc_streams", 500))
    lag = int(config.get("wctm_ooc_lag", 30))
    horizon = int(config.get("wctm_ooc_horizon", WCTM_OOC_HORIZON))
    return [
        _selective_observation_row(seed, n_streams, horizon, threshold_c),
        _drift_during_lag_row(seed, n_streams, horizon, lag, threshold_c),
    ]


def _selective_observation_row(
    seed: int, n_streams: int, horizon: int, threshold_c: float
) -> dict[str, Any]:
    """Labels only for approved instances: a score-selected monitored sub-stream."""
    scores = np.random.default_rng([int(seed), 301]).standard_normal(
        (n_streams, horizon)
    )
    uniforms = np.random.default_rng([int(seed), 302]).uniform(
        size=(n_streams, horizon)
    )
    cutoff = float(np.quantile(scores, WCTM_OOC_APPROVAL_QUANTILE))
    alarm_count = 0
    observed_total = 0
    for stream in range(n_streams):
        approved = scores[stream] <= cutoff
        observed = scores[stream][approved]
        observed_total += int(observed.size)
        if observed.size == 0:
            continue
        pvalues = _online_conformal_pvalues(
            observed.reshape(1, -1), uniforms[stream][approved].reshape(1, -1)
        )
        if float(_wctm_wealth_matrix(pvalues).max()) >= threshold_c:
            alarm_count += 1
    total_slots = n_streams * horizon
    return {
        "variant": "selective_observation",
        "n_streams": int(n_streams),
        "horizon": int(horizon),
        "alarm_count": int(alarm_count),
        "alarm_rate": alarm_count / n_streams if n_streams else 0.0,
        "alarm_rate_bounds": _one_sided_bounds(alarm_count, n_streams),
        "validity_claim": False,
        "claim_status": "hypothesis",
        "stream_is_iid": False,
        "mechanism": (
            "labels are observed only for approved instances (score-conditional "
            "selection by an approval cutoff), so the monitored stream is a "
            "selected sub-stream of the deployment stream and the ready-label "
            "observation process is not iid; measured without a validity claim"
        ),
        "approval_quantile": WCTM_OOC_APPROVAL_QUANTILE,
        "observation_rate": observed_total / total_slots if total_slots else 0.0,
        "alarm_rule": WCTM_ALARM_RULE,
    }


def _drift_during_lag_row(
    seed: int, n_streams: int, horizon: int, lag: int, threshold_c: float
) -> dict[str, Any]:
    """Score distribution steps while labels are in flight: non-iid, non-uniform."""
    scores = np.random.default_rng([int(seed), 401]).standard_normal(
        (n_streams, horizon)
    )
    t_drift = max(1, int(horizon * WCTM_OOC_DRIFT_QUANTILE))
    if t_drift < horizon:
        scores[:, t_drift:] += WCTM_OOC_DRIFT_SIZE
    uniforms = np.random.default_rng([int(seed), 402]).uniform(
        size=(n_streams, horizon)
    )
    matured = _apply_label_lag(_online_conformal_pvalues(scores, uniforms), lag)
    if 0 < lag < horizon:
        sample = matured[:, lag:].ravel()
    else:
        sample = matured.ravel()
    ks = stats.kstest(sample, "uniform")
    alarm_count = int((_wctm_wealth_matrix(matured).max(axis=1) >= threshold_c).sum())
    return {
        "variant": "drift_during_lag",
        "n_streams": int(n_streams),
        "horizon": int(horizon),
        "lag": int(lag),
        "t_drift": int(t_drift),
        "drift_size": WCTM_OOC_DRIFT_SIZE,
        "alarm_count": int(alarm_count),
        "alarm_rate": alarm_count / n_streams if n_streams else 0.0,
        "alarm_rate_bounds": _one_sided_bounds(alarm_count, n_streams),
        "validity_claim": False,
        "claim_status": "hypothesis",
        "stream_is_iid": False,
        "mechanism": (
            "the score distribution steps by "
            f"{WCTM_OOC_DRIFT_SIZE} sd while labels are still in flight "
            f"(label lag {lag}); the matured stream is non-stationary, so the "
            "online conformal p-values are non-uniform and the ready-label "
            "exchangeability assumption fails; measured without a validity claim"
        ),
        "pvalue_sample": [float(x) for x in sample],
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "alarm_rule": WCTM_ALARM_RULE,
    }


def _one_sided_bounds(count: int, n: int) -> dict[str, Any]:
    """Exact one-sided Clopper-Pearson bounds at ``WCTM_BOUND_LEVEL``."""
    return {
        "method": "clopper_pearson",
        "level": WCTM_BOUND_LEVEL,
        "lower": _cp_lower(count, n, WCTM_BOUND_LEVEL),
        "upper": _cp_upper(count, n, WCTM_BOUND_LEVEL),
    }


def _cp_lower(count: int, n: int, level: float) -> float:
    if count <= 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - level, count, n - count + 1))


def _cp_upper(count: int, n: int, level: float) -> float:
    if count >= n:
        return 1.0
    return float(stats.beta.ppf(level, count + 1, n - count))


def _cp_two_sided(count: int, n: int, level: float) -> tuple[float, float]:
    alpha = 1.0 - level
    low = 0.0 if count <= 0 else float(stats.beta.ppf(alpha / 2.0, count, n - count + 1))
    high = (
        1.0
        if count >= n
        else float(stats.beta.ppf(1.0 - alpha / 2.0, count + 1, n - count))
    )
    return low, high


def _median_delay_interval(delays: list[int]) -> dict[str, Any]:
    """Distribution-free order-statistic interval for the median delay."""
    values = sorted(float(delay) for delay in delays)
    median = float(np.median(values)) if values else float("nan")
    n = len(values)
    if n == 0:
        return {
            "method": "order_statistic_binomial",
            "level": WCTM_DELAY_LEVEL,
            "low": median,
            "high": median,
        }
    alpha = 1.0 - WCTM_DELAY_LEVEL
    rank = 1
    while rank <= n and float(stats.binom.cdf(rank - 1, n, 0.5)) <= alpha / 2.0:
        rank += 1
    low = min(values[min(rank - 1, n - 1)], median)
    high = max(values[max(n - rank, 0)], median)
    return {
        "method": "order_statistic_binomial",
        "level": WCTM_DELAY_LEVEL,
        "low": float(low),
        "high": float(high),
    }


def _miss_rate_interval(misses: int, eligible: int) -> dict[str, Any]:
    """Exact two-sided Clopper-Pearson interval for the miss rate."""
    low, high = _cp_two_sided(misses, eligible, WCTM_DELAY_LEVEL)
    return {
        "method": "clopper_pearson",
        "level": WCTM_DELAY_LEVEL,
        "low": float(low),
        "high": float(high),
    }


# --------------------------------------------------------------------------- #
# Round 2 (C22): stream monitoring observability under adversary drift with
# maturation lag. The shared realistic stream (exec.data) replaces the iid
# synthetic score streams of C12/C13: exactly one fraud typology drifts at
# ``t_drift``, labels mature per record (``label_arrival_index`` /
# ``label_observed``) and the deployed approval rule censors the highest scores.
#
# Pre-registered design (every size comes from the config, never from the data):
#   * steps: STREAM_MONITOR_STEPS equal slices ("days") of the run's arrival
#     order; the change sits at ``t_drift // step size``;
#   * monitored statistic: the sum of the top-k model scores inside a trailing
#     window of STREAM_MONITOR_WINDOW_STEPS steps. ``ready`` uses every arriving
#     record; ``matured`` uses only the records whose labels matured inside the
#     window (approved-only, so the matured view is a censored sub-stream);
#   * p-value: conformal rank of the step statistic among the pre-change
#     calibration steps, with an independent Unif[0, 1] tie-breaker (the Eq. 9
#     construction with a fixed calibration set instead of the online bag);
#   * wealth: the Eq. 6 equal mixture over eps in {-1, 0, 1}; alarm the first
#     step with wealth >= c.
# The ready-label guarantee is NOT claimed: drift and maturation break
# exchangeability, so the rows carry ``validity_claim`` False and
# ``claim_status`` "hypothesis" and are measured observations only.
#
# Measured at the CI config (stream_n 20000, t_drift 10000, n_runs 20): on the
# 20 drifted streams the monitor fires after the change with no pre-change false
# alarm in 9 (ready) / 3 (matured) runs, against 2 / 40 drift-free control
# streams for the same detector: the maturation lag delays and weakens
# detection, because the approval rule censors the drifted high scores.

STREAM_MONITOR_STEPS = 100
STREAM_MONITOR_WINDOW_STEPS = 3
STREAM_MONITOR_TOP_K = 3
STREAM_MONITOR_TIE_RNG_TAG = 909
STREAM_MONITOR_TAIL_QUANTILE = 0.999
STREAM_MONITOR_BUDGET_QUANTILES: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)
STREAM_MONITOR_ALARM_RULE = "first monitored step with wealth M_t >= c (M_0 = 1)"
STREAM_MONITOR_STATISTIC = (
    "sum of the top-k model scores of a trailing window of "
    f"{STREAM_MONITOR_WINDOW_STEPS} steps"
)
STREAM_MONITOR_PVALUE_METHOD = "conformal_rank_fixed_calibration_uniform_tiebreak"
STREAM_MONITOR_DRIFT_NAME = "alert_band_excess_mass_per_1000"
STREAM_MONITOR_DRIFT_DEFINITION = (
    "per window: 1000 * sum over the budget-exceeding alerts of (score - alert "
    "threshold) / window size, i.e. the observed-minus-budget excess risk mass "
    "of the alert band; a one-bin, tail-weighted PSI-style marginal-shift index "
    "on the model score (the binned population stability index is reported "
    "alongside as psi_values)"
)
STREAM_MONITOR_SOURCE = (
    "Prinster et al. 2505.04608 v4 (corrected version, as ingested): Eq. 6 "
    "betting form h_eps(p) = 1 + eps (p - 0.5), eps in {-1, 0, 1}; Eq. 9 "
    "weighted conformal p-value with u ~ Unif[0, 1]. The one-sided guarantee "
    "P(ever alarm) <= 1/c is a ready-label statement and is NOT claimed for this "
    "stream: the adversary drift plus approved-only maturation breaks "
    "exchangeability, so every row is a measured observation."
)

_STREAM_MONITOR_CACHE: dict[str, dict[str, Any]] = {}


def run_stream_monitoring_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic C22 stream-monitoring experiment for ``config``.

    Returns the alert-budget view of the shared stream (``windows`` with
    ``start``/``end``/``alert_rate``/``exceeds_budget``), a PSI-style
    ``drift_statistic`` over the same windows, and the ``wctm`` summaries for
    the ``ready`` and ``matured`` label views. Each ``wctm`` row reports the
    delay and miss rate of the WCTM on ``config["n_runs"]`` independently seeded
    streams, with an order-statistic median-delay interval and an exact
    two-sided Clopper-Pearson miss-rate interval, computable from the returned
    per-run ``delays`` alone.
    """
    key = config_key(config)
    if key not in _STREAM_MONITOR_CACHE:
        _STREAM_MONITOR_CACHE[key] = _run_stream_monitoring(config)
    return copy.deepcopy(_STREAM_MONITOR_CACHE[key])


def _run_stream_monitoring(config: Mapping[str, Any]) -> dict[str, Any]:
    from exec.data import generate_realistic_stream  # local: keep Round-1 import light

    seed = seed_of(config)
    n_windows = max(1, int(config.get("n_windows", 10)))
    n_runs = max(1, int(config.get("n_runs", 20)))
    threshold_c = float(config.get("wctm_threshold_c", 20.0))
    alert_budget = float(config.get("alert_budget", 0.01))

    stream = generate_realistic_stream(config)
    day_size = max(1, int(stream["n"]) // STREAM_MONITOR_STEPS)
    t_change_step = min(
        STREAM_MONITOR_STEPS, int(stream["t_drift"]) // day_size
    )

    windows, drift = _stream_windows_and_drift(stream, n_windows, alert_budget)
    daily = _stream_daily_series(
        stream, drift["alert_threshold"], alert_budget, day_size
    )
    wctm = {
        variant: _stream_wctm_summary(
            config, variant, seed, day_size, t_change_step, threshold_c
        )
        for variant in ("ready", "matured")
    }
    return {
        "seed": int(seed),
        "config_hash": config_hash(config),
        "stream_n": int(stream["n"]),
        "t_drift": int(stream["t_drift"]),
        "alert_budget": float(alert_budget),
        "n_windows": int(n_windows),
        "n_runs": int(n_runs),
        "wctm_threshold_c": float(threshold_c),
        "windows": windows,
        "daily": daily,
        "drift_statistic": drift,
        "wctm": wctm,
        "alert_rule": {
            "description": (
                "budgeted alert rule: a record is alerted when its model score "
                "is at or above the (1 - alert_budget) quantile of the pre-change "
                "reference scores"
            ),
            "alert_budget": float(alert_budget),
            "threshold": drift["alert_threshold"],
            "reference": "records with arrival_index < stream_t_drift",
        },
        "monitor_design": {
            "steps_per_run": STREAM_MONITOR_STEPS,
            "step_size_records": int(day_size),
            "window_steps": STREAM_MONITOR_WINDOW_STEPS,
            "top_k": STREAM_MONITOR_TOP_K,
            "burn_in_steps": STREAM_MONITOR_WINDOW_STEPS,
            "calibration": "pre-change steps [burn_in_steps, t_change_step)",
            "statistic": STREAM_MONITOR_STATISTIC,
            "pvalue_method": STREAM_MONITOR_PVALUE_METHOD,
            "alarm_rule": STREAM_MONITOR_ALARM_RULE,
            "tie_break_rng_tag": STREAM_MONITOR_TIE_RNG_TAG,
            "source": STREAM_MONITOR_SOURCE,
        },
        "claim_status": "hypothesis",
        "validity_claim": False,
        "source": STREAM_MONITOR_SOURCE,
    }


# --------------------------------------------------------------------------- #
# windows, alert rate and the reported drift statistic


def _stream_windows_and_drift(
    stream: Mapping[str, Any], n_windows: int, alert_budget: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transactions = stream["transactions"]
    n = len(transactions)
    scores = np.asarray(
        [float(t["model_score"]) for t in transactions], dtype=float
    )
    arrivals = np.asarray(
        [int(t["arrival_index"]) for t in transactions], dtype=int
    )
    t_drift = int(stream["t_drift"])
    pre_change = arrivals < t_drift
    reference = scores[pre_change] if bool(pre_change.any()) else scores

    alert_quantile = float(min(max(1.0 - alert_budget, 0.0), 1.0))
    alert_threshold = float(np.quantile(reference, alert_quantile))
    tail_threshold = float(np.quantile(reference, STREAM_MONITOR_TAIL_QUANTILE))

    bounds = [int(round(index * n / n_windows)) for index in range(n_windows + 1)]
    windows: list[dict[str, Any]] = []
    drift_values: list[float] = []
    psi_values: list[float] = []
    for index in range(n_windows):
        start = min(bounds[index], n)
        end = min(bounds[index + 1], n)
        end = max(end, start)
        window_scores = scores[start:end]
        size = int(window_scores.size)
        alerts = window_scores[window_scores >= alert_threshold]
        alert_rate = float(alerts.size / size) if size else 0.0
        if size:
            excess_mass = (
                float(np.sum(alerts - alert_threshold)) * 1000.0 / float(size)
            )
        else:
            excess_mass = 0.0
        drift_values.append(excess_mass)
        psi_values.append(_window_psi(reference, window_scores))
        windows.append(
            {
                "start": int(start),
                "end": int(end),
                "size": int(size),
                "alert_count": int(alerts.size),
                "alert_rate": float(alert_rate),
                "alerts_per_1000": float(1000.0 * alert_rate),
                "exceeds_budget": bool(alert_rate > alert_budget),
                "drift_statistic": float(excess_mass),
            }
        )

    pre_values = [
        float(value)
        for value, window in zip(drift_values, windows)
        if window["end"] <= t_drift
    ]
    post_values = [
        float(value)
        for value, window in zip(drift_values, windows)
        if window["start"] >= t_drift
    ]
    threshold = (
        float(max(pre_values)) if pre_values else float("nan")
    )
    first_alarm = next(
        (
            index
            for index, window in enumerate(windows)
            if window["start"] >= t_drift
            and float(drift_values[index]) > threshold
        ),
        None,
    )
    drift = {
        "name": STREAM_MONITOR_DRIFT_NAME,
        "definition": STREAM_MONITOR_DRIFT_DEFINITION,
        "values": [float(value) for value in drift_values],
        "pre_drift_mean": (
            float(np.mean(pre_values)) if pre_values else float("nan")
        ),
        "post_drift_mean": (
            float(np.mean(post_values)) if post_values else float("nan")
        ),
        "threshold": threshold,
        "threshold_rule": (
            "maximum of the pre-change window values (a pre-change envelope "
            "calibrated without any post-change value)"
        ),
        "first_alarm_window": (
            int(first_alarm) if first_alarm is not None else None
        ),
        "psi_values": [float(value) for value in psi_values],
        "alert_threshold": float(alert_threshold),
        "tail_quantile": float(STREAM_MONITOR_TAIL_QUANTILE),
        "tail_threshold": float(tail_threshold),
        "reference": "records with arrival_index < stream_t_drift",
    }
    return windows, drift


def _stream_daily_series(
    stream: Mapping[str, Any],
    alert_threshold: float,
    alert_budget: float,
    day_size: int,
) -> dict[str, Any]:
    """Per-step alert-rate and drift-statistic series for the timeline panels."""
    transactions = stream["transactions"]
    n = len(transactions)
    scores = np.asarray(
        [float(t["model_score"]) for t in transactions], dtype=float
    )
    step_index: list[int] = []
    alert_count: list[int] = []
    alert_rate: list[float] = []
    drift_statistic: list[float] = []
    for step in range(STREAM_MONITOR_STEPS):
        start = step * day_size
        end = min(n, (step + 1) * day_size)
        if start >= n:
            break
        values = scores[start:end]
        size = int(values.size)
        alerts = values[values >= alert_threshold]
        rate = float(alerts.size / size) if size else 0.0
        step_index.append(int(step))
        alert_count.append(int(alerts.size))
        alert_rate.append(rate)
        drift_statistic.append(
            float(np.sum(alerts - alert_threshold)) * 1000.0 / size
            if size
            else 0.0
        )
    return {
        "definition": STREAM_MONITOR_DRIFT_DEFINITION,
        "step_index": step_index,
        "alert_count": alert_count,
        "alert_rate": alert_rate,
        "alerts_per_1000": [float(1000.0 * rate) for rate in alert_rate],
        "drift_statistic": drift_statistic,
        "alert_budget": float(alert_budget),
        "alert_threshold": float(alert_threshold),
        "t_drift_step": int(int(stream["t_drift"]) // day_size),
        "step_size_records": int(day_size),
    }


def _window_psi(reference: np.ndarray, sample: np.ndarray) -> float:
    """Binned population stability index of ``sample`` against ``reference``.

    The bins are the reference quantiles in ``STREAM_MONITOR_BUDGET_QUANTILES``
    with the two extreme bins open, so the index is tail-weighted; a flat score
    distribution of a drifted stream barely moves it (see the C22 notes).
    """
    if reference.size < 2 or sample.size == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, STREAM_MONITOR_BUDGET_QUANTILES))
    if edges.size < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    expected = np.histogram(reference, edges)[0] / float(reference.size)
    observed = np.histogram(sample, edges)[0] / float(sample.size)
    expected = np.clip(expected, 1e-9, None)
    observed = np.clip(observed, 1e-9, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


# --------------------------------------------------------------------------- #
# the WCTM on the stream: ready labels versus matured labels


def _stream_monitor_statistic(
    stream: Mapping[str, Any], variant: str, day_size: int
) -> np.ndarray:
    """Per-step monitored statistic for one label view of one stream."""
    transactions = stream["transactions"]
    n = len(transactions)
    scores = np.asarray(
        [float(t["model_score"]) for t in transactions], dtype=float
    )
    label_arrival = np.asarray(
        [int(t["label_arrival_index"]) for t in transactions], dtype=int
    )
    label_observed = np.asarray(
        [bool(t["label_observed"]) for t in transactions], dtype=bool
    )
    window = STREAM_MONITOR_WINDOW_STEPS * day_size
    statistic = np.zeros(STREAM_MONITOR_STEPS, dtype=float)
    for step in range(STREAM_MONITOR_STEPS):
        high = min(n, (step + 1) * day_size)
        low = max(0, high - window)
        if variant == "ready":
            values = scores[low:high]
        else:
            mask = label_observed & (label_arrival >= low) & (label_arrival < high)
            values = scores[mask]
        if values.size:
            k = min(STREAM_MONITOR_TOP_K, int(values.size))
            statistic[step] = float(np.sort(values)[-k:].sum())
    return statistic


def _stream_monitor_run(
    stream: Mapping[str, Any],
    variant: str,
    day_size: int,
    t_change_step: int,
    threshold_c: float,
) -> dict[str, Any]:
    """One WCTM run: statistic, p-values, wealth and the first alarm step."""
    statistic = _stream_monitor_statistic(stream, variant, day_size)
    burn_in = min(STREAM_MONITOR_WINDOW_STEPS, max(0, t_change_step - 1))
    calibration = list(range(burn_in, t_change_step))
    if not calibration:
        calibration = [max(0, t_change_step - 1)]
    steps = list(range(burn_in, STREAM_MONITOR_STEPS))
    seed = int(stream["seed"])
    uniforms = np.random.default_rng(
        [seed, int(STREAM_MONITOR_TIE_RNG_TAG)]
    ).uniform(size=STREAM_MONITOR_STEPS)
    pvalues = np.empty(len(steps), dtype=float)
    for position, step in enumerate(steps):
        reference_steps = [j for j in calibration if j != step]
        if not reference_steps:
            reference_steps = calibration
        reference = statistic[reference_steps]
        greater = int(np.sum(reference > statistic[step]))
        equal = int(np.sum(reference == statistic[step]))
        pvalues[position] = (
            1.0 + greater + uniforms[step] * equal
        ) / (len(reference) + 1.0)
    wealth = _wctm_wealth_matrix(pvalues.reshape(1, -1))[0]
    crossed = np.nonzero(wealth >= threshold_c)[0]
    first_alarm = int(crossed[0]) + burn_in if crossed.size else None
    return {
        "statistic": statistic,
        "steps": steps,
        "pvalues": pvalues,
        "wealth": wealth,
        "burn_in": int(burn_in),
        "first_alarm_step": first_alarm,
    }


def _stream_wctm_summary(
    config: Mapping[str, Any],
    variant: str,
    base_seed: int,
    day_size: int,
    t_change_step: int,
    threshold_c: float,
) -> dict[str, Any]:
    from exec.data import generate_realistic_stream  # local import

    n_runs = max(1, int(config.get("n_runs", 20)))
    delays: list[int] = []
    pre_change_alarms = 0
    details: list[dict[str, Any]] = []
    statistic_paths: list[np.ndarray] = []
    wealth_paths: list[np.ndarray] = []
    steps: list[int] = []
    alarm_run: int | None = None
    for run in range(n_runs):
        run_seed = int(base_seed) + run
        run_stream = generate_realistic_stream({**dict(config), "seed": run_seed})
        result = _stream_monitor_run(
            run_stream, variant, day_size, t_change_step, threshold_c
        )
        steps = result["steps"]
        statistic_paths.append(result["statistic"][steps[0]:])
        wealth_paths.append(result["wealth"])
        first_alarm = result["first_alarm_step"]
        if first_alarm is None:
            details.append(
                {
                    "run": run,
                    "seed": run_seed,
                    "first_alarm_step": None,
                    "delay": None,
                    "pre_change_alarm": False,
                    "max_wealth": float(result["wealth"].max()),
                }
            )
            continue
        if first_alarm < t_change_step:
            pre_change_alarms += 1
            details.append(
                {
                    "run": run,
                    "seed": run_seed,
                    "first_alarm_step": int(first_alarm),
                    "delay": None,
                    "pre_change_alarm": True,
                    "max_wealth": float(result["wealth"].max()),
                }
            )
            continue
        delays.append(int(first_alarm - t_change_step))
        if alarm_run is None:
            alarm_run = run
        details.append(
            {
                "run": run,
                "seed": run_seed,
                "first_alarm_step": int(first_alarm),
                "delay": int(first_alarm - t_change_step),
                "pre_change_alarm": False,
                "max_wealth": float(result["wealth"].max()),
            }
        )

    eligible = n_runs - pre_change_alarms
    detected = len(delays)
    ordered_delays = sorted(int(delay) for delay in delays)
    median_delay = (
        float(np.median(ordered_delays)) if ordered_delays else float("nan")
    )
    misses = eligible - detected
    if eligible > 0:
        miss_rate = float(misses / eligible)
        miss_interval = _miss_rate_interval(misses, eligible)
    else:
        miss_rate = 1.0
        miss_interval = {
            "method": "clopper_pearson",
            "level": WCTM_DELAY_LEVEL,
            "low": float("nan"),
            "high": float("nan"),
        }
    statistic_matrix = np.vstack(statistic_paths) if statistic_paths else np.zeros((1, 1))
    wealth_matrix = np.vstack(wealth_paths) if wealth_paths else np.zeros((1, 1))
    chosen = alarm_run if alarm_run is not None else int(
        np.argmax([float(row.max()) for row in wealth_paths])
    )
    return {
        "variant": variant,
        "n_runs": int(n_runs),
        "n_pre_change_false_alarms": int(pre_change_alarms),
        "n_detected": int(detected),
        "n_eligible": int(eligible),
        "delays": ordered_delays,
        "median_delay": median_delay,
        "median_delay_interval": _median_delay_interval(ordered_delays),
        "miss_rate": miss_rate,
        "miss_rate_interval": miss_interval,
        "threshold_c": float(threshold_c),
        "t_change_step": int(t_change_step),
        "step_size_records": int(day_size),
        "window_steps": STREAM_MONITOR_WINDOW_STEPS,
        "top_k": STREAM_MONITOR_TOP_K,
        "statistic": STREAM_MONITOR_STATISTIC,
        "pvalue_method": STREAM_MONITOR_PVALUE_METHOD,
        "alarm_rule": STREAM_MONITOR_ALARM_RULE,
        "validity_claim": False,
        "claim_status": "hypothesis",
        "run_details": details,
        "step_index": [int(step) for step in steps],
        "median_statistic_path": [
            float(value) for value in np.median(statistic_matrix, axis=0)
        ],
        "median_log_wealth_path": [
            float(value)
            for value in np.log(np.maximum(np.median(wealth_matrix, axis=0), WEALTH_FLOOR))
        ],
        "alarm_run": int(chosen),
        "alarm_run_seed": int(details[chosen]["seed"]) if details else None,
        "alarm_first_step": details[chosen]["first_alarm_step"] if details else None,
        "alarm_delay": details[chosen]["delay"] if details else None,
        "alarm_log_wealth_path": [
            float(value)
            for value in np.log(np.maximum(wealth_matrix[chosen], WEALTH_FLOOR))
        ],
    }


# --------------------------------------------------------------------------- #
# Round 3 (C25): equal-footing monitor comparison on the classification
# nonconformity score 1 - p_hat(y|x) (Prinster et al. 2505.04608 v4,
# Appendix E.1, as ingested).
#
# The three monitors of the project's comparison -- ``alert_rate`` (budgeted
# alert incidence), ``drift_stat`` (tail-weighted excess mass) and ``wctm``
# (the weighted-conformal test martingale) -- are evaluated on the SAME runs
# with the SAME sequential alarm rule, and each reports false-alarm incidence
# (exact one-sided Clopper-Pearson upper bound), detection delay and miss rate:
# no monitor gets a metric the others don't.
#
# Design (every constant is pre-registered here, no size tuned from results):
#   * the population/model is built ONCE by ``generate_realistic_stream``,
#     which trains the notebook's LogisticRegression classifier on the
#     approved/label-observed training split (``score_model`` provenance is
#     echoed back in the result); a record's classification nonconformity
#     score is ``1 - p_hat(y|x)`` for its observed label -- for y = 1 it is
#     ``1 - model_score`` and for y = 0 it is ``model_score``. Never a
#     synthetic sign process;
#   * a run is ``stream_n`` records in EQUAL_FOOTING_STEPS equal steps; the
#     change sits at step ``stream_t_drift // step_size``;
#   * the null run draws every record i.i.d. from the base (pre-change) score
#     law. The drift run keeps that law before the change and, after it,
#     replaces each record -- with probability equal to the drifted typology's
#     post-change share -- by an independent draw from the drifted typology's
#     post-change (score, label) law: a genuine change in the score law at the
#     change point, the drifted population's scores mixed into the base law;
#   * per step the three monitored statistics are the alert rate at the
#     pre-registered budget quantile of the base scores, the tail-weighted
#     excess mass above that quantile (the C22 drift statistic), and the mean
#     classification nonconformity score;
#   * every monitor uses the same sequential test: the Eq. 9 conformal rank of
#     the step statistic against the pre-change calibration steps with an
#     independent uniform tie-breaker, the Eq. 6 equal mixture over
#     eps in {-1, 0, 1}, and the first step with wealth M_t >= c. A pre-change
#     first alarm is a false alarm for that run; only the remaining runs are
#     eligible for the delay and miss-rate summaries (the C12/C22 convention).

EQUAL_FOOTING_STEPS = 100
EQUAL_FOOTING_BURN_IN_STEPS = 1
EQUAL_FOOTING_MONITORS: tuple[str, ...] = ("alert_rate", "drift_stat", "wctm")
EQUAL_FOOTING_SCORE_KIND = "classification"
EQUAL_FOOTING_SCORE_FORMULA = "1 - p_hat(y|x)"
EQUAL_FOOTING_SCORE_DEFINITION = (
    "classification nonconformity score 1 - p_hat(y|x) (Prinster et al., "
    "Appendix E.1): the step mean over the step's records of "
    "1 - p_hat(y|x) with the observed label y, so 1 - model_score for "
    "y = 1 and model_score for y = 0"
)
EQUAL_FOOTING_ALARM_RULE = "first monitored step t with wealth M_t >= c (M_0 = 1)"
EQUAL_FOOTING_NULL_RNG_TAG = 921
EQUAL_FOOTING_DRIFT_RNG_TAG = 922
EQUAL_FOOTING_TIE_RNG_TAG = 923
EQUAL_FOOTING_SOURCE = (
    "Prinster et al. 2505.04608 v4 (corrected version, as ingested): the "
    "classification nonconformity score is 1 - p_hat(y|x) (Appendix E.1); "
    "Eq. 6 betting form h_eps(p) = 1 + eps (p - 0.5), eps in {-1, 0, 1}; "
    "Eq. 9 weighted conformal p-value with u ~ Unif[0, 1]"
)
EQUAL_FOOTING_DRIFT_RULE = (
    "after the change point each record is replaced, with probability equal "
    "to the drifted typology's post-change share, by an independent draw from "
    "the drifted typology's post-change (score, label) law; before the change "
    "point, and in the whole null run, records are i.i.d. draws from the base "
    "(pre-change) score law"
)
EQUAL_FOOTING_CLAIM_STATUS = "hypothesis"

_EQUAL_FOOTING_CACHE: dict[str, dict[str, Any]] = {}


def run_equal_footing_monitor_comparison(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the deterministic C25 equal-footing monitor comparison for ``config``.

    Every monitor reports, on the same 300 null and 300 drift runs: the
    false-alarm incidence on the null stream (with the exact one-sided
    Clopper-Pearson upper bound at ``level``), and -- on the drift stream --
    the pre-change false alarms, detection delays with a distribution-free
    median-delay interval and the miss rate with an exact two-sided
    Clopper-Pearson interval. The WCTM is built from the classification
    nonconformity score ``1 - p_hat(y|x)`` of the stream's trained model.
    """
    key = "equal_footing::" + config_key(config)
    if key not in _EQUAL_FOOTING_CACHE:
        _EQUAL_FOOTING_CACHE[key] = _run_equal_footing_monitor_comparison(config)
    return copy.deepcopy(_EQUAL_FOOTING_CACHE[key])


def _run_equal_footing_monitor_comparison(config: Mapping[str, Any]) -> dict[str, Any]:
    from exec.data import generate_realistic_stream  # local: keep Round-1 import light

    seed = seed_of(config)
    n = int(config.get("stream_n", 0))
    if n <= 0:
        raise ValueError("config['stream_n'] must be a positive integer")
    t_drift = int(config.get("stream_t_drift", n // 2))
    n_null_runs = int(config.get("n_null_runs", 300))
    n_drift_runs = int(config.get("n_drift_runs", 300))
    threshold_c = float(config.get("wctm_threshold_c", 20.0))
    alert_budget = float(config.get("alert_budget", 0.01))

    stream = generate_realistic_stream(config)
    base_scores, base_labels, drift_law, drift_share, drift_typology = (
        _equal_footing_laws(stream, t_drift)
    )

    step_size = max(1, n // EQUAL_FOOTING_STEPS)
    n_steps = max(1, n // step_size)
    t_change_step = min(n_steps, max(1, t_drift // step_size))
    alert_threshold = float(np.quantile(base_scores, min(max(1.0 - alert_budget, 0.0), 1.0)))

    null_stats = _equal_footing_step_statistics(
        _equal_footing_runs(
            np.random.default_rng([seed, EQUAL_FOOTING_NULL_RNG_TAG]),
            n_null_runs, n, t_drift, base_scores, base_labels, drift_law, drift_share, False,
        ),
        n_steps, step_size, alert_threshold,
    )
    drift_stats = _equal_footing_step_statistics(
        _equal_footing_runs(
            np.random.default_rng([seed, EQUAL_FOOTING_DRIFT_RNG_TAG]),
            n_drift_runs, n, t_drift, base_scores, base_labels, drift_law, drift_share, True,
        ),
        n_steps, step_size, alert_threshold,
    )

    monitors: dict[str, dict[str, Any]] = {}
    for index, monitor in enumerate(EQUAL_FOOTING_MONITORS):
        null_first = _equal_footing_alarm_steps(
            null_stats[monitor],
            np.random.default_rng([seed, EQUAL_FOOTING_TIE_RNG_TAG, index, 0]).uniform(
                size=(n_null_runs, n_steps)
            ),
            t_change_step, n_steps, threshold_c,
        )
        drift_first = _equal_footing_alarm_steps(
            drift_stats[monitor],
            np.random.default_rng([seed, EQUAL_FOOTING_TIE_RNG_TAG, index, 1]).uniform(
                size=(n_drift_runs, n_steps)
            ),
            t_change_step, n_steps, threshold_c,
        )
        row: dict[str, Any] = {
            "null": _equal_footing_null_row(null_first, n_null_runs),
            "drift": _equal_footing_drift_row(drift_first, n_drift_runs, t_change_step),
            "monitored_statistic": _equal_footing_statistic_definition(monitor),
            "alarm_rule": EQUAL_FOOTING_ALARM_RULE,
        }
        if monitor == "wctm":
            row["score_definition"] = EQUAL_FOOTING_SCORE_DEFINITION
        monitors[monitor] = row

    return {
        "score_kind": EQUAL_FOOTING_SCORE_KIND,
        "score_formula": EQUAL_FOOTING_SCORE_FORMULA,
        "n_null_runs": int(n_null_runs),
        "n_drift_runs": int(n_drift_runs),
        "monitors": monitors,
        "seed": int(seed),
        "config_hash": config_hash(config),
        "stream_n": int(n),
        "t_drift": int(t_drift),
        "steps_per_run": int(n_steps),
        "step_size_records": int(step_size),
        "t_change_step": int(t_change_step),
        "wctm_threshold_c": float(threshold_c),
        "alert_budget": float(alert_budget),
        "alert_threshold": float(alert_threshold),
        "score_model": {
            "name": str(stream["score_model"]["name"]),
            "feature_names": list(stream["score_model"]["feature_names"]),
            "train_subset": str(stream["score_model"]["train_subset"]),
            "n_train": int(stream["score_model"]["n_train"]),
            "n_train_positives": int(stream["score_model"]["n_train_positives"]),
        },
        "drift_law": {
            "rule": EQUAL_FOOTING_DRIFT_RULE,
            "drift_typology": drift_typology,
            "drifted_records": int(drift_law[0].size),
            "post_change_records": int(n - t_drift),
            "drifted_share": float(drift_share),
        },
        "claim_status": EQUAL_FOOTING_CLAIM_STATUS,
        "source": EQUAL_FOOTING_SOURCE,
    }


# --------------------------------------------------------------------------- #
# C25 internals


def _equal_footing_laws(
    stream: Mapping[str, Any], t_drift: int
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray], float, str]:
    """Base (pre-change) score law and the drifted typology's post-change law."""
    transactions = stream["transactions"]
    scores = np.asarray([float(t["model_score"]) for t in transactions], dtype=float)
    labels = np.asarray([int(t["Y"]) for t in transactions], dtype=int)
    arrivals = np.asarray([int(t["arrival_index"]) for t in transactions], dtype=int)
    typology = np.asarray([str(t["typology"]) for t in transactions], dtype=object)
    pre_change = arrivals < t_drift
    post_change = ~pre_change
    if not pre_change.any() or not post_change.any():
        raise ValueError("C25 needs both pre-change and post-change records in the stream")
    drift_typology = str(stream.get("drift_typology", ""))
    drifted = post_change & (typology == drift_typology)
    if not drifted.any():
        drifted = post_change
    base = (scores[pre_change], labels[pre_change])
    law = (scores[drifted], labels[drifted])
    share = float(drifted.sum()) / float(post_change.sum())
    return base[0], base[1], law, share, drift_typology


def _equal_footing_runs(
    rng: np.random.Generator,
    n_runs: int,
    n: int,
    t_drift: int,
    base_scores: np.ndarray,
    base_labels: np.ndarray,
    drift_law: tuple[np.ndarray, np.ndarray],
    drift_share: float,
    drift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_runs`` score streams from the base law (and the drift law)."""
    index = rng.integers(0, base_scores.size, size=(n_runs, n))
    scores = base_scores[index]
    labels = base_labels[index]
    post_length = n - t_drift
    if drift and post_length > 0:
        drifted_scores, drifted_labels = drift_law
        replaced = rng.random((n_runs, post_length)) < drift_share
        draw = rng.integers(0, drifted_scores.size, size=(n_runs, post_length))
        scores[:, t_drift:] = np.where(replaced, drifted_scores[draw], scores[:, t_drift:])
        labels[:, t_drift:] = np.where(replaced, drifted_labels[draw], labels[:, t_drift:])
    return scores, labels


def _equal_footing_step_statistics(
    run: tuple[np.ndarray, np.ndarray],
    n_steps: int,
    step_size: int,
    alert_threshold: float,
) -> dict[str, np.ndarray]:
    """Per-step alert rate, tail-weighted excess mass and classification score."""
    scores, labels = run
    usable = n_steps * step_size
    score_steps = scores[:, :usable].reshape(-1, n_steps, step_size)
    nonconformity = np.where(labels[:, :usable] == 1, 1.0 - scores[:, :usable], scores[:, :usable])
    alerts = score_steps >= alert_threshold
    return {
        "alert_rate": alerts.mean(axis=2),
        "drift_stat": np.where(alerts, score_steps - alert_threshold, 0.0).sum(axis=2)
        * 1000.0
        / float(step_size),
        "wctm": nonconformity.reshape(-1, n_steps, step_size).mean(axis=2),
    }


def _equal_footing_alarm_steps(
    statistic: np.ndarray,
    tie_uniforms: np.ndarray,
    t_change_step: int,
    n_steps: int,
    threshold_c: float,
) -> np.ndarray:
    """First monitored step with wealth >= c, or -1; the shared sequential test."""
    n_runs = statistic.shape[0]
    burn_in = min(EQUAL_FOOTING_BURN_IN_STEPS, max(0, t_change_step - 1))
    calibration = list(range(burn_in, t_change_step))
    if not calibration:
        calibration = [max(0, t_change_step - 1)]
    steps = list(range(burn_in, n_steps))
    pvalues = np.empty((n_runs, len(steps)), dtype=float)
    for position, step in enumerate(steps):
        reference = [j for j in calibration if j != step]
        if not reference:
            reference = calibration
        values = statistic[:, reference]
        test = statistic[:, step : step + 1]
        greater = (values > test).sum(axis=1)
        equal = (values == test).sum(axis=1)
        pvalues[:, position] = (
            1.0 + greater + tie_uniforms[:, step] * equal
        ) / (len(reference) + 1.0)
    wealth = _wctm_wealth_matrix(pvalues)
    crossed = wealth >= threshold_c
    ever = crossed.any(axis=1)
    return np.where(ever, crossed.argmax(axis=1) + burn_in, -1).astype(int)


def _equal_footing_null_row(first_alarm: np.ndarray, n_runs: int) -> dict[str, Any]:
    """False-alarm incidence on the null stream with the exact CP upper bound."""
    false_alarms = int((first_alarm >= 0).sum())
    return {
        "n_runs": int(n_runs),
        "n_false_alarms": false_alarms,
        "empirical_false_alarm_rate": false_alarms / n_runs if n_runs else 0.0,
        "false_alarm_upper_bound": _cp_upper(false_alarms, n_runs, WCTM_BOUND_LEVEL),
        "level": WCTM_BOUND_LEVEL,
        "method": "clopper_pearson",
    }


def _equal_footing_drift_row(
    first_alarm: np.ndarray, n_runs: int, t_change_step: int
) -> dict[str, Any]:
    """Delay and miss-rate summary on the drift stream (C12/C22 convention)."""
    pre_change = int(((first_alarm >= 0) & (first_alarm < t_change_step)).sum())
    eligible = int(n_runs - pre_change)
    delays = sorted(int(step - t_change_step) for step in first_alarm[first_alarm >= t_change_step])
    detected = len(delays)
    misses = eligible - detected
    return {
        "n_runs": int(n_runs),
        "n_pre_change_false_alarms": pre_change,
        "n_detected": int(detected),
        "delays": delays,
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "median_delay_interval": _median_delay_interval(delays),
        "miss_rate": float(misses / eligible) if eligible else 1.0,
        "miss_rate_interval": _miss_rate_interval(misses, eligible)
        if eligible
        else {
            "method": "clopper_pearson",
            "level": WCTM_DELAY_LEVEL,
            "low": float("nan"),
            "high": float("nan"),
        },
        "t_change_step": int(t_change_step),
        "alarm_rule": EQUAL_FOOTING_ALARM_RULE,
    }


def _equal_footing_statistic_definition(monitor: str) -> str:
    if monitor == "alert_rate":
        return (
            "per-step fraction of records whose model score is at or above the "
            "pre-registered (1 - alert_budget) quantile of the base pre-change scores"
        )
    if monitor == "drift_stat":
        return (
            "per-step 1000 * sum over the budget-exceeding alerts of "
            "(score - alert threshold) / step size, the C22 tail-weighted excess mass"
        )
    return (
        "per-step mean of the classification nonconformity score 1 - p_hat(y|x) "
        "with the observed label, the Prinster Appendix E.1 score"
    )
