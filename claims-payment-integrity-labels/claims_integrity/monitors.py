"""Monitoring instruments with pre-declared hypotheses and trigger scenarios.

Both monitors are *derived here* observability devices.  Their hypotheses and
trigger rules are fixed before the simulation runs, and their measured
false-alarm and miss rates are reported (a monitor that never fires
would score a perfect miss rate, so both scenarios are exercised).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta


def pairwise_agreement(labels: np.ndarray) -> float:
    """Mean pairwise agreement among observed reviewer labels per item."""

    y = np.asarray(labels)
    total, agree = 0, 0
    for row in y:
        observed = row[row >= 0]
        for i in range(observed.size):
            for j in range(i + 1, observed.size):
                total += 1
                agree += int(observed[i] == observed[j])
    return float(agree / total) if total else float("nan")


def agreement_drift_monitor(
    n_replicates: int = 300,
    window: int = 10,
    threshold: float = 0.60,
    stationary_mean: float = 0.70,
    drift_mean: float = 0.50,
    concentration: float = 24.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Detect a drop in reviewer label agreement.

    H0: windowed mean agreement >= threshold.  H1 (drift scenario): agreement
    drops to ``drift_mean``.  The monitor triggers when a full window's mean
    agreement is below ``threshold``.  Series are Beta draws around the two
    means (same concentration), so the measured rates are properties of the
    declared rule, not tuned on the outcome.
    """

    rng = np.random.default_rng(seed)

    def run(mean: float) -> int:
        alpha = mean * concentration
        beta_param = (1.0 - mean) * concentration
        series = beta.rvs(alpha, beta_param, size=window, random_state=rng)
        return int(series.mean() < threshold)

    false_alarms = sum(run(stationary_mean) for _ in range(n_replicates))
    misses = sum(1 - run(drift_mean) for _ in range(n_replicates))
    return {
        "monitor": "reviewer_agreement_drift",
        "hypothesis": f"windowed mean agreement >= {threshold}",
        "trigger_scenario": f"agreement drops to {drift_mean} over a {window}-observation window",
        "window": int(window),
        "threshold": float(threshold),
        "replicates": int(n_replicates),
        "false_alarm_rate": float(false_alarms / n_replicates),
        "false_alarm_interval": _wilson(false_alarms, n_replicates),
        "miss_rate": float(misses / n_replicates),
        "miss_interval": _wilson(misses, n_replicates),
    }


def queue_volume_monitor(
    n_replicates: int = 300,
    window: int = 8,
    capacity: float = 100.0,
    slack: float = 1.10,
    overload_factor: float = 1.60,
    seed: int = 0,
) -> dict[str, Any]:
    """Detect an overloaded review queue.

    H0: windowed mean queue volume <= capacity * slack.  H1: volume rises to
    ``overload_factor`` times capacity.  Trigger when a full window's mean
    exceeds the threshold.
    """

    rng = np.random.default_rng(seed + 101)
    threshold = capacity * slack

    def run(rate: float) -> int:
        series = rng.poisson(rate, size=window)
        return int(series.mean() > threshold)

    false_alarms = sum(run(capacity) for _ in range(n_replicates))
    misses = sum(1 - run(capacity * overload_factor) for _ in range(n_replicates))
    return {
        "monitor": "queue_volume_overload",
        "hypothesis": f"windowed mean volume <= {threshold:.1f}",
        "trigger_scenario": f"volume rises to {overload_factor:.2f}x capacity over a {window}-observation window",
        "window": int(window),
        "threshold": float(threshold),
        "replicates": int(n_replicates),
        "false_alarm_rate": float(false_alarms / n_replicates),
        "false_alarm_interval": _wilson(false_alarms, n_replicates),
        "miss_rate": float(misses / n_replicates),
        "miss_interval": _wilson(misses, n_replicates),
    }


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    z = 1.96
    denom = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
