"""the accepts-trained prior, Algorithm 1 and its corruption sweep.

Every number is measured on this run's own synthetic populations.

* fits the deterministic accepts-only scorecard and judges its prior on
  the REJECTED region: the reliability points compare the predicted reject
  default probability with the latent reject rate (Wilson intervals), and the
  overlap panel shows that the accept/reject clouds live in different parts of
  the standardised feature space.
* extends Algorithm 1's own pseudo-label draws far past its 60-draw
  budget: the running mean of the pseudo-label default rate is plotted per trial
  against the latent reject rate, and its per-draw step is plotted on a log-log
  axis against the 1e-6 stopping tolerance. The draw sequence uses exactly
  Algorithm 1's seeding rule (``default_rng([seed, 0])``).
* uses the paired study (``run_reject_inference_study``)
  for the per-trial absolute errors of the accepts-only and Bayesian arms and
  for the prior-flip sweep means, and recomputes the per-trial sweep errors with
  the study's own ranking-hoisted Algorithm-1 step so each sweep point carries a
  trial bootstrap band. The recomputed means are checked against the study's
  aggregates before anything is written.

The BASL arm is not drawn here: the canonical
early-stop selects iteration 0 on the measured streams, so the arm would plot on
top of accepts-only.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from label_delay.config import seed_of
from label_delay.evaluation import (
    _study_bayesian_metric_fast,
    _study_corrupt_prior,
    _study_fit_scorecard,
    _study_scorecard_predict,
    generate_reject_inference_stream,
    reject_prior,
    run_reject_inference_evaluation,
    run_reject_inference_study,
)
from label_delay.vizlib import ROOT, save_study_figure, write_study_sidecar

NAME = "alg1_prior_corruption"
CAPTION_LINES = [
    "the left panels judge the accepts-only prior on the rejected region (predicted against latent default rate, and the accepted-versus-rejected feature clouds); the middle panels show Algorithm 1's running pseudo-label mean settling against the latent reject rate and its per-draw step against the stopping tolerance; the right panels show the paired per-trial error of the accepts-only and Bayesian arms and the prior-corruption sweep with its crossover.",
    'Where the prior is informative the Bayesian arm wins; as the prior is corrupted the advantage shrinks and then reverses at the dotted crossover. Everything here is synthetic.',
]

BLUE, VERM, GREEN, YELLOW, PURPLE, SKY, GREY = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#666666",
)
BLUES4 = ["#9ECAE1", "#4292C6", "#08519C", "#08306B"]
INK, GRID = "#222222", "#E4E4E4"

TOLERANCE = 1e-6
RELIABILITY_BINS = 8
OVERLAP_SAMPLE = 120
TRIALS = 4
DRAWS = 150_000
HEAD_DRAWS = 100
SAMPLES = 220
SWEEP_FRACTIONS = (0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0)
METRIC_LABELS = (("abr", "ABR"), ("auc", "AUC"), ("brier", "Brier"))
INTERVAL_LEVEL = 0.95

_CACHE: dict[str, dict[str, Any]] = {}


# ------------------------------------------------------------------ helpers
def _wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial fraction."""
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _bootstrap_mean(values: Any, level: float, n_boot: int, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap interval of the mean; interval always covers the mean."""
    array = np.asarray(values, dtype=float).reshape(-1)
    mean = float(array.mean()) if array.size else 0.0
    if array.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng([int(seed), 71])
    draws = rng.integers(0, array.size, size=(max(int(n_boot), 1), array.size))
    means = array[draws].mean(axis=1)
    tail = 0.5 * (1.0 - float(level))
    low = float(np.percentile(means, 100.0 * tail))
    high = float(np.percentile(means, 100.0 * (1.0 - tail)))
    return mean, min(low, mean), max(high, mean)


def _trial_seeds(seed: int, count: int) -> list[int]:
    """Algorithm 1's study seeds this run: one distinct stream seed per trial."""
    rng = np.random.default_rng([int(seed), 20260923])
    return [int(value) for value in rng.integers(0, 2**31 - 1, size=max(int(count), 0))]


def _stride_seeds(seed: int, count: int) -> list[int]:
    """Deterministic derived seeds for the post panels (one per trial)."""
    return [int(seed) + 1_000_003 * (index + 1) for index in range(max(int(count), 0))]


def _caption_lines() -> list[str]:
    return list(CAPTION_LINES)


# ------------------------------------------------------------------ panels
def _reliability_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = seed_of(config)
    stream = generate_reject_inference_stream(config)
    evaluation = run_reject_inference_evaluation(stream, config)
    scores = np.asarray(evaluation["scores_reject"], dtype=float)
    latent = np.asarray(stream["y_reject_latent"], dtype=float)
    edges = np.quantile(scores, np.linspace(0.0, 1.0, RELIABILITY_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, RELIABILITY_BINS - 1)

    rows: list[dict[str, Any]] = []
    reliability_rows = 0
    for index in range(RELIABILITY_BINS):
        mask = bins == index
        count = int(mask.sum())
        if count == 0:
            continue
        successes = int(latent[mask].sum())
        low, high = _wilson(successes, count)
        rows.append(
            {
                "panel": "reliability",
                "series": "reliability",
                "x": float(scores[mask].mean()),
                "y": float(successes / count),
                "y_low": low,
                "y_high": high,
                "n": count,
                "label": f"{count} rejects",
            }
        )
        reliability_rows += 1
    rows.append({"panel": "reliability", "series": "diagonal", "x": 0.0, "y": 0.0})
    rows.append({"panel": "reliability", "series": "diagonal", "x": 1.0, "y": 1.0})

    accept = np.asarray(stream["X_accept"], dtype=float)
    reject = np.asarray(stream["X_reject"], dtype=float)
    rng = np.random.default_rng([int(seed), 73])
    for name, matrix, colour_free in (("accepted_sample", accept, True), ("rejected_sample", reject, False)):
        take = min(int(OVERLAP_SAMPLE), matrix.shape[0])
        index = rng.choice(matrix.shape[0], size=take, replace=False)
        for row in matrix[index]:
            rows.append(
                {
                    "panel": "overlap",
                    "series": name,
                    "x": float(row[0]),
                    "y": float(row[1]),
                }
            )
    summary = {
        "n_accept": int(accept.shape[0]),
        "n_reject": int(reject.shape[0]),
        "prior_mean": float(np.mean(evaluation["prior_clean"])),
        "reject_latent_rate": float(latent.mean()),
        "reliability_bins": reliability_rows,
        "scores_accept_mean": float(np.mean(evaluation["scores_accept"])),
    }
    return rows, summary


def _running_mean_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = seed_of(config)
    trials = _trial_seeds(seed, TRIALS)
    rows: list[dict[str, Any]] = []
    truth_rates: list[float] = []
    settling: list[float] = []
    first_crossings: list[int | None] = []
    last_crossings: list[int | None] = []
    for trial, stream_seed in enumerate(trials):
        stream = generate_reject_inference_stream({**config, "seed": stream_seed})
        prior = np.asarray(reject_prior(stream), dtype=float)
        n_reject = int(prior.size)
        truth_rates.append(float(np.mean(np.asarray(stream["y_reject_latent"], dtype=float))))
        rng = np.random.default_rng([int(stream_seed), 0])
        block = 5_000
        counts = []
        for start in range(0, DRAWS, block):
            size = min(block, DRAWS - start)
            counts.append((rng.random((size, n_reject)) < prior).sum(axis=1))
        cumulative = np.cumsum(np.concatenate(counts))
        draws = np.arange(1, DRAWS + 1)
        running = cumulative / (draws * n_reject)
        previous = np.concatenate([[0.0], running[:-1]])
        change = np.abs(running - previous)
        below = np.flatnonzero(change <= TOLERANCE)
        above = np.flatnonzero(change > TOLERANCE)
        first_crossings.append(int(below[0]) + 1 if below.size else None)
        settle = int(above[-1]) + 2 if above.size else 1
        last_crossings.append(settle if settle <= DRAWS else None)
        settling.append(float(running[-1]))

        tail = np.geomspace(HEAD_DRAWS + 1, DRAWS, max(SAMPLES - HEAD_DRAWS, 1))
        markers = {value for value in (first_crossings[-1], last_crossings[-1]) if value is not None}
        indices = sorted(
            {index for index in range(1, HEAD_DRAWS + 1)}
            | {int(round(value)) for value in tail}
            | markers
        )
        group = f"trial_{trial + 1}"
        for index in indices:
            rows.append(
                {
                    "panel": "running_mean",
                    "series": "running_mean",
                    "group": group,
                    "x": float(index),
                    "y": float(running[index - 1]),
                    "n": n_reject,
                }
            )
            rows.append(
                {
                    "panel": "change_vs_tolerance",
                    "series": "running_change",
                    "group": group,
                    "x": float(index),
                    "y": float(change[index - 1]),
                    "n": n_reject,
                }
            )
    truth = float(np.mean(truth_rates))
    for edge in (1.0, float(DRAWS)):
        rows.append({"panel": "running_mean", "series": "latent_truth_rate", "x": edge, "y": truth, "n": TRIALS})
        rows.append(
            {
                "panel": "change_vs_tolerance",
                "series": "tolerance",
                "x": edge,
                "y": TOLERANCE,
                "n": TRIALS,
            }
        )
    summary = {
        "trials": TRIALS,
        "draws": DRAWS,
        "prior_means": [float(s) for s in settling],
        "truth_rate": truth,
        "first_crossings": first_crossings,
        "last_crossings": last_crossings,
    }
    return rows, summary


def _study(config: Mapping[str, Any]) -> dict[str, Any]:
    key = str(seed_of(config)) + str(config.get("n_trials")) + str(config.get("n_boot"))
    if key not in _CACHE:
        _CACHE[key] = run_reject_inference_study(
            {
                **dict(config),
                "n_trials": int(config.get("n_trials", 50)),
                "corruption_fractions": list(SWEEP_FRACTIONS),
                "n_boot": int(config.get("n_boot", 500)),
            }
        )
    return _CACHE[key]


def _sweep_per_trial(
    config: Mapping[str, Any], study: Mapping[str, Any]
) -> dict[str, dict[float, list[float]]]:
    """Per-trial sweep errors on the study's own streams, via the study's fast step.

    Fraction 0.0 is copied from the study's per-trial errors; every other
    fraction reuses the same stream seed, scorecard fit and corruption rule, so
    the aggregates are checked against ``prior_corruption_sweep`` below.
    """
    seed = int(config.get("seed", 1))
    tolerance = float(config.get("tolerance", TOLERANCE))
    min_draws = int(config.get("min_draws", 20))
    max_draws = int(config.get("max_draws", 60))
    errors: dict[str, dict[float, list[float]]] = {
        metric: {fraction: [] for fraction in SWEEP_FRACTIONS} for metric, _ in METRIC_LABELS
    }
    for trial in study["trials"]:
        stream_seed = int(trial["stream_seed"])
        stream = generate_reject_inference_stream({**config, "seed": stream_seed})
        y_accept = np.asarray(stream["y_accept"], dtype=float)
        model, constant = _study_fit_scorecard(stream["X_accept"], y_accept)
        scores_accept = np.clip(_study_scorecard_predict(model, constant, stream["X_accept"]), 0.0, 1.0)
        scores_reject = np.clip(_study_scorecard_predict(model, constant, stream["X_reject"]), 0.0, 1.0)
        prior_clean = reject_prior(stream)
        truth = trial["truth"]
        for metric, _ in METRIC_LABELS:
            errors[metric][0.0].append(float(trial["bayesian_abs_error"][metric]))
        for fraction in SWEEP_FRACTIONS:
            if fraction == 0.0:
                continue
            prior = _study_corrupt_prior(prior_clean, fraction, stream_seed)
            for metric, _ in METRIC_LABELS:
                value = float(
                    _study_bayesian_metric_fast(
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
                errors[metric][fraction].append(abs(value - float(truth[metric])))
    return errors


def _d9_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(config.get("seed", 1))
    n_boot = int(config.get("n_boot", 500))
    study = _study(config)
    n_trials = int(study["n_trials"])
    rows: list[dict[str, Any]] = []

    for series, key, index in (("accepts_only", "accepts_only_abs_error", 0), ("bayesian", "bayesian_abs_error", 1)):
        for metric, label in METRIC_LABELS:
            values = np.asarray([trial[key][metric] for trial in study["trials"]], dtype=float)
            mean, low, high = _bootstrap_mean(values, INTERVAL_LEVEL, n_boot, seed)
            rows.append(
                {
                    "panel": "paired_error",
                    "series": series,
                    "group": label,
                    "x": float(index),
                    "y": mean,
                    "y_low": low,
                    "y_high": high,
                    "n": n_trials,
                }
            )

    sweep = _sweep_per_trial(config, study)
    aggregates = {
        (row["flip_fraction"], metric): (row["mean_bayesian_abs_error"][metric], row["mean_accepts_only_abs_error"][metric])
        for row in study["prior_corruption_sweep"]
        for metric, _ in METRIC_LABELS
    }
    discrepancy = 0.0
    for fraction in SWEEP_FRACTIONS:
        for metric, label in METRIC_LABELS:
            bayesian_values = np.asarray(sweep[metric][fraction], dtype=float)
            accepts_values = np.asarray([trial["accepts_only_abs_error"][metric] for trial in study["trials"]], dtype=float)
            bayesian_mean, bayesian_low, bayesian_high = _bootstrap_mean(
                bayesian_values, INTERVAL_LEVEL, n_boot, seed
            )
            accepts_mean, accepts_low, accepts_high = _bootstrap_mean(
                accepts_values, INTERVAL_LEVEL, n_boot, seed
            )
            discrepancy = max(
                discrepancy,
                abs(bayesian_mean - aggregates[(float(fraction), metric)][0]),
                abs(accepts_mean - aggregates[(float(fraction), metric)][1]),
            )
            rows.append(
                {
                    "panel": "corruption_sweep",
                    "series": "accepts_only",
                    "group": label,
                    "x": float(fraction),
                    "y": accepts_mean,
                    "y_low": accepts_low,
                    "y_high": accepts_high,
                    "n": n_trials,
                }
            )
            rows.append(
                {
                    "panel": "corruption_sweep",
                    "series": "bayesian",
                    "group": label,
                    "x": float(fraction),
                    "y": bayesian_mean,
                    "y_low": bayesian_low,
                    "y_high": bayesian_high,
                    "n": n_trials,
                }
            )
    if discrepancy > 1e-9:
        raise AssertionError(
            f"{NAME}: local sweep means disagree with the study by {discrepancy:.3e}"
        )

    summary = {
        "n_trials": n_trials,
        "paired": {
            metric: {
                "accepts_only": float(study["paired"][metric]["mean_accepts_only_abs_error"]),
                "bayesian": float(study["paired"][metric]["mean_bayesian_abs_error"]),
                "difference": float(study["paired"][metric]["mean_paired_difference"]),
                "low": float(study["paired"][metric]["ci_low"]),
                "high": float(study["paired"][metric]["ci_high"]),
                "direction": study["paired"][metric]["direction"],
            }
            for metric, _ in METRIC_LABELS
        },
        "crossover": {metric: study["crossover"][metric] for metric, _ in METRIC_LABELS},
        "discrepancy": discrepancy,
    }
    return rows, summary


# ------------------------------------------------------------------ figure
def _style(ax: Any, grid: str = "y") -> None:
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _head(ax: Any, text: str, handles: list[Any] | None = None, ncol: int = 4, pad: int | None = None) -> None:
    if handles and pad is None:
        rows = max(1, math.ceil(len(handles) / max(ncol, 1)))
        pad = 12 + 17 * rows
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", pad=(pad or 8) if handles else 8)
    if handles:
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            ncol=ncol,
            frameon=False,
            fontsize=8.5,
            borderaxespad=0.1,
            handlelength=1.8,
            columnspacing=1.1,
            handletextpad=0.5,
        )


def _line(name: str, color: str, ls: str = "-", lw: float = 1.5) -> Line2D:
    return Line2D([0], [0], color=color, lw=lw, ls=ls, label=name)


def _figure(rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> Any:
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault((row["panel"], row["series"]), []).append(row)

    def series(panel: str, name: str) -> list[dict[str, Any]]:
        return by.get((panel, name), [])

    width, height = 24.0, 10.6
    fig = plt.figure(figsize=(width, height), layout="constrained")
    shell = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.17])
    caption = fig.add_subplot(shell[1])
    caption.axis("off")
    caption.text(
        0.0,
        1.0,
        "\n".join(
            textwrap.fill(("How to read this chart: " if index == 0 else "") + line, width=250)
            for index, line in enumerate(_caption_lines())
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        transform=caption.transAxes,
    )
    top = shell[0].subgridspec(
        2, 5, width_ratios=[0.98, 0.82, 0.78, 0.78, 0.78], hspace=0.4, wspace=0.28
    )
    left = top[:, 0].subgridspec(2, 1, hspace=0.42)
    middle = top[:, 1].subgridspec(2, 1, hspace=0.42)

    # --- column 1 ------------------------------------------------------
    ax_rel = fig.add_subplot(left[0])
    reliable = sorted(series("reliability", "reliability"), key=lambda r: r["x"])
    diagonal = sorted(series("reliability", "diagonal"), key=lambda r: r["x"])
    ax_rel.plot([r["x"] for r in diagonal], [r["y"] for r in diagonal], color=GREY, ls="--", lw=1.4)
    ax_rel.errorbar(
        [r["x"] for r in reliable],
        [r["y"] for r in reliable],
        yerr=[
            [r["y"] - r["y_low"] for r in reliable],
            [r["y_high"] - r["y"] for r in reliable],
        ],
        fmt="o",
        color=BLUE,
        ms=5,
        lw=1.4,
        capsize=3,
        label="reject bin",
    )
    ax_rel.set(xlabel="mean scorecard prior in bin", ylabel="latent default rate of rejects", xlim=(0, 1), ylim=(0, 1))
    _style(ax_rel, "both")
    _head(
        ax_rel,
        "prior vs latent truth on rejects",
        [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, ms=6, label="reject bin"),
            _line("perfect prior", GREY, "--", 1.4),
        ],
        2,
    )

    ax_ov = fig.add_subplot(left[1])
    for name, colour, label in (
        ("accepted_sample", BLUE, "accepted"),
        ("rejected_sample", VERM, "rejected"),
    ):
        data = series("overlap", name)
        ax_ov.scatter([r["x"] for r in data], [r["y"] for r in data], s=11, alpha=0.45, color=colour, linewidths=0)
    ax_ov.set(xlabel="standardised feature 1", ylabel="standardised feature 2")
    _style(ax_ov, "both")
    _head(
        ax_ov,
        "accepted vs rejected feature clouds",
        [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, ms=6, label="accepted"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=VERM, ms=6, label="rejected"),
        ],
        2,
    )

    # --- column 2 ------------------------------------------------------
    ax_mean = fig.add_subplot(middle[0])
    trials = sorted({r["group"] for r in series("running_mean", "running_mean")})
    handles = []
    for index, group in enumerate(trials):
        data = sorted((r for r in series("running_mean", "running_mean") if r["group"] == group), key=lambda r: r["x"])
        ax_mean.plot([r["x"] for r in data], [r["y"] for r in data], color=BLUES4[index % len(BLUES4)], lw=1.4)
        handles.append(_line(group.replace("_", " "), BLUES4[index % len(BLUES4)], lw=1.4))
    truth = series("running_mean", "latent_truth_rate")
    ax_mean.axhline(truth[0]["y"], color=GREY, ls="--", lw=1.4)
    handles.append(_line(f"latent reject rate {truth[0]['y']:.3f}", GREY, "--", 1.4))
    ax_mean.set(xscale="log", xlabel="pseudo-label draws", ylabel="running mean of pseudo-label rate", ylim=(0.0, 1.0))
    _style(ax_mean, "both")
    _head(ax_mean, "Algorithm-1 running mean per trial", handles, 3, pad=38)

    ax_change = fig.add_subplot(middle[1])
    settles = summary["last_crossings"]
    for index, group in enumerate(trials):
        data = sorted(
            (r for r in series("change_vs_tolerance", "running_change") if r["group"] == group), key=lambda r: r["x"]
        )
        colour = BLUES4[index % len(BLUES4)]
        ax_change.plot([r["x"] for r in data], [r["y"] for r in data], color=colour, lw=1.4)
        if settles[index] is not None:
            match = [r for r in data if r["x"] == float(settles[index])]
            if match:
                ax_change.plot(
                    [float(settles[index])],
                    [float(match[0]["y"])],
                    marker="o",
                    ms=5,
                    color=colour,
                    mec="white",
                    mew=0.8,
                    zorder=5,
                )
    tolerance = series("change_vs_tolerance", "tolerance")
    ax_change.axhline(tolerance[0]["y"], color=GREY, ls="--", lw=1.4)
    ax_change.set(
        xscale="log",
        yscale="log",
        xlabel="pseudo-label draws",
        ylabel="abs per-draw change of the running mean",
    )
    _style(ax_change, "both")
    _head(
        ax_change,
        "running-mean step vs tolerance",
        [
            _line("1e-6 stopping tolerance", GREY, "--", 1.4),
            *handles[:-1],
            Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, mec="white", ms=6, label="step settled"),
        ],
        3,
        pad=38,
    )

    # --- column 3 ------------------------------------------------------
    paired = [fig.add_subplot(top[0, column]) for column in (2, 3, 4)]
    for axis, (metric, label) in zip(paired, METRIC_LABELS):
        for series_name, colour in (("accepts_only", VERM), ("bayesian", BLUE)):
            data = [r for r in series("paired_error", series_name) if r["group"] == label]
            if not data:
                continue
            axis.errorbar(
                [r["x"] for r in data],
                [r["y"] for r in data],
                yerr=[
                    [r["y"] - r["y_low"] for r in data],
                    [r["y_high"] - r["y"] for r in data],
                ],
                fmt="o",
                color=colour,
                ms=6,
                lw=1.5,
                capsize=4,
            )
        axis.set_xticks([0, 1], ["accepts", "Bayesian"], fontsize=9)
        for tick, colour in zip(axis.get_xticklabels(), (VERM, BLUE)):
            tick.set_color(colour)
        axis.set_xlim(-0.6, 1.6)
        axis.set_title(f"{label}\n", loc="left", fontsize=10, fontweight="bold")
        _style(axis)
        axis.margins(y=0.28)
    paired[0].set_ylabel("mean absolute error vs latent truth")
    _head(
        paired[0],
        f"paired error, {summary['n_trials']} trials\n(vermillion = accepts-only, blue = Bayesian)",
        None,
    )

    sweep_axes = [fig.add_subplot(top[1, column]) for column in (2, 3, 4)]
    for axis, (metric, label) in zip(sweep_axes, METRIC_LABELS):
        ends: dict[str, float] = {}
        for series_name, colour in (("accepts_only", VERM), ("bayesian", BLUE)):
            data = sorted(
                (r for r in series("corruption_sweep", series_name) if r["group"] == label), key=lambda r: r["x"]
            )
            xs = [r["x"] for r in data]
            axis.fill_between(xs, [r["y_low"] for r in data], [r["y_high"] for r in data], color=colour, alpha=0.22, lw=0)
            axis.plot(xs, [r["y"] for r in data], color=colour, lw=1.8, marker="o", ms=3.5)
            ends[series_name] = float(data[-1]["y"])
            crossover = summary["crossover"][metric]
            if series_name == "bayesian" and crossover is not None:
                axis.axvline(crossover, color=colour, ls=":", lw=1.2)
        axis.set(xlabel="prior flip fraction", ylim=(0.0, None))
        axis.set_title(f"{label}\n", loc="left", fontsize=10, fontweight="bold")
        _style(axis)
        axis.margins(y=0.24)
        low_end = min(ends["accepts_only"], ends["bayesian"])
        high_end = max(ends["accepts_only"], ends["bayesian"])
        lowest = "bayesian" if ends["bayesian"] <= ends["accepts_only"] else "accepts_only"
        highest = "accepts_only" if lowest == "bayesian" else "bayesian"
        colours = {"accepts_only": VERM, "bayesian": BLUE}
        names = {"accepts_only": "accepts-only", "bayesian": "Bayesian"}
        span = float(axis.get_ylim()[1] - axis.get_ylim()[0])
        gap = 0.06 * span
        axis.text(
            0.97,
            low_end - gap,
            names[lowest],
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="top",
            fontsize=8.5,
            color=colours[lowest],
        )
        axis.text(
            0.97,
            high_end + gap,
            names[highest],
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=colours[highest],
        )
    sweep_axes[0].set_ylabel("absolute error vs latent truth")
    _head(
        sweep_axes[0],
        "corruption sweep (dotted = crossover)",
        None,
    )
    return fig


# ------------------------------------------------------------------ entry
def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the alg1_prior_corruption sidecar and figure."""
    d7_rows, d7_summary = _reliability_rows(config)
    d8_rows, d8_summary = _running_mean_rows(config)
    d9_rows, d9_summary = _d9_rows(config)
    rows = [*d7_rows, *d8_rows, *d9_rows]
    figure_summary = {**d7_summary, **d8_summary, **d9_summary}

    sidecar = write_study_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = save_study_figure(_figure(rows, figure_summary), NAME, out_dir=out_dir)
    plt.close()
    lines = _caption_lines()
    print(f"{NAME}: {len(rows)} rows -> {sidecar.name} and {figure.name}")
    print("How to read this chart: " + lines[0])
    print(lines[1])
    print(
        "accepts={n_accept} rejects={n_reject} prior mean on rejects={prior_mean:.4f} "
        "latent reject rate={reject_latent_rate:.4f} bins={reliability_bins}".format(**d7_summary)
    )
    print(
        "trials={trials} draws={draws} settled running mean={prior_means} latent reject rate={truth_rate:.4f} "
        "first step<=1e-6 at {first_crossings} settled at {last_crossings}".format(
            **{**d8_summary, "prior_means": [round(v, 4) for v in d8_summary["prior_means"]]}
        )
    )
    for metric, label in METRIC_LABELS:
        entry = d9_summary["paired"][metric]
        print(
            f"{label}: accepts-only={entry['accepts_only']:.4f} bayesian={entry['bayesian']:.4f} "
            f"paired diff={entry['difference']:+.4f} [{entry['low']:+.4f}, {entry['high']:+.4f}] "
            f"({entry['direction']}); crossover={d9_summary['crossover'][metric]}"
        )
    print(f"sweep means agree with the study to {d9_summary['discrepancy']:.2e}")
    return {"figure": str(figure), "sidecar": str(sidecar), "rows": int(len(rows)), "name": NAME}
