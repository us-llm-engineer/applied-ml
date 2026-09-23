"""E4--E6 of NB2: naive training versus delay-, selection- and reject-aware training.

Exec-side renderer for ``viz/spec/R2_NB2_naive_vs_aware.md`` (claim C19). Every
number is measured on this run's own shared fraud streams; no mock value is read
or reused.

* E5 (post) precision/recall at the 1% FPR comes from the frozen C19 experiment
  (:func:`exec.policies.run_naive_vs_aware_experiment`, ``n_seeds`` streams):
  each dot is the mean over seeds and each bar the t interval over seeds.
* E4 (mid), the E5 cumulative-gain curves, the oracle-label reference and E6
  (post) need per-seed scores and per-age refits the experiment does not expose,
  so a second pass over the same per-seed streams re-uses the experiment's own
  trainer (``exec.policies._r2_logistic_scores`` and the other helpers the C19
  experiment calls) and the same per-seed derivation (``R2_SEED_STRIDE``). The
  two passes are cross-checked row by row: the recomputed per-seed
  precision/recall must equal the experiment's rows for the same seeds, or the
  build fails.
* The oracle-label reference is a logistic fit on the latent ``Y`` of the
  evaluated population with the same trainer; it is truth-adjacent by
  construction and drawn as a grey reference, never as a deployable arm.
* E6 trains at six label ages: at age ``a`` a label counts as observed only when
  ``label_delay <= a``, the naive model treats every unresolved approved row as
  a negative, and the aware model re-estimates ``P(label_observed | x, score)``
  at that age and reweights with its inverse. E6 runs on the first
  ``AGE_SEEDS`` seeds to bound the refit cost; the panels report their own n.
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
from scipy import stats as _stats

from exec.config import seed_of
from exec.data import generate_realistic_stream
from exec.evaluation import precision_recall_at_fpr
from exec.policies import (
    R2_SEED_STRIDE,
    _r2_ipw_weights,
    _r2_logistic_scores,
    _r2_propensity_scores,
    _r2_stream_arrays,
    run_naive_vs_aware_experiment,
)
from exec.vizlib import ROOT, save_r2_figure, write_r2_sidecar

NAME = "R2_NB2_naive_vs_aware"
SPEC_PATH = ROOT / "viz" / "spec" / f"{NAME}.md"

BLUE, VERM, GREEN, YELLOW, PURPLE, SKY, GREY = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#666666",
)
INK, GRID = "#222222", "#E4E4E4"

METHODS = ("naive", "aware")
ARMS = ("naive", "aware", "oracle_label_reference")
RELIABILITY_BINS = 10
GAIN_FRACTIONS = np.geomspace(1e-3, 1.0, 25)
LABEL_AGES = (0, 5, 10, 20, 40, 60)
AGE_SEEDS = 12
INTERVAL_LEVEL = 0.95
METRICS = ("precision_at_1pct_fpr", "recall_at_1pct_fpr")
METRIC_KEYS = {"precision_at_1pct_fpr": "precision", "recall_at_1pct_fpr": "recall"}
EXPERIMENT_KEYS = {"precision_at_1pct_fpr": "precision_at_fpr", "recall_at_1pct_fpr": "recall_at_fpr"}

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


def _mean_interval(values: Any, level: float = INTERVAL_LEVEL) -> tuple[float, float, float]:
    """Mean and the two-sided t interval over seeds; degenerate-safe."""
    array = np.asarray(values, dtype=float).reshape(-1)
    n = int(array.size)
    mean = float(array.mean()) if n else 0.0
    if n < 2:
        return mean, mean, mean
    half = float(_stats.t.ppf(0.5 + level / 2.0, n - 1)) * float(array.std(ddof=1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def _column_intervals(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column mean and t interval of a seeds x points matrix, clipped to [0, 1]."""
    mean = curves.mean(axis=0)
    if curves.shape[0] < 2:
        return mean, mean, mean
    half = float(_stats.t.ppf(0.5 + INTERVAL_LEVEL / 2.0, curves.shape[0] - 1)) * curves.std(
        axis=0, ddof=1
    ) / math.sqrt(curves.shape[0])
    return mean, np.clip(mean - half, 0.0, 1.0), np.clip(mean + half, 0.0, 1.0)


def _cumulative_gain(labels: np.ndarray, scores: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Fraction of the positives caught when the top ``f`` share is reviewed."""
    order = np.argsort(-scores, kind="stable")
    positives = np.cumsum(labels[order] == 1.0)
    total = float(positives[-1]) if positives.size else 0.0
    if total <= 0.0:
        return np.zeros(fractions.shape, dtype=float)
    positions = np.minimum(np.maximum(1, np.rint(fractions * labels.size).astype(int)), labels.size)
    return positives[positions - 1] / total


def _stream_config(config: Mapping[str, Any], index: int) -> dict[str, Any]:
    """The experiment's per-seed stream derivation (same stride, same size)."""
    return {
        "seed": int(config["seed"]) + R2_SEED_STRIDE * (index + 1),
        "stream_n": int(config["stream_n"]),
        "stream_t_drift": int(config["stream_t_drift"]),
    }


def _caption_lines() -> list[str]:
    return [
        line[2:].strip()
        for line in SPEC_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("> ")
    ]


# ------------------------------------------------------------------ data
def _experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    key = "nva:" + str(seed_of(config)) + ":" + str(config.get("n_seeds"))
    if key not in _CACHE:
        # `exec.policies._R2_CACHE` is keyed by the config alone and is shared by
        # both R2 experiments, so the canonical config would collide with the
        # replicated-policy experiment when both run in one process (the
        # renderer does). The isolation tag changes only that cache key; every
        # experiment reads its sizes from the config it is handed and the tag is
        # ignored by the experiment (and never reaches the sidecar stamp).
        _CACHE[key] = run_naive_vs_aware_experiment({**dict(config), "_r2_figure": NAME})
    return _CACHE[key]


def _score_pass(config: Mapping[str, Any]) -> dict[str, Any]:
    """One pass over all C19 streams: per-seed scores, gains, oracle and age refits."""
    key = "scores:" + str(seed_of(config)) + ":" + str(config.get("stream_n"))
    if key in _CACHE:
        return _CACHE[key]
    n_seeds = int(config.get("n_seeds", 20))
    fpr = float(config.get("fpr_target", 0.01))
    pools = {arm: {"scores": [], "labels": []} for arm in ARMS}
    gains = {arm: [] for arm in ARMS}
    age_curves = {method: {age: [] for age in LABEL_AGES} for method in METHODS}
    oracle_metrics: dict[str, list[float]] = {metric: [] for metric in METRICS}
    per_seed: dict[tuple[str, int], dict[str, float]] = {}
    for index in range(n_seeds):
        stream_config = _stream_config(config, index)
        stream = generate_realistic_stream(stream_config)
        data = _r2_stream_arrays(stream)
        features, labels = data["X"], data["y"]
        approved, observed, model_score = data["approved"], data["observed"], data["model_score"]
        naive = np.asarray(
            _r2_logistic_scores(features[approved], labels[approved] * observed[approved], None, features),
            dtype=float,
        )
        propensity = _r2_propensity_scores(features, model_score, observed)
        aware = np.asarray(
            _r2_logistic_scores(
                features[observed], labels[observed], _r2_ipw_weights(propensity, observed), features
            ),
            dtype=float,
        )
        oracle = np.asarray(_r2_logistic_scores(features, labels, None, features), dtype=float)
        seed = int(stream_config["seed"])
        for arm, scores in (("naive", naive), ("aware", aware), ("oracle_label_reference", oracle)):
            pools[arm]["scores"].append(scores)
            pools[arm]["labels"].append(labels)
            gains[arm].append(_cumulative_gain(labels, scores, GAIN_FRACTIONS))
        for method, scores in (("naive", naive), ("aware", aware)):
            metrics = precision_recall_at_fpr(labels, scores, fpr)
            per_seed[(method, seed)] = {
                name: float(metrics[METRIC_KEYS[name]]) for name in METRICS
            }
        oracle_result = precision_recall_at_fpr(labels, oracle, fpr)
        for metric in METRICS:
            oracle_metrics[metric].append(float(oracle_result[METRIC_KEYS[metric]]))

        if index < AGE_SEEDS:
            delays = np.asarray([t["label_delay"] for t in stream["transactions"]], dtype=float)
            for age in LABEL_AGES:
                observed_age = observed & (delays <= float(age))
                naive_age = _r2_logistic_scores(
                    features[approved], labels[approved] * observed_age[approved], None, features
                )
                propensity_age = _r2_propensity_scores(features, model_score, observed_age)
                aware_age = _r2_logistic_scores(
                    features[observed_age],
                    labels[observed_age],
                    _r2_ipw_weights(propensity_age, observed_age),
                    features,
                )
                age_curves["naive"][age].append(
                    float(precision_recall_at_fpr(labels, naive_age, fpr)["recall"])
                )
                age_curves["aware"][age].append(
                    float(precision_recall_at_fpr(labels, aware_age, fpr)["recall"])
                )
    payload = {
        "pools": pools,
        "gains": gains,
        "age_curves": age_curves,
        "oracle_metrics": oracle_metrics,
        "per_seed": per_seed,
        "seeds": [int(_stream_config(config, index)["seed"]) for index in range(n_seeds)],
        "age_seeds": min(AGE_SEEDS, n_seeds),
    }
    _CACHE[key] = payload
    return payload


# ------------------------------------------------------------------ rows
def _reliability_rows(pass_data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    extremes: list[float] = []
    for method in METHODS:
        scores = np.concatenate(pass_data["pools"][method]["scores"])
        labels = np.concatenate(pass_data["pools"][method]["labels"])
        edges = np.quantile(scores, np.linspace(0.0, 1.0, RELIABILITY_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        bins = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, RELIABILITY_BINS - 1)
        summary[f"{method}_instances"] = int(scores.size)
        for index in range(RELIABILITY_BINS):
            mask = bins == index
            count = int(mask.sum())
            if count == 0:
                continue
            successes = int(labels[mask].sum())
            low, high = _wilson(successes, count)
            rows.append(
                {
                    "panel": "E4_reliability",
                    "series": method,
                    "x": float(scores[mask].mean()),
                    "y": float(successes / count),
                    "y_low": low,
                    "y_high": high,
                    "n": count,
                    "label": f"{count} instances",
                }
            )
        extremes.extend([float(scores.min()), float(scores.max())])
    low, high = max(min(extremes), 1e-6), max(extremes)
    rows.append({"panel": "E4_reliability", "series": "perfect_diagonal", "x": low, "y": low})
    rows.append({"panel": "E4_reliability", "series": "perfect_diagonal", "x": high, "y": high})
    return rows, summary


def _pr_rows(
    experiment: Mapping[str, Any], pass_data: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_seed = {(row["method"], int(row["seed"])): row for row in experiment["rows"]}
    per_seed = pass_data["per_seed"]
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"n_seeds": int(experiment["n_seeds"]), "oracle_seeds": len(pass_data["seeds"])}
    for metric, axis_index in zip(METRICS, (0, 1)):
        for arm_index, method in enumerate(METHODS):
            values = [
                float(row[EXPERIMENT_KEYS[metric]])
                for (name, _), row in by_seed.items()
                if name == method
            ]
            mean, low, high = _mean_interval(values)
            rows.append(
                {
                    "panel": "E5_pr_at_fpr",
                    "series": method,
                    "group": metric,
                    "x": float(arm_index),
                    "y": mean,
                    "y_low": low,
                    "y_high": high,
                    "n": len(values),
                }
            )
            summary[f"{method}_{metric}"] = mean
            overlap = [
                abs(float(by_seed[(method, seed)][EXPERIMENT_KEYS[metric]]) - per_seed[(method, seed)][metric])
                for seed in pass_data["seeds"]
                if (method, seed) in by_seed and (method, seed) in per_seed
            ]
            summary[f"{method}_{metric}_overlap"] = float(max(overlap)) if overlap else float("nan")
        mean, low, high = _mean_interval(pass_data["oracle_metrics"][metric])
        rows.append(
            {
                "panel": "E5_pr_at_fpr",
                "series": "oracle_label_reference",
                "group": metric,
                "x": 2.0,
                "y": mean,
                "y_low": low,
                "y_high": high,
                "n": len(pass_data["oracle_metrics"][metric]),
            }
        )
        summary[f"oracle_{metric}"] = mean
    return rows, summary


def _gain_rows(pass_data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for method in ARMS:
        curves = np.vstack(pass_data["gains"][method])
        mean, low, high = _column_intervals(curves)
        for x, y, lo, hi in zip(GAIN_FRACTIONS, mean, low, high):
            rows.append(
                {
                    "panel": "E5_gain",
                    "series": method,
                    "x": float(x),
                    "y": float(y),
                    "y_low": float(lo),
                    "y_high": float(hi),
                    "n": int(curves.shape[0]),
                }
            )
        summary[f"{method}_gain_at_1pct"] = float(mean[0])
        summary[f"{method}_gain_at_full"] = float(mean[-1])
    return rows, summary


def _age_rows(pass_data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    oracle_mean, oracle_low, oracle_high = _mean_interval(pass_data["oracle_metrics"]["recall_at_1pct_fpr"])
    for method in METHODS:
        for age in LABEL_AGES:
            values = pass_data["age_curves"][method][age]
            mean, low, high = _mean_interval(values)
            rows.append(
                {
                    "panel": "E6_metric_vs_age",
                    "series": method,
                    "x": float(age),
                    "y": mean,
                    "y_low": low,
                    "y_high": high,
                    "n": len(values),
                }
            )
        summary[f"{method}_age_0"] = float(np.mean(pass_data["age_curves"][method][LABEL_AGES[0]]))
        summary[f"{method}_age_max"] = float(np.mean(pass_data["age_curves"][method][LABEL_AGES[-1]]))
    for age in LABEL_AGES:
        rows.append(
            {
                "panel": "E6_metric_vs_age",
                "series": "oracle_label_reference",
                "x": float(age),
                "y": oracle_mean,
                "y_low": oracle_low,
                "y_high": oracle_high,
                "n": len(pass_data["oracle_metrics"]["recall_at_1pct_fpr"]),
            }
        )
    summary["oracle_recall"] = oracle_mean
    return rows, summary


# ------------------------------------------------------------------ figure
def _style(ax: Any, grid: str = "y") -> None:
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _head(ax: Any, text: str, handles: list[Any] | None = None, ncol: int = 4) -> None:
    rows = max(1, math.ceil(len(handles) / max(ncol, 1))) if handles else 0
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", pad=(12 + 17 * rows) if handles else 8)
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


def _line(name: str, color: str, ls: str = "-", lw: float = 1.6) -> Line2D:
    return Line2D([0], [0], color=color, lw=lw, ls=ls, label=name)


def _figure(rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> Any:
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault((row["panel"], row["series"]), []).append(row)

    def series(panel: str, name: str) -> list[dict[str, Any]]:
        return by.get((panel, name), [])

    width, height = 22.0, 8.8
    fig = plt.figure(figsize=(width, height), layout="constrained")
    shell = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.17])
    caption = fig.add_subplot(shell[1])
    caption.axis("off")
    caption.text(
        0.0,
        1.0,
        "\n".join(
            textwrap.fill(("How to read this chart: " if index == 0 else "") + line, width=255)
            for index, line in enumerate(_caption_lines())
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        transform=caption.transAxes,
    )
    top = shell[0].subgridspec(2, 3, width_ratios=[1.0, 1.05, 1.0], hspace=0.45, wspace=0.26)

    colours = {"naive": VERM, "aware": BLUE, "oracle_label_reference": GREY}

    # --- column 1: E4 ------------------------------------------------------
    ax_rel = fig.add_subplot(top[:, 0])
    diagonal = sorted(series("E4_reliability", "perfect_diagonal"), key=lambda r: r["x"])
    ax_rel.plot([r["x"] for r in diagonal], [r["y"] for r in diagonal], color=GREY, ls="--", lw=1.4)
    for method in METHODS:
        data = sorted(series("E4_reliability", method), key=lambda r: r["x"])
        ax_rel.errorbar(
            [r["x"] for r in data],
            [r["y"] for r in data],
            yerr=[[r["y"] - r["y_low"] for r in data], [r["y_high"] - r["y"] for r in data]],
            fmt="o",
            ms=5,
            lw=1.4,
            capsize=3,
            color=colours[method],
        )
    ax_rel.set(
        xscale="log",
        yscale="log",
        xlabel="mean predicted fraud probability (log axis)",
        ylabel="latent fraud rate in bin (log axis)",
    )
    reliability = [r for method in METHODS for r in series("E4_reliability", method)]
    if reliability:
        ax_rel.set_ylim(max(min(r["y_low"] for r in reliability) * 0.6, 1e-9), None)
        ax_rel.set_xlim(max(min(r["x"] for r in reliability) * 0.6, 1e-9), None)
    _style(ax_rel, "both")
    _head(
        ax_rel,
        "E4 (mid) reliability on the latent truth",
        [_line("naive", VERM), _line("aware", BLUE), _line("perfect calibration", GREY, "--", 1.4)],
        3,
    )

    # --- column 2: E5 ------------------------------------------------------
    pr_grid = top[0, 1].subgridspec(1, 2, wspace=0.32)
    pr_axes = [fig.add_subplot(pr_grid[index]) for index in (0, 1)]
    titles = {"precision_at_1pct_fpr": "precision at 1% FPR", "recall_at_1pct_fpr": "recall at 1% FPR"}
    for axis, metric in zip(pr_axes, METRICS):
        for arm_index, method in enumerate(ARMS):
            data = [r for r in series("E5_pr_at_fpr", method) if r["group"] == metric]
            if not data:
                continue
            axis.errorbar(
                [arm_index],
                [data[0]["y"]],
                yerr=[[data[0]["y"] - data[0]["y_low"]], [data[0]["y_high"] - data[0]["y"]]],
                fmt="o",
                ms=6,
                lw=1.5,
                capsize=4,
                color=colours[method],
            )
        axis.set_xticks([0, 1, 2], ["naive", "aware", "oracle\nlabels"], fontsize=8.5)
        for tick, method in zip(axis.get_xticklabels(), ARMS):
            tick.set_color(colours[method])
        axis.set_xlim(-0.6, 2.6)
        axis.margins(y=0.18)
        _style(axis)
    pr_axes[0].set_ylabel("metric at 1% FPR")
    pr_axes[0].set_title(f"{titles['precision_at_1pct_fpr']}\n", loc="left", fontsize=10, fontweight="bold")
    pr_axes[1].set_title(f"{titles['recall_at_1pct_fpr']}\n", loc="left", fontsize=10, fontweight="bold")
    _head(pr_axes[0], f"E5 (post) at 1% FPR, {summary['n_seeds']} seeds", None)

    ax_gain = fig.add_subplot(top[1, 1])
    ax_gain.plot([1e-3, 1.0], [1e-3, 1.0], color=GREY, ls="--", lw=1.2)
    for method in ARMS:
        data = sorted(series("E5_gain", method), key=lambda r: r["x"])
        xs = [r["x"] for r in data]
        ax_gain.fill_between(
            xs, [r["y_low"] for r in data], [r["y_high"] for r in data], color=colours[method], alpha=0.18, lw=0
        )
        ax_gain.plot(xs, [r["y"] for r in data], color=colours[method], lw=2.0)
    ax_gain.set(
        xscale="log",
        xlabel="share of the population reviewed, highest scores first (log axis)",
        ylabel="share of fraud caught",
        ylim=(0.0, 1.02),
    )
    _style(ax_gain, "both")
    _head(
        ax_gain,
        "E5 (post) cumulative gain (dashed = random review)",
        [
            _line("naive", VERM, lw=2.0),
            _line("aware", BLUE, lw=2.0),
            _line("oracle labels", GREY, lw=2.0),
            _line("random review", GREY, "--", 1.2),
        ],
        4,
    )

    # --- column 3: E6 ------------------------------------------------------
    ax_age = fig.add_subplot(top[:, 2])
    oracle = sorted(series("E6_metric_vs_age", "oracle_label_reference"), key=lambda r: r["x"])
    ax_age.plot([r["x"] for r in oracle], [r["y"] for r in oracle], color=GREY, ls="--", lw=1.4)
    for method in METHODS:
        data = sorted(series("E6_metric_vs_age", method), key=lambda r: r["x"])
        xs = [r["x"] for r in data]
        ax_age.fill_between(
            xs, [r["y_low"] for r in data], [r["y_high"] for r in data], color=colours[method], alpha=0.2, lw=0
        )
        ax_age.plot(xs, [r["y"] for r in data], color=colours[method], lw=2.2, marker="o", ms=4.5)
    ax_age.set(xlabel="label age at evaluation (days)", ylabel="recall at 1% FPR", ylim=(0.0, None))
    _style(ax_age)
    _head(
        ax_age,
        "E6 (post) quality against label maturity",
        [_line("naive", VERM, lw=2.2), _line("aware", BLUE, lw=2.2), _line("oracle labels", GREY, "--", 1.4)],
        3,
    )
    return fig


# ------------------------------------------------------------------ entry
def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the R2_NB2_naive_vs_aware sidecar and figure."""
    experiment = _experiment(config)
    pass_data = _score_pass(config)

    e4_rows, e4_summary = _reliability_rows(pass_data)
    e5_rows, e5_summary = _pr_rows(experiment, pass_data)
    gain_rows, gain_summary = _gain_rows(pass_data)
    age_rows, age_summary = _age_rows(pass_data)
    rows = [*e4_rows, *e5_rows, *gain_rows, *age_rows]
    summary = {**e4_summary, **e5_summary, **gain_summary, **age_summary}

    failures = [
        abs(value)
        for key, value in summary.items()
        if key.endswith("_overlap") and math.isfinite(float(value))
    ]
    if failures and max(failures) > 1e-12:
        raise AssertionError(
            f"{NAME}: the extra per-seed pass disagrees with the frozen C19 experiment by {max(failures):.3e}"
        )

    sidecar = write_r2_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = save_r2_figure(_figure(rows, summary), NAME, out_dir=out_dir)
    plt.close()
    lines = _caption_lines()
    print(f"{NAME}: {len(rows)} rows -> {sidecar.name} and {figure.name}")
    print("How to read this chart: " + lines[0])
    print(lines[1])
    print(
        "E4: pooled instances naive={naive_instances} aware={aware_instances} over {oracle_seeds} seeds".format(
            **summary
        )
    )
    print(
        "E5: naive precision={naive_precision_at_1pct_fpr:.4f} recall={naive_recall_at_1pct_fpr:.4f}; "
        "aware precision={aware_precision_at_1pct_fpr:.4f} recall={aware_recall_at_1pct_fpr:.4f}; "
        "oracle precision={oracle_precision_at_1pct_fpr:.4f} recall={oracle_recall_at_1pct_fpr:.4f}".format(
            **summary
        )
    )
    paired = experiment["paired_differences"]
    print(
        "E5 paired aware-naive: precision {:.4f} [{:.4f}, {:.4f}], recall {:.4f} [{:.4f}, {:.4f}]".format(
            paired["precision"]["mean"],
            paired["precision"]["low"],
            paired["precision"]["high"],
            paired["recall"]["mean"],
            paired["recall"]["low"],
            paired["recall"]["high"],
        )
    )
    print(
        "E5 gain at 1% reviewed: naive={naive_gain_at_1pct:.4f} aware={aware_gain_at_1pct:.4f} "
        "oracle={oracle_label_reference_gain_at_1pct:.4f}".format(**summary)
    )
    print(
        "E6 recall at 1% FPR: naive {naive_age_0:.4f} -> {naive_age_max:.4f}, aware {aware_age_0:.4f} -> "
        "{aware_age_max:.4f} over {} seeds, oracle {oracle_recall:.4f}".format(
            pass_data["age_seeds"], **summary
        )
    )
    print(
        "E5 overlap check vs the C19 experiment: max |diff| precision {:.2e}, recall {:.2e}".format(
            max(
                abs(summary["naive_precision_at_1pct_fpr_overlap"]),
                abs(summary["aware_precision_at_1pct_fpr_overlap"]),
            ),
            max(
                abs(summary["naive_recall_at_1pct_fpr_overlap"]),
                abs(summary["aware_recall_at_1pct_fpr_overlap"]),
            ),
        )
    )
    return {"figure": str(figure), "sidecar": str(sidecar), "rows": int(len(rows)), "name": NAME}
