"""D4--D6 of NB1: WCTM validity on ready labels and the measured breaks.

Exec-side renderer for ``viz/spec/R2_NB1_wctm_validity.md`` (claims C11--C13).
Every row is measured on this run's own streams; the frozen monitoring engine is
used throughout and no mock value is read or reused.

* D4 (pre) builds the online weighted-conformal p-values with the frozen
  ``weighted_conformal_pvalue`` (Eq. 9 with the uniform tie-breaker) on iid
  ready-label null streams of the C11--C13 null block, histograms them and
  reports the KS distance to uniformity with a bootstrap interval.
* D5 (mid) draws the first ``n_paths`` streams of the frozen null block and of
  the frozen shift experiment (the same seed tags, so the paths ARE those
  experiments' streams) and turns each into a wealth path with the frozen
  ``wctm_wealth_path`` (Eq. 6/7). Alarms are the first steps at or above c.
* D6 (post) reads ``run_wctm_experiment``: the ready-label null, the
  approved-only "selection" and the drift-during-lag variants with their exact
  one-sided Clopper-Pearson bounds, a peeking sabotage measured on the null
  block's own p-values (alarm when any p < alpha), and the delay CDFs per label
  lag from the shift experiment.
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
from scipy import stats

from exec.config import seed_of
from exec.monitoring import (
    run_wctm_experiment,
    wctm_wealth_path,
    weighted_conformal_pvalue,
)
from exec.vizlib import ROOT, save_r2_figure, write_r2_sidecar

NAME = "R2_NB1_wctm_validity"
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
LAG_COLOURS = ["#9ECAE1", "#4292C6", "#08519C"]
INK, GRID = "#222222", "#E4E4E4"

WEALTH_FLOOR = 1e-12
BOUND_LEVEL = 0.99
D4_DEFAULT_STREAMS = 20
D4_DEFAULT_HORIZON = 100
D4_DEFAULT_BINS = 10
D5_DEFAULT_PATHS = 4
DEFAULT_ALPHA = 0.05
SCENARIOS = ("null", "peeking", "selection", "drift_lag")


def _one_sided_bounds(count: int, size: int, level: float = BOUND_LEVEL) -> tuple[float, float]:
    """Exact one-sided Clopper-Pearson bounds on an ever-alarm rate."""
    if size <= 0:
        return 0.0, 1.0
    lower = 0.0 if count <= 0 else float(stats.beta.ppf(1.0 - level, count, size - count + 1))
    upper = 1.0 if count >= size else float(stats.beta.ppf(level, count + 1, size - count))
    return lower, upper


def _online_pvalues(scores: np.ndarray, uniforms: np.ndarray) -> list[float]:
    """Eq. 9 online p-values through the frozen ``weighted_conformal_pvalue``."""
    horizon = int(scores.size)
    values: list[float] = []
    for step in range(horizon):
        values.append(
            weighted_conformal_pvalue(list(scores[: step + 1]), u=float(uniforms[step]))
        )
    return values


def _d4_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    seed = seed_of(config)
    streams = int(config.get("wctm_d4_streams", D4_DEFAULT_STREAMS))
    horizon = int(config.get("wctm_d4_horizon", D4_DEFAULT_HORIZON))
    bins = int(config.get("wctm_d4_bins", D4_DEFAULT_BINS))
    scores = np.random.default_rng([seed, 101]).standard_normal((streams, horizon))
    uniforms = np.random.default_rng([seed, 102]).uniform(size=(streams, horizon))
    pvalues = np.asarray(
        [_online_pvalues(scores[stream], uniforms[stream]) for stream in range(streams)],
        dtype=float,
    ).ravel()

    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(pvalues, bins=edges)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        rows.append(
            {
                "panel": "D4_pvalue_uniformity",
                "series": "p_hist",
                "x": float(0.5 * (edges[index] + edges[index + 1])),
                "y": float(counts[index] / pvalues.size),
                "n": int(counts[index]),
            }
        )
    for edge in (0.0, 1.0):
        rows.append(
            {
                "panel": "D4_pvalue_uniformity",
                "series": "uniform_ref",
                "x": float(edge),
                "y": float(1.0 / bins),
            }
        )

    ks = float(stats.kstest(pvalues, "uniform").statistic)
    rng = np.random.default_rng([seed, 103])
    draws = int(config.get("n_boot", 500))
    bootstrap = np.empty(max(draws, 1), dtype=float)
    for draw in range(bootstrap.size):
        resample = pvalues[rng.integers(0, pvalues.size, pvalues.size)]
        bootstrap[draw] = float(stats.kstest(resample, "uniform").statistic)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    rows.append(
        {
            "panel": "D4_pvalue_uniformity",
            "series": "ks",
            "y": ks,
            "y_low": float(min(low, ks)),
            "y_high": float(max(high, ks)),
            "n": int(pvalues.size),
            "label": f"KS distance {ks:.3f} [{min(low, ks):.3f}, {max(high, ks):.3f}] (n = {pvalues.size})",
        }
    )
    return rows, {
        "pvalues": int(pvalues.size),
        "bins": bins,
        "ks": ks,
        "ks_low": float(min(low, ks)),
        "ks_high": float(max(high, ks)),
        "mean": float(pvalues.mean()),
    }


def _d5_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    seed = seed_of(config)
    threshold_c = float(config.get("wctm_threshold_c", 20.0))
    shift_config = dict(config.get("wctm_shift", {}) or {})
    t_change = int(shift_config.get("t_change", 60))
    shift_size = float(shift_config.get("shift_size", 2.0))
    horizon = int(config.get("wctm_shift_horizon", 240))
    paths = int(config.get("wctm_path_count", D5_DEFAULT_PATHS))
    log_c = float(math.log(threshold_c))

    null_scores = np.random.default_rng([seed, 101]).standard_normal((paths, horizon))
    null_uniforms = np.random.default_rng([seed, 102]).uniform(size=(paths, horizon))
    shift_scores = np.random.default_rng([seed, 201]).standard_normal((paths, horizon))
    shift_scores[:, t_change:] += shift_size
    shift_uniforms = np.random.default_rng([seed, 202]).uniform(size=(paths, horizon))

    rows: list[dict[str, Any]] = []
    alarm_count = 0
    alarms: list[tuple[int, float]] = []
    for index in range(paths):
        for name, scores, uniforms in (
            ("null_path", null_scores, null_uniforms),
            ("shifted_path", shift_scores, shift_uniforms),
        ):
            wealth = np.asarray(
                wctm_wealth_path(_online_pvalues(scores[index], uniforms[index])), dtype=float
            )
            logs = np.log(np.maximum(wealth, WEALTH_FLOOR))
            group = f"{'null' if name == 'null_path' else 'shift'}_{index}"
            for step in range(horizon):
                rows.append(
                    {
                        "panel": "D5_wealth_paths",
                        "series": name,
                        "group": group,
                        "x": float(step),
                        "y": float(logs[step]),
                    }
                )
            crossed = np.nonzero(wealth >= threshold_c)[0]
            if crossed.size:
                first = int(crossed[0])
                rows.append(
                    {
                        "panel": "D5_wealth_paths",
                        "series": "alarm",
                        "group": group,
                        "x": float(first),
                        "y": float(logs[first]),
                        "label": f"alarm at t = {first}",
                    }
                )
                alarm_count += 1
                alarms.append((first, float(logs[first])))
    for edge in (0.0, float(horizon - 1)):
        rows.append(
            {
                "panel": "D5_wealth_paths",
                "series": "log_c",
                "x": edge,
                "y": log_c,
                "label": "log c",
            }
        )
    rows.append(
        {"panel": "D5_wealth_paths", "series": "t_change", "x": float(t_change), "label": "t_change"}
    )
    return rows, {
        "paths": paths,
        "horizon": horizon,
        "t_change": t_change,
        "shift_size": shift_size,
        "log_c": log_c,
        "alarms": alarm_count,
        "first_alarm": min((step for step, _ in alarms), default=-1),
    }


def _d6_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    experiment = run_wctm_experiment(config)
    threshold_c = float(experiment["threshold_c"])
    bound = float(experiment["one_sided_bound"])
    null_block = experiment["null"]
    variants = {row["variant"]: row for row in experiment["out_of_guarantee"]}
    alpha = float(config.get("peeking_alpha", DEFAULT_ALPHA))

    pvalues = np.asarray(null_block["pvalues"], dtype=float)
    peeking_count = int(np.sum(pvalues.min(axis=1) < alpha))
    peeking_size = int(pvalues.shape[0])
    peeking_low, peeking_high = _one_sided_bounds(peeking_count, peeking_size)

    scenarios = {
        "null": (
            float(null_block["alarm_rate"]),
            float(null_block["alarm_rate_bounds"]["lower"]),
            float(null_block["alarm_rate_bounds"]["upper"]),
            int(null_block["n_streams"]),
            "null (iid ready labels)",
        ),
        "peeking": (
            peeking_count / peeking_size,
            peeking_low,
            peeking_high,
            peeking_size,
            f"peeking (alarm when any p < {alpha:g})",
        ),
        "selection": (
            float(variants["selective_observation"]["alarm_rate"]),
            float(variants["selective_observation"]["alarm_rate_bounds"]["lower"]),
            float(variants["selective_observation"]["alarm_rate_bounds"]["upper"]),
            int(variants["selective_observation"]["n_streams"]),
            "selection (approved-only labels)",
        ),
        "drift_lag": (
            float(variants["drift_during_lag"]["alarm_rate"]),
            float(variants["drift_during_lag"]["alarm_rate_bounds"]["lower"]),
            float(variants["drift_during_lag"]["alarm_rate_bounds"]["upper"]),
            int(variants["drift_during_lag"]["n_streams"]),
            f"drift during a label lag of {int(variants['drift_during_lag']['lag'])}",
        ),
    }

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(SCENARIOS):
        rate, low, high, size, label = scenarios[name]
        rows.append(
            {
                "panel": "D6_alarm_rate",
                "series": name,
                "x": float(index),
                "y": rate,
                "y_low": low,
                "y_high": high,
                "n": size,
                "label": label,
            }
        )
    for edge in (0.0, float(len(SCENARIOS) - 1)):
        rows.append(
            {
                "panel": "D6_alarm_rate",
                "series": "bound_1_over_c",
                "x": edge,
                "y": bound,
                "label": f"1/c = {bound:g}",
            }
        )

    cdf_points: dict[int, int] = {}
    for row in experiment["shift"]["rows"]:
        lag = int(row["label_lag"])
        delays = [int(delay) for delay in row["delays"]]
        eligible = int(row["n_runs"]) - int(row["n_pre_change_false_alarms"])
        cdf_points[lag] = max(delays) if delays else 0
        for delay in range(cdf_points[lag] + 1):
            detected = sum(1 for value in delays if value <= delay)
            rows.append(
                {
                    "panel": "D6_delay_cdf",
                    "series": f"lag_{lag}",
                    "group": str(lag),
                    "x": float(delay),
                    "y": float(detected / eligible) if eligible else 0.0,
                    "n": eligible,
                }
            )
    summary = {"threshold_c": threshold_c, "bound": bound}
    for name in SCENARIOS:
        summary[f"{name}_rate"] = scenarios[name][0]
    for row in experiment["shift"]["rows"]:
        summary[f"lag_{int(row['label_lag'])}_median_delay"] = float(row["median_delay"])
        summary[f"lag_{int(row['label_lag'])}_miss_rate"] = float(row["miss_rate"])
    return rows, summary


def _caption_lines() -> list[str]:
    return [
        line[2:].strip()
        for line in SPEC_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("> ")
    ]


def _style(ax: Any, grid: str = "y") -> None:
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _head(ax: Any, text: str, handles: list[Any] | None = None, ncol: int = 4) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", pad=24 if handles else 8)
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


def _line(name: str, color: str, ls: str = "-", lw: float = 1.4) -> Line2D:
    return Line2D([0], [0], color=color, lw=lw, ls=ls, label=name)


def _figure(rows: list[dict[str, Any]]) -> Any:
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault((row["panel"], row["series"]), []).append(row)

    def series(panel: str, name: str) -> list[dict[str, Any]]:
        return by.get((panel, name), [])

    def panel_rows(panel: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["panel"] == panel]

    width, height = 21.0, 10.6
    fig = plt.figure(figsize=(width, height), layout="constrained")
    shell = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.17])
    caption = fig.add_subplot(shell[1])
    caption.axis("off")
    lines = _caption_lines()
    caption.text(
        0.0,
        1.0,
        "\n".join(
            textwrap.fill(("How to read this chart: " if index == 0 else "") + line, width=245)
            for index, line in enumerate(lines)
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        transform=caption.transAxes,
    )

    top = shell[0].subgridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 1.05],
        height_ratios=[1.0, 1.0],
        hspace=0.34,
        wspace=0.16,
    )

    ax_p = fig.add_subplot(top[:, 0])
    histogram = series("D4_pvalue_uniformity", "p_hist")
    ax_p.bar(
        [row["x"] for row in histogram],
        [row["y"] for row in histogram],
        width=0.9 / len(histogram),
        color=BLUE,
        alpha=0.85,
        label="null p-values",
    )
    reference = series("D4_pvalue_uniformity", "uniform_ref")
    ax_p.plot(
        [row["x"] for row in reference],
        [row["y"] for row in reference],
        color=INK,
        ls="--",
        lw=1.4,
        label="uniform reference",
    )
    peak = max(row["y"] for row in histogram + reference)
    ax_p.set(xlabel="conformal p-value", ylabel="share of p-values in the bin", xlim=(0.0, 1.0), ylim=(0.0, 1.3 * peak))
    _style(ax_p)
    ks_row = series("D4_pvalue_uniformity", "ks")[0]
    _head(
        ax_p,
        "D4 (pre) null p-values are uniform\n" + ks_row["label"],
        [Patch(color=BLUE, alpha=0.85, label="null p-values"), _line("uniform reference", INK, "--")],
        2,
    )

    ax_w = fig.add_subplot(top[:, 1])
    for name, colour, label in (("null_path", BLUE, "null paths"), ("shifted_path", VERM, "shifted paths")):
        groups: list[str] = []
        for row in series("D5_wealth_paths", name):
            if row["group"] not in groups:
                groups.append(row["group"])
        for group in groups:
            path = [row for row in series("D5_wealth_paths", name) if row["group"] == group]
            ax_w.plot(
                [row["x"] for row in path],
                [row["y"] for row in path],
                color=colour,
                lw=1.2,
                alpha=0.8,
                label=label if group == groups[0] else None,
            )
    alarms = series("D5_wealth_paths", "alarm")
    ax_w.scatter(
        [row["x"] for row in alarms],
        [row["y"] for row in alarms],
        marker="X",
        s=70,
        color=INK,
        zorder=5,
        edgecolor="white",
        label="alarm",
    )
    log_c = series("D5_wealth_paths", "log_c")
    ax_w.plot(
        [row["x"] for row in log_c],
        [row["y"] for row in log_c],
        color=INK,
        ls="--",
        lw=1.3,
        label="log c",
    )
    ax_w.axvline(series("D5_wealth_paths", "t_change")[0]["x"], color=GREY, ls=":", lw=1.4, label="t_change")
    ax_w.text(log_c[0]["x"] + 2.0, log_c[0]["y"] + 0.35, "log c", ha="left", fontsize=9)
    ax_w.set(xlabel="time step t", ylabel="log martingale wealth log M_t")
    _style(ax_w)
    _head(
        ax_w,
        "D5 (mid) wealth crosses log c only under the score shift",
        [
            _line("null paths", BLUE, lw=2),
            _line("shifted paths", VERM, lw=2),
            Line2D([0], [0], marker="X", color="w", markerfacecolor=INK, ms=8, label="alarm"),
            _line("log c", INK, "--"),
            _line("t_change", GREY, ":"),
        ],
        5,
    )

    right = top[:, 2].subgridspec(2, 1, hspace=0.42)
    ax_rate = fig.add_subplot(right[0])
    ax_cdf = fig.add_subplot(right[1])

    scenario_colours = {"null": BLUE, "peeking": VERM, "selection": YELLOW, "drift_lag": PURPLE}
    tick_labels: list[str] = []
    for name in SCENARIOS:
        row = series("D6_alarm_rate", name)[0]
        ax_rate.errorbar(
            [row["x"]],
            [row["y"]],
            yerr=[[row["y"] - row["y_low"]], [row["y_high"] - row["y"]]],
            fmt="o",
            color=scenario_colours[name],
            ms=8,
            capsize=4,
            lw=2,
            label=name.replace("_", " "),
        )
        tick_labels.append(textwrap.fill(name.replace("_", " "), 11))
    bound_rows = series("D6_alarm_rate", "bound_1_over_c")
    ax_rate.axhline(bound_rows[0]["y"], color=INK, ls="--", lw=1.3)
    ax_rate.text(
        float(len(SCENARIOS) - 1) + 0.45,
        bound_rows[0]["y"] * 1.18,
        bound_rows[0]["label"],
        ha="right",
        fontsize=9,
    )
    ax_rate.set_yscale("log")
    ax_rate.set_xticks(list(range(len(SCENARIOS))), tick_labels)
    ax_rate.set_xlim(-0.5, float(len(SCENARIOS)) - 0.5)
    ax_rate.set_ylim(0.006, 1.6)
    ax_rate.set(ylabel="ever-alarm rate within T (log axis)")
    _style(ax_rate)
    _head(
        ax_rate,
        "D6 (post) ever-alarm rate against the one-sided 1/c bound",
        [Patch(color=scenario_colours[name], label=name.replace("_", " ")) for name in SCENARIOS]
        + [_line("1/c", INK, "--")],
        5,
    )

    lags = []
    for row in panel_rows("D6_delay_cdf"):
        if row["group"] not in lags:
            lags.append(row["group"])
    handles = []
    for lag, colour in zip(lags, LAG_COLOURS):
        points = series("D6_delay_cdf", f"lag_{lag}")
        points = sorted(points, key=lambda row: row["x"])
        ax_cdf.step(
            [row["x"] for row in points],
            [row["y"] for row in points],
            where="post",
            color=colour,
            lw=2.0,
            label=f"L = {lag}",
        )
        handles.append(_line(f"L = {lag}", colour, lw=2.0))
    ax_cdf.set(xlabel="detection delay after t_change (steps)", ylabel="share of streams detected", ylim=(0.0, 1.04))
    _style(ax_cdf)
    _head(ax_cdf, "D6 (post) detection delay by label lag L", handles, 3)

    return fig


def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the R2_NB1_wctm_validity sidecar and figure."""
    d4_rows, d4_summary = _d4_rows(config)
    d5_rows, d5_summary = _d5_rows(config)
    d6_rows, d6_summary = _d6_rows(config)
    rows = [*d4_rows, *d5_rows, *d6_rows]

    sidecar = write_r2_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = save_r2_figure(_figure(rows), NAME, out_dir=out_dir)
    lines = _caption_lines()
    print(f"{NAME}: {len(rows)} rows -> {sidecar.name} and {figure.name}")
    print("How to read this chart: " + lines[0])
    print(lines[1])
    print(
        "D4: p-values={pvalues} bins={bins} KS={ks:.4f} [{ks_low:.4f}, {ks_high:.4f}] mean={mean:.4f}".format(
            **d4_summary
        )
    )
    print(
        "D5: paths={paths} horizon={horizon} t_change={t_change} shift={shift_size:g} log_c={log_c:.3f} "
        "alarms={alarms} first_alarm={first_alarm}".format(**d5_summary)
    )
    print(
        "D6: c={threshold_c:g} 1/c={bound:g} null={null_rate:.4f} peeking={peeking_rate:.4f} "
        "selection={selection_rate:.4f} drift_lag={drift_lag_rate:.4f} | median delays "
        "L0={lag_0_median_delay:.0f} L10={lag_10_median_delay:.0f} L30={lag_30_median_delay:.0f} | miss "
        "L0={lag_0_miss_rate:.3f} L10={lag_10_miss_rate:.3f} L30={lag_30_miss_rate:.3f}".format(**d6_summary)
    )
    return {"figure": figure, "sidecar": sidecar, "rows": len(rows), "name": NAME}
