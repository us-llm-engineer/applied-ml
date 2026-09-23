"""R2_NB3_monitoring_and_ledger: E9 (post) + E10 (post) of the monitoring-and-ledger study.

Panels
------
E9 (three stacked timelines sharing the day axis, one dot-plot column): the daily
alert rate against its budget, the daily score-drift statistic against its
pre-change envelope, and the WCTM log wealth on matured labels against log c,
each with its first alarm marked; median detection delay and miss rate per
monitor with intervals.
E10: label cost and wall-clock seconds per study, fresh against cached.

Data sources: ``exec.monitoring.run_stream_monitoring_experiment`` (windows,
daily series, drift statistic and WCTM summaries) and ``exec.ledger.run_estimator``
for four studies (sar, bayesian, wctm and sar under an alternate seed), each run
fresh and then cached. Nothing is read from ``viz/mock``.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from exec.config import config_hash, seed_of
from exec.ledger import run_estimator
from exec.monitoring import run_stream_monitoring_experiment
from exec.vizlib import ROOT, save_r2_figure, write_r2_sidecar

NAME = "R2_NB3_monitoring_and_ledger"

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

#: One colour per monitor, held fixed across the timelines, the dot plots and
#: their legends.
MONITOR_COLORS = {"alert_rate": PURPLE, "drift_stat": GREEN, "wctm": BLUE}
MONITOR_LABELS = {
    "alert_rate": "alert rate",
    "drift_stat": "drift stat",
    "wctm": "WCTM (matured)",
}

#: The four ledger studies. ``(study name, estimator, seed offset)``; the fourth
#: is the SAR estimator under an alternate seed, so the ledger covers four
#: distinct studies while the other three share one canonical config.
STUDIES = (
    ("sar", "sar", 0),
    ("bayesian", "bayesian", 0),
    ("wctm", "wctm", 0),
    ("sar (alt seed)", "sar", 1),
)
LEDGER_FILENAME = "r2_monitoring_ledger.jsonl"
CACHE_ROOT = ROOT / "exec" / "results" / "ledger_cache"

#: Moving-block bootstrap for the alert-rate and drift-statistic monitors (one
#: stream only: the replicate streams behind the WCTM summary are not exposed
#: for the other two monitors). Block draws preserve local day-to-day drift.
BOOTSTRAP_DRAWS = 400
BOOTSTRAP_BLOCK = 5

#: Verbatim two-line "How to read this chart" block from
#: ``viz/spec/R2_NB3_monitoring_and_ledger.md`` (wrapped for the PNG footer only).
HOW_TO_READ_LINE_1 = (
    "Top left: three timelines on one day axis after the adversary drift (dashed "
    "vertical line): alert rate against its budget, the score-drift statistic against "
    "its threshold, and WCTM log wealth against log c; a marker shows each monitor's "
    "first alarm and its delay. Top right: median detection delay and miss rate per "
    "monitor with intervals. Bottom: label cost and wall-clock seconds per study, "
    "fresh against cached."
)
HOW_TO_READ_LINE_2 = (
    "Look for: lines crossing their threshold after the drift day and not before, "
    "delays that differ by monitor with intervals, and cached bars with non-zero cost "
    "but far less time. For the job: it shows which monitor sees a label-delayed drift "
    "first, what the delay costs, and that the run ledger keeps cost honest across "
    "cached re-runs."
)

_CACHE: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}


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


def _first_crossing(xs: Any, ys: Any, after: int, threshold: float) -> tuple[int, float] | None:
    """First (x, y) with x > after and y > threshold, or None."""
    for x, y in zip(xs, ys):
        if int(x) > after and float(y) > threshold:
            return int(x), float(y)
    return None


def _block_draw(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One moving-block bootstrap resample of a 1-d series."""
    values = np.asarray(values, dtype=float)
    out: list[float] = []
    while len(out) < values.size:
        start = int(rng.integers(0, values.size))
        out.extend(values[start : start + BOOTSTRAP_BLOCK].tolist())
    return np.asarray(out[: values.size], dtype=float)


def _bootstrap_first_crossings(
    series: np.ndarray,
    threshold: float | None,
    reference: np.ndarray | None,
    rng: np.random.Generator,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[int | None]:
    """Block-bootstrap first-crossing delays for a monitor (None = missed)."""
    delays: list[int | None] = []
    for _ in range(draws):
        drawn = _block_draw(series, rng)
        limit = threshold
        if reference is not None:
            limit = float(_block_draw(reference, rng).max())
        hit = _first_crossing(np.arange(drawn.size), drawn, -1, float(limit))
        delays.append(int(hit[0]) if hit is not None else None)
    return delays


def _summarise_delay(delays: list[int | None]) -> dict[str, Any]:
    """Point estimate and interval of a bootstrap first-crossing delay."""
    crossed = sorted(delay for delay in delays if delay is not None)
    misses = len(delays) - len(crossed)
    low = high = float("nan")
    median = float("nan")
    if crossed:
        median = float(np.median(crossed))
        low = float(np.percentile(crossed, 2.5))
        high = float(np.percentile(crossed, 97.5))
    miss_low, miss_high = _wilson(misses, len(delays))
    return {
        "delay_point": median,
        "delay_low": low,
        "delay_high": high,
        "miss_rate": misses / len(delays) if delays else float("nan"),
        "miss_low": miss_low,
        "miss_high": miss_high,
        "n_draws": len(delays),
        "n_crossed": len(crossed),
    }


# ------------------------------------------------------------------ aggregation
def _monitor_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    monitoring = run_stream_monitoring_experiment(config)
    daily = monitoring["daily"]
    steps = [int(value) for value in daily["step_index"]]
    t_drift_step = int(daily["t_drift_step"])
    budget_per_1000 = float(monitoring["alert_budget"]) * 1000.0
    alert_rate = [float(value) for value in daily["alerts_per_1000"]]
    drift_series = [float(value) for value in daily["drift_statistic"]]
    pre = [value for step, value in zip(steps, drift_series) if step < t_drift_step]
    drift_threshold = float(max(pre))

    alert_hit = _first_crossing(steps, alert_rate, t_drift_step, budget_per_1000)
    drift_hit = _first_crossing(steps, drift_series, t_drift_step, drift_threshold)
    if alert_hit is None or drift_hit is None:
        raise ValueError("a score-based monitor never crossed its threshold after the drift")

    matured = monitoring["wctm"]["matured"]
    wctm_steps = [int(value) for value in matured["step_index"]]
    alarm_path = [float(value) for value in matured["alarm_log_wealth_path"]]
    median_path = [float(value) for value in matured["median_log_wealth_path"]]
    log_c = float(np.log(float(monitoring["wctm_threshold_c"])))
    alarm_step = matured["alarm_first_step"]
    if alarm_step is None:
        raise ValueError("the matured-label WCTM never alarmed")
    position = wctm_steps.index(int(alarm_step))
    if alarm_path[position] < log_c:
        raise ValueError("the WCTM alarm step does not sit above log c")

    rows: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        rows.append(
            {"panel": "E9_alert_rate", "series": "alert_rate", "x": step, "y": round(alert_rate[index], 8)}
        )
        rows.append(
            {"panel": "E9_drift_stat", "series": "drift_stat", "x": step, "y": round(drift_series[index], 8)}
        )
    for index, step in enumerate(wctm_steps):
        rows.append(
            {"panel": "E9_wctm_wealth", "series": "wctm_log_wealth", "x": step, "y": round(alarm_path[index], 8)}
        )
        rows.append(
            {
                "panel": "E9_wctm_wealth",
                "series": "wctm_log_wealth_median",
                "x": step,
                "y": round(median_path[index], 8),
            }
        )
    span = (steps[0], steps[-1])
    rows.append({"panel": "E9_alert_rate", "series": "alert_budget", "x": span[0], "y": round(budget_per_1000, 8)})
    rows.append({"panel": "E9_alert_rate", "series": "alert_budget", "x": span[1], "y": round(budget_per_1000, 8)})
    rows.append({"panel": "E9_drift_stat", "series": "drift_threshold", "x": span[0], "y": round(drift_threshold, 8)})
    rows.append({"panel": "E9_drift_stat", "series": "drift_threshold", "x": span[1], "y": round(drift_threshold, 8)})
    rows.append({"panel": "E9_wctm_wealth", "series": "log_c", "x": wctm_steps[0], "y": round(log_c, 8)})
    rows.append({"panel": "E9_wctm_wealth", "series": "log_c", "x": wctm_steps[-1], "y": round(log_c, 8)})
    rows.append({"panel": "E9_alert_rate", "series": "t_drift", "x": t_drift_step})
    rows.append(
        {
            "panel": "E9_alert_rate",
            "series": "first_alarm",
            "x": alert_hit[0],
            "y": round(alert_hit[1], 8),
            "label": f"first alarm day {alert_hit[0]} (delay {alert_hit[0] - t_drift_step} d)",
        }
    )
    rows.append(
        {
            "panel": "E9_drift_stat",
            "series": "first_alarm",
            "x": drift_hit[0],
            "y": round(drift_hit[1], 8),
            "label": f"first alarm day {drift_hit[0]} (delay {drift_hit[0] - t_drift_step} d)",
        }
    )
    rows.append(
        {
            "panel": "E9_wctm_wealth",
            "series": "first_alarm",
            "x": int(alarm_step),
            "y": round(alarm_path[position], 8),
            "label": f"first alarm day {int(alarm_step)} (delay {int(alarm_step) - t_drift_step} d)",
        }
    )

    rng = np.random.default_rng([seed_of(config), 4242])
    post_alert = np.asarray([value for step, value in zip(steps, alert_rate) if step >= t_drift_step])
    post_drift = np.asarray([value for step, value in zip(steps, drift_series) if step >= t_drift_step])
    alert_summary = _summarise_delay(
        _bootstrap_first_crossings(post_alert, budget_per_1000, None, rng)
    )
    drift_summary = _summarise_delay(
        _bootstrap_first_crossings(post_drift, None, np.asarray(pre), rng)
    )
    wctm_interval = matured["median_delay_interval"]
    wctm_miss = matured["miss_rate_interval"]
    delay_table = {
        "alert_rate": {
            "delay_point": float(alert_hit[0] - t_drift_step),
            "delay_low": alert_summary["delay_low"],
            "delay_high": alert_summary["delay_high"],
            "n": alert_summary["n_draws"],
            "label": f"moving-block bootstrap, {alert_summary['n_draws']} draws",
        },
        "drift_stat": {
            "delay_point": float(drift_hit[0] - t_drift_step),
            "delay_low": drift_summary["delay_low"],
            "delay_high": drift_summary["delay_high"],
            "n": drift_summary["n_draws"],
            "label": f"moving-block bootstrap, {drift_summary['n_draws']} draws",
        },
        "wctm": {
            "delay_point": float(matured["median_delay"]),
            "delay_low": float(wctm_interval["low"]),
            "delay_high": float(wctm_interval["high"]),
            "n": int(matured["n_runs"]),
            "label": (
                f"order-statistic interval, {matured['n_detected']}/{matured['n_eligible']} "
                f"replicate streams detected"
            ),
        },
    }
    miss_table = {
        "alert_rate": {
            "miss_rate": float(alert_summary["miss_rate"]),
            "miss_low": float(alert_summary["miss_low"]),
            "miss_high": float(alert_summary["miss_high"]),
            "n": alert_summary["n_draws"],
            "label": f"Wilson interval, {alert_summary['n_draws']} bootstrap draws",
        },
        "drift_stat": {
            "miss_rate": float(drift_summary["miss_rate"]),
            "miss_low": float(drift_summary["miss_low"]),
            "miss_high": float(drift_summary["miss_high"]),
            "n": drift_summary["n_draws"],
            "label": f"Wilson interval, {drift_summary['n_draws']} bootstrap draws",
        },
        "wctm": {
            "miss_rate": float(matured["miss_rate"]),
            "miss_low": float(wctm_miss["low"]),
            "miss_high": float(wctm_miss["high"]),
            "n": int(matured["n_runs"]),
            "label": (
                f"Clopper-Pearson interval, {matured['n_eligible']} eligible of "
                f"{matured['n_runs']} replicate streams"
            ),
        },
    }
    for index, monitor in enumerate(("alert_rate", "drift_stat", "wctm")):
        delay = delay_table[monitor]
        rows.append(
            {
                "panel": "E9_detection_delay",
                "series": "delay_median",
                "group": monitor,
                "x": index,
                "y": round(delay["delay_point"], 8),
                "y_low": round(delay["delay_low"], 8),
                "y_high": round(delay["delay_high"], 8),
                "n": delay["n"],
                "label": delay["label"],
            }
        )
        miss = miss_table[monitor]
        rows.append(
            {
                "panel": "E9_miss_rate",
                "series": "miss_rate",
                "group": monitor,
                "x": index,
                "y": round(miss["miss_rate"], 8),
                "y_low": round(miss["miss_low"], 8),
                "y_high": round(miss["miss_high"], 8),
                "n": miss["n"],
                "label": miss["label"],
            }
        )

    meta = {
        "t_drift_step": t_drift_step,
        "step_size_records": int(daily["step_size_records"]),
        "budget_per_1000": budget_per_1000,
        "drift_threshold": drift_threshold,
        "log_c": log_c,
        "threshold_c": float(monitoring["wctm_threshold_c"]),
        "alert_first_alarm": alert_hit,
        "drift_first_alarm": drift_hit,
        "wctm_first_alarm": (int(alarm_step), float(alarm_path[position])),
        "delay_table": delay_table,
        "miss_table": miss_table,
        "steps": steps,
        "wctm_steps": wctm_steps,
        "n_runs": int(monitoring["n_runs"]),
        "config_hash": str(monitoring["config_hash"]),
    }
    return rows, meta


def _ledger_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(tempfile.mkdtemp(prefix="build-", dir=CACHE_ROOT))
    ledger_path = ROOT / "exec" / "results" / LEDGER_FILENAME
    rows: list[dict[str, Any]] = []
    table: dict[str, dict[str, float]] = {}
    try:
        for index, (study, estimator, offset) in enumerate(STUDIES):
            run_config = {
                **dict(config),
                "seed": int(config["seed"]) + offset,
                "ledger_path": str(ledger_path),
                "cache_dir": str(cache_dir),
            }
            started = time.perf_counter()
            fresh = run_estimator(estimator, run_config)
            fresh_seconds = time.perf_counter() - started
            started = time.perf_counter()
            cached = run_estimator(estimator, run_config)
            cached_seconds = time.perf_counter() - started
            if fresh["ledger_row"]["cached"] or not cached["ledger_row"]["cached"]:
                raise RuntimeError(f"the ledger did not serve {study!r} from the cache on the re-run")
            fresh_cost = int(fresh["ledger_row"]["label_cost"])
            cached_cost = int(cached["ledger_row"]["label_cost"])
            if fresh_cost <= 0 or cached_cost <= 0:
                raise RuntimeError(f"a ledger run of {study!r} reported no label cost")
            for series, cost, seconds in (
                ("fresh", fresh_cost, fresh_seconds),
                ("cached", cached_cost, cached_seconds),
            ):
                rows.append(
                    {"panel": "E10_ledger_cost", "series": series, "group": study, "x": index, "y": cost}
                )
                rows.append(
                    {
                        "panel": "E10_ledger_wallclock",
                        "series": series,
                        "group": study,
                        "x": index,
                        "y": round(max(float(seconds), 1e-6), 6),
                    }
                )
            table[study] = {
                "estimator": estimator,
                "label_cost": float(fresh_cost),
                "fresh_seconds": float(fresh_seconds),
                "cached_seconds": float(cached_seconds),
                "run_id": str(fresh["ledger_row"]["run_id"]),
                "data_hash": str(fresh["ledger_row"]["data_hash"]),
            }
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
    return rows, {"studies": table, "ledger_path": str(ledger_path)}


def _cached(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = config_hash(config)
    if key not in _CACHE:
        monitor_rows, monitor_meta = _monitor_rows(config)
        ledger_rows, ledger_meta = _ledger_rows(config)
        meta = {**monitor_meta, **ledger_meta}
        _CACHE[key] = (monitor_rows + ledger_rows, meta)
    rows, meta = _CACHE[key]
    return list(rows), dict(meta)


# ------------------------------------------------------------------ rendering
def _style(ax: Any, grid: str = "y") -> None:
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _head(ax: Any, text: str, handles: Any = None, ncol: int = 4) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", pad=26 if handles else 8)
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


def _alarm_marker() -> Line2D:
    return Line2D([0], [0], marker="X", color="w", markerfacecolor=INK, ms=8, label="first alarm")


def _caption_text(width: int) -> str:
    first = textwrap.fill("How to read this chart: " + HOW_TO_READ_LINE_1, width=width)
    second = textwrap.fill(HOW_TO_READ_LINE_2, width=width)
    return first + "\n" + second


def _render(rows: list[dict[str, Any]], meta: Mapping[str, Any]) -> Any:
    def by(panel: str, series: str, group: str | None = None) -> list[dict[str, Any]]:
        selected = [
            r for r in rows if r["panel"] == panel and r["series"] == series
        ]
        if group is not None:
            selected = [r for r in selected if r["group"] == group]
        return selected

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#888888",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(20.0, 13.2), layout="constrained")
    outer = fig.add_gridspec(3, 1, height_ratios=[3.0, 1.05, 0.14])
    caption = fig.add_subplot(outer[2])
    caption.axis("off")
    caption.text(
        0.0,
        1.0,
        _caption_text(int(20.0 * 13)),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        transform=caption.transAxes,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[2.3, 1.0], wspace=0.15)

    # --- top left: three timelines sharing the day axis ---------------------
    timelines = [fig.add_subplot(spec) for spec in top[0].subgridspec(3, 1, hspace=0.12)]
    for index in range(1, 3):
        timelines[index].sharex(timelines[0])
    t_drift = float(meta["t_drift_step"])
    specs = (
        ("E9_alert_rate", "alert_rate", "alert_budget", "alerts per 1,000 transactions",
         "alert budget", MONITOR_COLORS["alert_rate"], "E9 (post) alert rate against its budget"),
        ("E9_drift_stat", "drift_stat", "drift_threshold", "drift statistic (per 1,000)",
         "drift threshold", MONITOR_COLORS["drift_stat"], "E9 (post) score-drift statistic"),
        ("E9_wctm_wealth", "wctm_log_wealth", "log_c", "WCTM log wealth on matured labels (nats)",
         f"log c = {meta['log_c']:.2f}", MONITOR_COLORS["wctm"], "E9 (post) WCTM on matured labels"),
    )
    for axis, (panel, series, reference, ylabel, ref_label, color, title) in zip(timelines, specs):
        data = by(panel, series)
        axis.plot([r["x"] for r in data], [r["y"] for r in data], color=color, lw=2)
        medians = by(panel, "wctm_log_wealth_median")
        if medians:
            axis.plot(
                [r["x"] for r in medians],
                [r["y"] for r in medians],
                color=color,
                lw=1.2,
                ls="--",
                alpha=0.65,
            )
        reference_rows = by(panel, reference)
        axis.plot(
            [r["x"] for r in reference_rows],
            [r["y"] for r in reference_rows],
            color=INK,
            ls="--",
            lw=1.3,
        )
        axis.axvline(t_drift, color=GREY, ls=":", lw=1.4)
        alarm = by(panel, "first_alarm")[0]
        axis.scatter([alarm["x"]], [alarm["y"]], marker="X", s=90, color=INK, edgecolor="white", zorder=6)
        axis.annotate(
            alarm["label"],
            (alarm["x"], alarm["y"]),
            xytext=(8, -16 if panel != "E9_wctm_wealth" else 8),
            textcoords="offset points",
            fontsize=9,
        )
        axis.set(ylabel=ylabel)
        _style(axis)
        handles = [
            _line(series.replace("_", " "), color, lw=2),
            _line(ref_label, INK, "--"),
            _line("drift day", GREY, ":"),
            _alarm_marker(),
        ]
        if medians:
            handles.insert(1, _line("median over replicate streams", color, "--", 1.2))
        _head(axis, title, handles, len(handles))
    timelines[2].set_xlabel(
        "time (day; one step = %d transactions)" % int(meta["step_size_records"])
    )
    for axis in timelines[:2]:
        axis.tick_params(labelbottom=False)

    # --- top right: detection delay and miss rate --------------------------
    dots = [fig.add_subplot(spec) for spec in top[1].subgridspec(2, 1, hspace=0.16)]
    monitors = ("alert_rate", "drift_stat", "wctm")
    for axis, panel, ylabel, title, table in (
        (dots[0], "E9_detection_delay", "median delay (days after drift)",
         "E9 (post) detection delay", meta["delay_table"]),
        (dots[1], "E9_miss_rate", "miss rate",
         "E9 (post) miss rate", meta["miss_table"]),
    ):
        for index, monitor in enumerate(monitors):
            selected = [
                r for r in rows
                if r["panel"] == panel and r["group"] == monitor
            ][0]
            y = float(selected["y"])
            low, high = float(selected["y_low"]), float(selected["y_high"])
            axis.errorbar(
                [index], [y], yerr=[[y - low], [high - y]],
                fmt="o", color=MONITOR_COLORS[monitor], capsize=4, ms=8, lw=2,
            )
        axis.set_xticks([0, 1, 2], [MONITOR_LABELS[m] for m in monitors])
        axis.set_xlim(-0.6, 2.6)
        axis.set_ylim(0, None)
        axis.set(ylabel=ylabel)
        _style(axis)
        _head(axis, title, [Patch(color=MONITOR_COLORS[m], label=MONITOR_LABELS[m]) for m in monitors], 3)

    # --- bottom: the run ledger, fresh versus cached -----------------------
    bottom = outer[1].subgridspec(1, 2, wspace=0.15)
    study_rows = sorted(by("E10_ledger_cost", "fresh"), key=lambda r: r["x"])
    study_names = [r["group"] for r in study_rows]
    for panel, ylabel, title, log_axis in (
        ("E10_ledger_cost", "labels bought (cost)", "E10 (post) ledger label cost", False),
        ("E10_ledger_wallclock", "wall-clock (s, log axis)", "E10 (post) ledger wall-clock", True),
    ):
        axis = fig.add_subplot(bottom[0 if not log_axis else 1])
        for series, color, offset in (("fresh", BLUE, -0.19), ("cached", YELLOW, 0.19)):
            data = sorted(by(panel, series), key=lambda r: r["x"])
            axis.bar(
                [r["x"] + offset for r in data],
                [r["y"] for r in data],
                width=0.36,
                color=color,
            )
        axis.set_xticks(range(len(study_names)), [textwrap.fill(name, 9) for name in study_names])
        if log_axis:
            axis.set_yscale("log")
            values = [float(r["y"]) for r in by(panel, "fresh") + by(panel, "cached")]
            axis.set_ylim(min(values) * 0.3, max(values) * 3.0)
        axis.set(ylabel=ylabel)
        _style(axis)
        _head(axis, title, [Patch(color=BLUE, label="fresh run"), Patch(color=YELLOW, label="cached re-run")], 2)
    return fig


def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Run the monitoring experiment and the ledger studies, then render the figure."""
    rows, meta = _cached(config)
    sidecar_path = write_r2_sidecar(NAME, rows, config, out_dir=out_dir)
    fig = _render(rows, meta)
    figure_path = save_r2_figure(fig, NAME, out_dir=out_dir)
    plt.close(fig)
    print("How to read this chart: " + HOW_TO_READ_LINE_1)
    print(HOW_TO_READ_LINE_2)
    print(
        f"{NAME}: {len(rows)} rows; alert breach day {meta['alert_first_alarm'][0]} "
        f"(delay {meta['alert_first_alarm'][0] - meta['t_drift_step']} d), drift crossing day "
        f"{meta['drift_first_alarm'][0]} (delay {meta['drift_first_alarm'][0] - meta['t_drift_step']} d), "
        f"WCTM first alarm day {meta['wctm_first_alarm'][0]} (delay "
        f"{meta['wctm_first_alarm'][0] - meta['t_drift_step']} d); WCTM median delay "
        f"{meta['delay_table']['wctm']['delay_point']:.0f} d, miss rate "
        f"{meta['miss_table']['wctm']['miss_rate']:.2f}"
    )
    for study, values in meta["studies"].items():
        print(
            f"  ledger {study}: cost {values['label_cost']:.0f} labels, fresh "
            f"{values['fresh_seconds']:.2f} s, cached {values['cached_seconds']:.4f} s"
        )
    return {
        "figure": str(figure_path),
        "sidecar": str(sidecar_path),
        "rows": int(len(rows)),
        "name": NAME,
    }
