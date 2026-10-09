"""Like-for-like comparison of the three monitors.

Every number is measured on this run's own null and drift streams.

The study (:func:`label_delay.monitoring.run_equal_footing_monitor_comparison`) pushes
the same ``n_null_runs`` null and ``n_drift_runs`` drift streams through all
three monitors: the budget-limited alert rate, the tail-weighted drift statistic
and the WCTM, which is built from the classification nonconformity score
``1 - p_hat(y|x)`` rather than a raw sign process. One panel per metric keeps the
monitors on the same footing: false-alarm incidence on the null stream with the
exact one-sided Clopper-Pearson upper bound, median detection delay on the drift
stream with its distribution-free interval, and miss rate on the drift stream
with the exact two-sided Clopper-Pearson interval. The figure is only
consistent if the delay and miss-rate orderings agree; a contradiction would be
annotated in the affected rows, never hidden behind a tuned number.
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from label_delay.config import config_hash
from label_delay.monitoring import run_equal_footing_monitor_comparison
from label_delay.vizlib import ROOT, save_study_figure, write_study_sidecar

NAME = "monitor_comparison"
CAPTION_LINES = [
    'each panel puts the three monitors (budget-limited alert rate, tail-weighted drift statistic and WCTM on the classification score) on the same stream and the same score: false-alarm incidence on the null stream with its exact upper bound, median detection delay on the drift stream, and miss rate on the drift stream with exact intervals.',
    'The ranking is only meaningful where the delay and miss-rate orderings agree; any contradiction is annotated in the affected rows rather than tuned away.',
]

BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
INK, GRID = "#222222", "#E4E4E4"

MONITORS = ("alert_rate", "drift_stat", "wctm")
MONITOR_COLORS = {"alert_rate": BLUE, "drift_stat": VERM, "wctm": GREEN}
MONITOR_LABELS = {"alert_rate": "alert rate", "drift_stat": "drift stat", "wctm": "WCTM"}
WCTM_SCORE_LABEL = "score=1 - p_hat(y|x)"

_CACHE: dict[str, dict[str, Any]] = {}


# ------------------------------------------------------------------ data
def _caption_lines() -> list[str]:
    return list(CAPTION_LINES)


def _comparison(config: Mapping[str, Any]) -> dict[str, Any]:
    key = config_hash(config)
    if key not in _CACHE:
        _CACHE[key] = run_equal_footing_monitor_comparison(config)
    return _CACHE[key]


def _ranking_note(delays: Mapping[str, float], misses: Mapping[str, float]) -> str:
    """Explanatory note when the delay and miss-rate orderings disagree."""
    delay_rank = sorted(MONITORS, key=lambda monitor: (delays[monitor], monitor))
    miss_rank = sorted(MONITORS, key=lambda monitor: (misses[monitor], monitor))
    if delay_rank == miss_rank:
        return ""
    return (
        "ranking differs between the delay and miss-rate panels (fastest: "
        + ", ".join(delay_rank)
        + "; lowest miss: "
        + ", ".join(miss_rank)
        + "); both orderings are reported as measured"
    )


def _rows(comparison: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "n_null_runs": int(comparison["n_null_runs"]),
        "n_drift_runs": int(comparison["n_drift_runs"]),
        "score_kind": str(comparison["score_kind"]),
        "score_formula": str(comparison["score_formula"]),
        "alert_budget": float(comparison["alert_budget"]),
        "monitors": {},
    }
    delays: dict[str, float] = {}
    misses: dict[str, float] = {}
    for index, monitor in enumerate(MONITORS):
        entry = comparison["monitors"][monitor]
        null_row, drift_row = entry["null"], entry["drift"]
        n_null = int(null_row["n_runs"])
        n_drift = int(drift_row["n_runs"])
        false_alarms = int(null_row["n_false_alarms"])
        upper = float(null_row["false_alarm_upper_bound"])
        incidence = float(null_row["empirical_false_alarm_rate"])
        delay_interval = drift_row["median_delay_interval"]
        delay = float(drift_row["median_delay"])
        miss_interval = drift_row["miss_rate_interval"]
        miss = float(drift_row["miss_rate"])
        score_label = WCTM_SCORE_LABEL if monitor == "wctm" else ""
        rows.append(
            {
                "panel": "false_alarm_incidence",
                "series": monitor,
                "group": "null_stream",
                "x": float(index),
                "y": round(incidence, 8),
                "y_low": 0.0,
                "y_high": round(upper, 8),
                "n": n_null,
                "label": score_label,
            }
        )
        rows.append(
            {
                "panel": "detection_delay",
                "series": monitor,
                "group": "drift_stream",
                "x": float(index),
                "y": round(delay, 8),
                "y_low": round(float(delay_interval["low"]), 8),
                "y_high": round(float(delay_interval["high"]), 8),
                "n": n_drift,
                "label": score_label,
            }
        )
        rows.append(
            {
                "panel": "miss_rate",
                "series": monitor,
                "group": "drift_stream",
                "x": float(index),
                "y": round(miss, 8),
                "y_low": round(float(miss_interval["low"]), 8),
                "y_high": round(float(miss_interval["high"]), 8),
                "n": n_drift,
                "label": score_label,
            }
        )
        summary["monitors"][monitor] = {
            "n_null_runs": n_null,
            "n_drift_runs": n_drift,
            "false_alarms": false_alarms,
            "false_alarm_incidence": incidence,
            "false_alarm_upper": upper,
            "median_delay": delay,
            "delay_low": float(delay_interval["low"]),
            "delay_high": float(delay_interval["high"]),
            "miss_rate": miss,
            "miss_low": float(miss_interval["low"]),
            "miss_high": float(miss_interval["high"]),
        }
        delays[monitor] = delay
        misses[monitor] = miss

    note = _ranking_note(delays, misses)
    if note:
        for row in rows:
            if row["panel"] in ("detection_delay", "miss_rate"):
                row["label"] = (row["label"] + "; " if row["label"] else "") + note
    summary["ranking_consistent"] = note == ""
    summary["delay_rank"] = sorted(MONITORS, key=lambda monitor: (delays[monitor], monitor))
    summary["miss_rank"] = sorted(MONITORS, key=lambda monitor: (misses[monitor], monitor))
    return rows, summary


# ------------------------------------------------------------------ figure
def _style(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _marker(label: str, color: str, ms: float = 7) -> Line2D:
    return Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor=color, ms=ms, label=label
    )


def _caption_text() -> str:
    return "\n".join(
        textwrap.fill(("How to read this chart: " if index == 0 else "") + line, width=245)
        for index, line in enumerate(_caption_lines())
    )


def _draw_panel(ax: Any, rows: list[dict[str, Any]], annotate: Callable[[dict[str, Any], float], str]) -> None:
    for row in sorted(rows, key=lambda item: item["x"]):
        y = float(row["y"])
        low = float(row["y_low"])
        high = float(row["y_high"])
        colour = MONITOR_COLORS[row["series"]]
        ax.errorbar(
            [row["x"]],
            [y],
            yerr=[[y - low], [high - y]],
            fmt="o",
            ms=7,
            lw=1.7,
            capsize=4,
            color=colour,
        )
        ax.annotate(
            annotate(row, high),
            xy=(row["x"], high),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.4,
            color=colour,
        )
    ax.set_xticks(range(len(MONITORS)), [MONITOR_LABELS[monitor] for monitor in MONITORS], rotation=15)
    for tick, monitor in zip(ax.get_xticklabels(), MONITORS):
        tick.set_color(MONITOR_COLORS[monitor])
    ax.set_xlim(-0.6, 2.6)
    _style(ax)


def _figure(rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> Any:
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
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by.setdefault((row["panel"], row["series"]), []).append(row)

    def series(panel: str, name: str) -> list[dict[str, Any]]:
        return by.get((panel, name), [])

    width, height = 21.0, 6.8
    fig = plt.figure(figsize=(width, height), layout="constrained")
    shell = fig.add_gridspec(3, 1, height_ratios=[0.085, 1.0, 0.15])
    legend_ax = fig.add_subplot(shell[0])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[_marker(MONITOR_LABELS[monitor], MONITOR_COLORS[monitor]) for monitor in MONITORS],
        loc="center left",
        ncol=3,
        frameon=False,
        fontsize=9.5,
        handlelength=1.6,
        columnspacing=1.5,
    )
    caption = fig.add_subplot(shell[2])
    caption.axis("off")
    caption.text(
        0.0,
        1.0,
        _caption_text(),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        transform=caption.transAxes,
    )
    fig.suptitle(
        "equal-footing monitor comparison on the classification score "
        f"1 - p_hat(y|x) ({summary['n_null_runs']} null + {summary['n_drift_runs']} drift runs)",
        x=0.0,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    grid = shell[1].subgridspec(1, 3, wspace=0.30)

    # --- panel 1: false-alarm incidence (null stream) ----------------------
    ax_false_alarm = fig.add_subplot(grid[0])
    _draw_panel(
        ax_false_alarm,
        [row for monitor in MONITORS for row in series("false_alarm_incidence", monitor)],
        lambda row, high: f"{summary['monitors'][row['series']]['false_alarms']}/"
        f"{summary['monitors'][row['series']]['n_null_runs']}",
    )
    budget = float(summary["alert_budget"])
    ax_false_alarm.axhline(budget, color=GREY, ls="--", lw=1.3)
    ax_false_alarm.annotate(
        f"nominal alert budget = {budget:.0%}",
        xy=(0.02, budget),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8.4,
        color=GREY,
    )
    upper_max = max(
        float(row["y_high"]) for monitor in MONITORS for row in series("false_alarm_incidence", monitor)
    )
    ax_false_alarm.set(
        ylabel="false-alarm incidence on the null stream",
        ylim=(0.0, upper_max * 1.22),
        title="false-alarm incidence (null stream)",
    )

    # --- panel 2: median detection delay (drift stream) --------------------
    ax_delay = fig.add_subplot(grid[1])
    _draw_panel(
        ax_delay,
        [row for monitor in MONITORS for row in series("detection_delay", monitor)],
        lambda row, high: f"{float(row['y']):.0f}",
    )
    delay_rows = [row for monitor in MONITORS for row in series("detection_delay", monitor)]
    ax_delay.set(
        ylabel="median delay (steps after the change)",
        ylim=(
            min(float(row["y_low"]) for row in delay_rows) * 0.88,
            max(float(row["y_high"]) for row in delay_rows) * 1.12,
        ),
        title="detection delay (drift stream)",
    )

    # --- panel 3: miss rate (drift stream) ---------------------------------
    ax_miss = fig.add_subplot(grid[2])
    _draw_panel(
        ax_miss,
        [row for monitor in MONITORS for row in series("miss_rate", monitor)],
        lambda row, high: f"{float(row['y']):.2f}",
    )
    miss_rows = [row for monitor in MONITORS for row in series("miss_rate", monitor)]
    ax_miss.set(
        ylabel="miss rate (fraction of eligible drift runs)",
        ylim=(0.0, min(1.05, max(float(row["y_high"]) for row in miss_rows) * 1.12)),
        title="miss rate (drift stream)",
    )
    return fig


# ------------------------------------------------------------------ entry
def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the monitor_comparison sidecar and figure."""
    comparison = _comparison(config)
    rows, summary = _rows(comparison)
    sidecar = write_study_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = save_study_figure(_figure(rows, summary), NAME, out_dir=out_dir)
    plt.close()
    lines = _caption_lines()
    print(f"{NAME}: {len(rows)} rows -> {sidecar.name} and {figure.name}")
    print("How to read this chart: " + lines[0])
    print(lines[1])
    for monitor in MONITORS:
        values = summary["monitors"][monitor]
        print(
            f"{monitor}: null false alarms {values['false_alarms']}/{values['n_null_runs']} "
            f"(upper bound {values['false_alarm_upper']:.4f}), median delay {values['median_delay']:.0f} "
            f"[{values['delay_low']:.0f}, {values['delay_high']:.0f}], miss rate {values['miss_rate']:.4f} "
            f"[{values['miss_low']:.4f}, {values['miss_high']:.4f}]"
        )
    print(
        "delay ranking (fastest first): "
        + " < ".join(summary["delay_rank"])
        + "; miss-rate ranking (lowest first): "
        + " < ".join(summary["miss_rank"])
        + ("; consistent" if summary["ranking_consistent"] else "; INCONSISTENT (rows carry the note)")
    )
    return {"figure": str(figure), "sidecar": str(sidecar), "rows": int(len(rows)), "name": NAME}
