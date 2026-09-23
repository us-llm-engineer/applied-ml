"""NB3 monitoring-validity panel: three-panel figure from the C5--C6 experiment.

Pre panel: label-arrival lanes (marker = label status, colour = variant).
Mid panel: WATCH log-wealth paths (line = variant, dashed log-threshold reference).
Post panel: false-alarm incidence with Wilson intervals, 1/threshold reference,
and annotated detection delay. All rows are measured by ``exec.monitoring``;
``viz/mock`` is never a data source. Paths are floored at 1e-12 before logging.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from exec.monitoring import run_monitoring_experiment
from exec.vizlib import save_figure, write_sidecar

NAME = "NB3_monitoring_validity_panel"
PALETTE = {"ready": "#0072B2", "selective": "#D55E00", "delayed": "#009E73"}
VARIANT_ORDER = ("ready", "selective", "delayed")
LANES = {"censored": 0, "selected": 1, "ready": 2}
VARIANT_OFFSET = {"ready": 0.16, "selective": 0.0, "delayed": -0.16}
MARKERS = {"ready": "o", "selected": "s", "censored": "X"}
LOG_FLOOR_DISPLAY = -8.0

CAPTION_LINES = (
    "Expected valid pattern: the ready stream is visibly separate and labelled as the "
    "validity reference. Delayed and selective variants retain false-alarm and "
    "detection-delay observations but never inherit a validity claim.",
    "Falsifier pattern: delayed/selected behavior indistinguishable from the ready "
    "reference is evidence against C6; a claimed guarantee for non-ready variants is "
    "invalid regardless of the plotted rates.",
)
FLOOR_NOTE = (
    "Mid-panel log-wealth paths are clipped and floored at log(1e-12) before display."
)


def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the NB3 figure and sidecar from the monitoring experiment."""
    result = run_monitoring_experiment(config)
    rows = _rows(result)
    sidecar = write_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = _render(result, out_dir)
    print("\n".join((*CAPTION_LINES, FLOOR_NOTE)))
    return {
        "figure": str(figure),
        "sidecar": str(sidecar),
        "rows": len(rows),
        "name": NAME,
    }


def _summaries(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summaries = {"ready": dict(result["ready_label_null"])}
    for row in result["out_of_guarantee_variants"]:
        summaries[str(row["variant"])] = dict(row)
    return summaries


def _rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in result["label_status_sequence"]:
        rows.append(
            {
                "panel": "pre",
                "update_index": int(event["update_index"]),
                "label_status": str(event["label_status"]),
                "variant": str(event["variant"]),
                "event_index": int(event["event_index"]),
            }
        )
    for path in result["wealth_paths"]:
        rows.append(
            {
                "panel": "mid",
                "update_index": int(path["update_index"]),
                "variant": str(path["variant"]),
                "log_wealth": float(path["log_wealth"]),
                "threshold_c": float(path["threshold_c"]),
                "stream": int(path["stream"]),
            }
        )
    summaries = _summaries(result)
    delays = {
        str(row["variant"]): float(row["detection_delay"])
        for row in result["detection_delay_by_variant"]
    }
    for variant in VARIANT_ORDER:
        summary = summaries[variant]
        interval = summary["false_alarm_binomial_interval"]
        rows.append(
            {
                "panel": "post",
                "variant": variant,
                "false_alarm_incidence": float(summary["empirical_false_alarm_rate"]),
                "interval_low": float(interval["low"]),
                "interval_high": float(interval["high"]),
                "detection_delay": float(delays[variant]),
                "threshold_c": float(summary["threshold_c"]),
                "validity_claim": str(summary["validity_claim"]),
            }
        )
    return rows


def _render(result: Mapping[str, Any], out_dir: str | None) -> Any:
    with plt.rc_context({"font.size": 9, "axes.titleweight": "bold", "figure.dpi": 150}):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
        _pre_panel(axes[0], result["label_status_sequence"])
        _mid_panel(
            axes[1],
            result["wealth_paths"],
            float(result["ready_label_null"]["threshold_c"]),
        )
        _post_panel(axes[2], result)
        fig.text(
            0.5,
            0.012,
            "\n".join(
                textwrap.fill(line, 190) for line in (*CAPTION_LINES, FLOOR_NOTE)
            ),
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#333333",
        )
        fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.24, wspace=0.30)
        path = save_figure(fig, NAME, out_dir=out_dir)
        plt.close(fig)
    return path


def _pre_panel(ax: plt.Axes, events: list[dict[str, Any]]) -> None:
    for status in ("ready", "selected", "censored"):
        subset = [event for event in events if event["label_status"] == status]
        if not subset:
            continue
        ax.scatter(
            [int(event["update_index"]) for event in subset],
            [
                LANES[status] + VARIANT_OFFSET[str(event["variant"])]
                for event in subset
            ],
            s=70,
            marker=MARKERS[status],
            c=[PALETTE[str(event["variant"])] for event in subset],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
    ax.set(
        title="Pre: label arrival status",
        xlabel="update index",
        ylabel="arrival lane",
        yticks=[0, 1, 2],
        yticklabels=["censored", "selected", "ready"],
        ylim=(-0.6, 2.6),
    )
    handles = [
        Line2D([], [], marker=MARKERS[status], color="#555555", linestyle="", markersize=7, label=status)
        for status in ("ready", "selected", "censored")
    ]
    handles += [
        Line2D([], [], marker="o", color=PALETTE[variant], linestyle="", markersize=7, label=variant)
        for variant in VARIANT_ORDER
    ]
    ax.legend(handles=handles, fontsize=6.5, ncol=2, loc="center right", framealpha=0.9)


def _mid_panel(ax: plt.Axes, wealth_paths: list[dict[str, Any]], threshold_c: float) -> None:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for path in wealth_paths:
        key = (str(path["variant"]), int(path["stream"]))
        grouped.setdefault(key, []).append(
            (int(path["update_index"]), float(path["log_wealth"]))
        )
    all_logs = [value for points in grouped.values() for _, value in points]
    y_min = max(LOG_FLOOR_DISPLAY, min(all_logs) - 0.2)
    y_max = max(5.0, max(all_logs) + 0.4)
    for variant in VARIANT_ORDER:
        streams = sorted(stream for (v, stream) in grouped if v == variant)
        for index, stream in enumerate(streams):
            points = sorted(grouped[(variant, stream)])
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            if index == 0:
                ax.plot(
                    xs,
                    ys,
                    color=PALETTE[variant],
                    linewidth=1.4,
                    marker="o",
                    markersize=2.5,
                    markevery=10,
                    label=variant,
                    zorder=3,
                )
            else:
                ax.plot(xs, ys, color=PALETTE[variant], linewidth=0.6, alpha=0.3, zorder=1)
    ax.axhline(
        math.log(threshold_c),
        color="#333333",
        linestyle="--",
        linewidth=1,
        label=f"log threshold (c={threshold_c:g})",
    )
    ax.set(
        title="Mid: WATCH log-wealth paths",
        xlabel="update index",
        ylabel="log wealth (floored at log 1e-12)",
        ylim=(y_min, y_max),
    )
    ax.legend(fontsize=6.5, loc="upper left", ncol=2, framealpha=0.9)


def _post_panel(ax: plt.Axes, result: Mapping[str, Any]) -> None:
    summaries = _summaries(result)
    delays = {
        str(row["variant"]): float(row["detection_delay"])
        for row in result["detection_delay_by_variant"]
    }
    threshold_c = float(result["ready_label_null"]["threshold_c"])
    for index, variant in enumerate(VARIANT_ORDER):
        summary = summaries[variant]
        rate = float(summary["empirical_false_alarm_rate"])
        interval = summary["false_alarm_binomial_interval"]
        low = float(interval["low"])
        high = float(interval["high"])
        delay = delays[variant]
        claim = "validity reference" if variant == "ready" else "no validity claim"
        ax.errorbar(
            rate,
            index,
            xerr=[[max(0.0, rate - low)], [max(0.0, high - rate)]],
            fmt="o",
            color=PALETTE[variant],
            capsize=3,
            markersize=6,
            label=f"{variant} ({claim}); delay={delay:.1f}",
            zorder=3,
        )
        ax.annotate(
            f"delay={delay:.1f}",
            xy=(rate, index),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=7,
            color=PALETTE[variant],
        )
    ax.axvline(
        1.0 / threshold_c,
        color="#333333",
        linestyle=":",
        linewidth=1,
        label=f"1/threshold = {1.0 / threshold_c:.2f}",
    )
    ax.set(
        title="Post: false alarms and detection delay",
        xlabel="false-alarm incidence",
        yticks=[0, 1, 2],
        yticklabels=list(VARIANT_ORDER),
        ylim=(-0.7, 2.7),
    )
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
