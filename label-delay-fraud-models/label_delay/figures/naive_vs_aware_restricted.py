"""Naive versus aware recall at 1% FPR, mechanism-restricted.

Every number is measured on this run's own audit population.

The study (:func:`label_delay.policies.run_naive_vs_aware_mechanism_study`) fits the
audit-side ``eta(x)`` (latent-fraud probability) and ``e(x)`` (label-observation
probability among the latent positives), restricts to the Cannings-violating set
``eta(x) >= 1/2`` and ``e(x) < 1/(2 eta(x))``, and trains the naive and the aware
model on ``n_seeds`` shared draws. Each panel draws the mean recall at 1% FPR
over the seeds with the paired 95% t interval; the ``paired_diff`` row carries
the mean of (naive - aware) and the one-sided 95% upper confidence bound,
computed with the paired-t closed form. The check
is directional: on the restricted panel the aware dot must sit above the naive
dot with that upper bound at or below zero, and the restricted-subset row must
carry at least 1,000 confirmed frauds; the whole-population panel is contrast
only and nothing about it is asserted.
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

from label_delay.config import config_hash
from label_delay.policies import run_naive_vs_aware_mechanism_study
from label_delay.vizlib import ROOT, save_study_figure, write_study_sidecar

NAME = "naive_vs_aware_restricted"
CAPTION_LINES = [
    'each panel shows mean recall at 1% FPR for the naive and the aware model with paired 95% t intervals over seeds; the restricted panel keeps only the subset where eta(x) >= 1/2 and e(x) < 1/(2 eta(x)), the whole-population panel is contrast only.',
    'On the restricted subset the aware dot should sit above the naive dot, with the one-sided upper bound on the naive-minus-aware difference at or below zero.',
]

BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
INK, GRID = "#222222", "#E4E4E4"
INTERVAL_LEVEL = 0.95

#: Panel id, partition key and title; the restricted panel carries the falsifier.
PANELS = (
    (
        "restricted",
        "cannings_subset",
        "restricted: eta(x) >= 1/2 and e(x) < 1/(2 eta(x))",
    ),
    ("unrestricted", "whole_population", "whole population (contrast only)"),
)
SERIES_COLORS = {"naive": VERM, "aware": BLUE}

_CACHE: dict[str, dict[str, Any]] = {}


# ------------------------------------------------------------------ data
def _caption_lines() -> list[str]:
    return list(CAPTION_LINES)


def _mean_interval(values: Any) -> tuple[float, float, float]:
    """Mean and the paired two-sided t interval over seeds; degenerate-safe."""
    array = np.asarray(values, dtype=float).reshape(-1)
    n = int(array.size)
    mean = float(array.mean()) if n else 0.0
    if n < 2:
        return mean, mean, mean
    half = float(_stats.t.ppf(0.5 + INTERVAL_LEVEL / 2.0, n - 1)) * float(array.std(ddof=1)) / math.sqrt(n)
    return mean, mean - half, mean + half


def _one_sided_upper(diffs: Any) -> float:
    """One-sided 95% upper t bound on the mean of ``diffs`` (the paired-t closed form)."""
    values = np.asarray(diffs, dtype=float).reshape(-1)
    n = int(values.size)
    if n < 2:
        raise ValueError("a paired one-sided bound needs at least two seeds")
    mean = float(values.mean())
    half = float(_stats.t.ppf(INTERVAL_LEVEL, n - 1)) * float(values.std(ddof=1)) / math.sqrt(n)
    return mean + half


def _study(config: Mapping[str, Any]) -> dict[str, Any]:
    key = config_hash(config)
    if key not in _CACHE:
        _CACHE[key] = run_naive_vs_aware_mechanism_study(config)
    return _CACHE[key]


def _rows(study: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "n_seeds": int(study["n_seeds"]),
        "fpr_target": float(study["fpr_target"]),
        "panels": {},
    }
    for panel, part_key, _title in PANELS:
        part = study[part_key]
        per_seed = list(part["per_seed"])
        naive = np.asarray([float(row["naive_recall_at_1pct_fpr"]) for row in per_seed])
        aware = np.asarray([float(row["aware_recall_at_1pct_fpr"]) for row in per_seed])
        arm_stats: dict[str, tuple[float, float, float]] = {}
        for series, values, x in (("naive", naive, 0.0), ("aware", aware, 1.0)):
            mean, low, high = _mean_interval(values)
            arm_stats[series] = (mean, low, high)
            rows.append(
                {
                    "panel": panel,
                    "series": series,
                    "group": "recall_at_1pct_fpr",
                    "x": x,
                    "y": round(mean, 8),
                    "y_low": round(low, 8),
                    "y_high": round(high, 8),
                    "n": int(values.size),
                }
            )
        diffs = naive - aware
        mean_diff = float(diffs.mean())
        upper = _one_sided_upper(diffs)
        reported = float(part["naive_minus_aware"]["one_sided_upper_95"])
        if abs(upper - reported) > 1e-9:
            raise AssertionError(
                f"{NAME}: recomputed one-sided bound {upper:.12g} disagrees with the study's {reported:.12g}"
            )
        n_confirmed = int(part["n_confirmed_frauds"])
        rows.append(
            {
                "panel": panel,
                "series": "paired_diff",
                "group": "",
                "y": round(mean_diff, 8),
                "y_high": round(upper, 8),
                "n": int(diffs.size),
                "label": f"n_confirmed_frauds={n_confirmed}" if panel == "restricted" else "",
            }
        )
        summary["panels"][panel] = {
            "n_instances": int(part["n_instances"]),
            "n_confirmed_frauds": n_confirmed,
            "naive": arm_stats["naive"][0],
            "naive_low": arm_stats["naive"][1],
            "naive_high": arm_stats["naive"][2],
            "aware": arm_stats["aware"][0],
            "aware_low": arm_stats["aware"][1],
            "aware_high": arm_stats["aware"][2],
            "mean_diff": mean_diff,
            "upper": upper,
        }
    return rows, summary


# ------------------------------------------------------------------ figure
def _style(ax: Any) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def _head(ax: Any, text: str, handles: list[Any], ncol: int = 2) -> None:
    ax.set_title(text, loc="left", fontsize=10.5, fontweight="bold", pad=29)
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=ncol,
        frameon=False,
        fontsize=8.7,
        borderaxespad=0.1,
        handlelength=1.8,
        columnspacing=1.1,
        handletextpad=0.5,
    )


def _marker(label: str, color: str, ms: float = 7) -> Line2D:
    return Line2D(
        [0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor=color, ms=ms, label=label
    )


def _caption_text() -> str:
    return "\n".join(
        textwrap.fill(("How to read this chart: " if index == 0 else "") + line, width=235)
        for index, line in enumerate(_caption_lines())
    )


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

    arms = [row for row in rows if row["series"] in ("naive", "aware")]
    low = min(float(row["y_low"]) for row in arms)
    high = max(float(row["y_high"]) for row in arms)
    span = max(high - low, 1e-9)

    width, height = 20.0, 6.6
    fig = plt.figure(figsize=(width, height), layout="constrained")
    shell = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.17])
    caption = fig.add_subplot(shell[1])
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
        f"naive versus aware recall at 1% FPR, {summary['n_seeds']} seeds per panel",
        x=0.0,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    grid = shell[0].subgridspec(1, 2, wspace=0.14)

    axes: list[Any] = []
    for index, (panel, _part_key, title) in enumerate(PANELS):
        ax = fig.add_subplot(grid[index], sharey=axes[0] if axes else None)
        axes.append(ax)
        for name in ("naive", "aware"):
            row = series(panel, name)[0]
            y = float(row["y"])
            ax.errorbar(
                [row["x"]],
                [y],
                yerr=[[y - float(row["y_low"])], [float(row["y_high"]) - y]],
                fmt="o",
                ms=7,
                lw=2.2,
                capsize=5,
                color=SERIES_COLORS[name],
            )
        naive_y = float(series(panel, "naive")[0]["y"])
        aware_y = float(series(panel, "aware")[0]["y"])
        midpoint = 0.5 * (naive_y + aware_y)
        ax.plot([0.0, 0.5], [naive_y, midpoint], color=VERM, lw=2.0, alpha=0.8, zorder=1.5)
        ax.plot([0.5, 1.0], [midpoint, aware_y], color=BLUE, lw=2.0, alpha=0.8, zorder=1.5)
        panel_summary = summary["panels"][panel]
        upper = panel_summary["upper"]
        direction = (
            "direction: aware above naive (upper bound <= 0)"
            if upper <= 0
            else "direction: not established (upper bound > 0)"
        )
        lines = [
            f"mean(naive - aware) = {panel_summary['mean_diff']:+.4f}",
            f"one-sided 95% upper bound = {upper:+.4f}",
        ]
        if panel == "restricted":
            lines.append(
                f"n_confirmed_frauds = {panel_summary['n_confirmed_frauds']} "
                f"(of {panel_summary['n_instances']} instances)"
            )
        lines.append(direction)
        ax.text(
            0.5,
            0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.6,
            linespacing=1.3,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=GREY, linewidth=0.8, alpha=0.95),
        )
        ax.set_xticks([0.0, 1.0], ["naive", "aware"])
        for tick, name in zip(ax.get_xticklabels(), ("naive", "aware")):
            tick.set_color(SERIES_COLORS[name])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(low - 0.55 * span, high + 0.10 * span)
        _style(ax)
        _head(ax, title, [_marker("naive", VERM), _marker("aware", BLUE)], 2)
    axes[0].set_ylabel("recall at 1% FPR (paired 95% t interval)")
    return fig


# ------------------------------------------------------------------ entry
def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the naive_vs_aware_restricted sidecar and figure."""
    study = _study(config)
    rows, summary = _rows(study)
    sidecar = write_study_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = save_study_figure(_figure(rows, summary), NAME, out_dir=out_dir)
    plt.close()
    lines = _caption_lines()
    print(f"{NAME}: {len(rows)} rows -> {sidecar.name} and {figure.name}")
    print("How to read this chart: " + lines[0])
    print(lines[1])
    for panel, _part_key, _title in PANELS:
        values = summary["panels"][panel]
        print(
            f"{panel}: naive {values['naive']:.4f} [{values['naive_low']:.4f}, {values['naive_high']:.4f}], "
            f"aware {values['aware']:.4f} [{values['aware_low']:.4f}, {values['aware_high']:.4f}], "
            f"mean(naive - aware) {values['mean_diff']:+.4f}, one-sided 95% upper bound "
            f"{values['upper']:+.4f}, n_confirmed_frauds {values['n_confirmed_frauds']}, "
            f"n_instances {values['n_instances']}"
        )
    return {"figure": str(figure), "sidecar": str(sidecar), "rows": int(len(rows)), "name": NAME}
