"""SAR against an independent truth and the Coudray Eq. 15 bound.

Every row is measured on this run's own synthetic populations.

* draws one population with a known instance-dependent labelling
  propensity ``e(x) = e_m + (1 - e_m) sigmoid(a x)`` and a latent positive rate
  ``sigmoid(b0 + b1 x)``. The panel shows the true ``e(x)`` (rising, floored at
  ``e_m``), the observed labelled share per x bin and the latent versus
  observed-positive rate: the observed rate is ``P(Y = 1, L = 1 | x)``, so the
  gap is exactly the selection distortion that the SAR weights undo.
* runs the pure-PU SAR risk of a fixed rule on ``n_replicates``
  replicates with the correct propensity and with the propensity halved. The
  independent truth ``R(g)`` is Monte-Carlo over a separate 400,000-point
  sample of the same population, and the band is ``truth +/- 3 SE`` of the
  Monte-Carlo mean of the correct-propensity series, as is standard for a Monte-Carlo mean.
* reuses the excess-risk study (``run_excess_risk_study``)
  over a (n, e_m) grid and ``fit_excess_ratio_trend`` for the trend interval.
  The heat-map ``z`` is the cell's mean excess risk divided by the Eq. 15
  expression at ``kappa_1 = 1``; the ``kappa_hat`` row carries the study's
  fitted constant with a bootstrap interval on the max-over-cells rule and the
  trend slopes in its label.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from label_delay.config import seed_of
from label_delay.recovery import fit_excess_ratio_trend, run_excess_risk_study
from label_delay.vizlib import ROOT, save_study_figure, write_study_sidecar

NAME = "sar_truth_and_bound"
CAPTION_LINES = [
    'the left panel shows the true labelling propensity e(x) rising from its floor e_m, so positives are observed far more often at high x and the observed positive rate sits below the latent rate; the middle panel compares the SAR risk with the correct propensity (inside the Monte-Carlo band around the independent truth) against a halved propensity (outside it); the right panel shows measured excess risk relative to the Eq. 15 form, with the fitted constant kappa_hat.',
    'A lower propensity floor widens the bound and the measured excess follows it. Everything here is synthetic and says nothing about a real labelling process.',
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
BLUES3 = ["#9ECAE1", "#4292C6", "#08519C"]
INK, GRID = "#222222", "#E4E4E4"

PROPENSITY_FLOOR = 0.15
PROPENSITY_SLOPE = 1.6
PROPENSITY_GRID = (-2.5, 2.5, 25)
PROPENSITY_BINS = 10
DEFAULT_PROPENSITY_INSTANCES = 60_000
LATENT_LOGIT_INTERCEPT = -1.0
LATENT_LOGIT_SLOPE = 1.1

FLOOR = 0.15
MARGIN_H = 0.2
SCORE_THRESHOLD = 0.3
DEFAULT_TRANSACTIONS = 20_000
TRUTH_SAMPLE = 400_000

DEFAULT_N_VALUES = (200, 800, 3200, 12800)
DEFAULT_E_M_VALUES = (0.1, 0.3, 0.9)
TREND_INTERVAL_LEVEL = 0.95


def _sigmoid(value: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=float)))


def _propensity(x: Any, floor: float = PROPENSITY_FLOOR) -> Any:
    return floor + (1.0 - floor) * _sigmoid(PROPENSITY_SLOPE * np.asarray(x, dtype=float))


def _d2_margin(x1: np.ndarray) -> np.ndarray:
    return MARGIN_H + (1.0 - MARGIN_H) * np.abs(x1 / (1.0 + np.abs(x1)))


def _d2_propensity(x2: np.ndarray, floor: float = FLOOR) -> np.ndarray:
    return floor + (1.0 - floor) * _sigmoid(x2)


def _d2_risk(score: np.ndarray, propensity: np.ndarray, observed: np.ndarray) -> float:
    ratio = observed / propensity
    decision = (score > SCORE_THRESHOLD).astype(float)
    return float(np.mean(np.where(decision >= 0.5, 1.0 - ratio, ratio)))


def _d1_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    seed = seed_of(config)
    size = int(config.get("n_propensity_instances", config.get("stream_n", DEFAULT_PROPENSITY_INSTANCES)))
    rng = np.random.default_rng([seed, 501])
    x = rng.standard_normal(size)
    propensity = _propensity(x)
    latent_probability = _sigmoid(LATENT_LOGIT_INTERCEPT + LATENT_LOGIT_SLOPE * x)
    latent = (rng.random(size) < latent_probability).astype(float)
    labelled = rng.random(size) < propensity

    low, high, points = PROPENSITY_GRID
    rows: list[dict[str, Any]] = []
    for grid_x in np.linspace(low, high, points):
        rows.append({"panel": "propensity", "series": "e_true", "x": float(grid_x), "y": float(_propensity(grid_x))})
    for edge in (low, high):
        rows.append(
            {
                "panel": "propensity",
                "series": "e_m_floor",
                "x": float(edge),
                "y": float(PROPENSITY_FLOOR),
            }
        )
    edges = np.quantile(x, np.linspace(0.0, 1.0, PROPENSITY_BINS + 1))
    bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, PROPENSITY_BINS - 1)
    width = 0.9 * float(np.median(np.diff(edges)))
    for index in range(PROPENSITY_BINS):
        mask = bins == index
        centre = float(x[mask].mean())
        count = int(mask.sum())
        rows.append(
            {
                "panel": "propensity",
                "series": "labeled_fraction",
                "x": centre,
                "y": float(labelled[mask].mean()),
                "n": count,
            }
        )
        rows.append(
            {
                "panel": "positive_rates",
                "series": "latent_positive_rate",
                "x": centre,
                "y": float(latent[mask].mean()),
                "n": count,
            }
        )
        rows.append(
            {
                "panel": "positive_rates",
                "series": "observed_positive_rate",
                "x": centre,
                "y": float((latent * labelled)[mask].mean()),
                "n": count,
            }
        )
    summary = {
        "instances": size,
        "e_min": float(propensity.min()),
        "e_max": float(propensity.max()),
        "labelled_share": float(labelled.mean()),
        "latent_share": float(latent.mean()),
        "observed_positive_share": float((latent * labelled).mean()),
        "bar_width": width,
    }
    return rows, summary


def _d2_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    seed = seed_of(config)
    replicates = int(config.get("n_replicates", 100))
    transactions = int(config.get("n_d2_transactions", DEFAULT_TRANSACTIONS))
    correct: list[float] = []
    halved: list[float] = []
    for replicate in range(replicates):
        rng = np.random.default_rng([seed, 601, replicate])
        x1 = rng.standard_normal(transactions)
        x2 = rng.standard_normal(transactions)
        eta = 0.5 + 0.5 * _d2_margin(x1) * np.sign(x1)
        latent = (rng.random(transactions) < eta).astype(float)
        propensity = _d2_propensity(x2)
        observed = latent * (rng.random(transactions) < propensity).astype(float)
        correct.append(_d2_risk(x1, propensity, observed))
        halved.append(_d2_risk(x1, propensity / 2.0, observed))

    truth_rng = np.random.default_rng([seed, 602])
    truth_x1 = truth_rng.standard_normal(TRUTH_SAMPLE)
    truth_eta = 0.5 + 0.5 * _d2_margin(truth_x1) * np.sign(truth_x1)
    truth_y = (truth_rng.random(TRUTH_SAMPLE) < truth_eta).astype(float)
    truth_risk = float(np.mean(truth_y != (truth_x1 > SCORE_THRESHOLD).astype(float)))

    correct_array = np.asarray(correct, dtype=float)
    halved_array = np.asarray(halved, dtype=float)
    sem = float(correct_array.std(ddof=1) / math.sqrt(correct_array.size))
    band = 3.0 * sem

    rows: list[dict[str, Any]] = []
    for replicate, value in enumerate(correct):
        rows.append(
            {
                "panel": "unbiasedness",
                "series": "correct_propensity",
                "x": float(replicate),
                "y": float(value),
                "n": replicates,
            }
        )
    for replicate, value in enumerate(halved):
        rows.append(
            {
                "panel": "unbiasedness",
                "series": "halved_propensity",
                "x": float(replicate),
                "y": float(value),
                "n": replicates,
            }
        )
    rows.append(
        {
            "panel": "unbiasedness",
            "series": "truth",
            "x": 0.0,
            "y": truth_risk,
            "y_low": truth_risk - band,
            "y_high": truth_risk + band,
            "n": TRUTH_SAMPLE,
            "label": f"independent truth R(g), +/-3 SE ({band:.4f})",
        }
    )
    summary = {
        "replicates": replicates,
        "transactions": transactions,
        "truth_risk": truth_risk,
        "band": band,
        "correct_mean": float(correct_array.mean()),
        "halved_mean": float(halved_array.mean()),
        "correct_sd": float(correct_array.std(ddof=1)),
        "halved_sd": float(halved_array.std(ddof=1)),
    }
    return rows, summary


def _study_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(config),
        "n_values": [int(value) for value in config.get("n_values", DEFAULT_N_VALUES)],
        "e_m_values": [float(value) for value in config.get("e_m_values", DEFAULT_E_M_VALUES)],
        "n_replicates": int(config.get("n_replicates", 100)),
        "n_boot": int(config.get("n_boot", 500)),
        "level": float(config.get("level", TREND_INTERVAL_LEVEL)),
    }


def _kappa_interval(
    cells: list[Mapping[str, Any]], draws: int, seed: int
) -> tuple[float, float]:
    """Bootstrap interval for the max-over-cells mean-ratio rule that defines kappa_hat."""
    rng = np.random.default_rng([int(seed), 603])
    boot = np.empty(max(int(draws), 1), dtype=float)
    for draw in range(boot.size):
        ratios = []
        for cell in cells:
            values = np.asarray(cell["excess_risks"], dtype=float)
            resample = values[rng.integers(0, values.size, values.size)]
            ratios.append(float(resample.mean()) / float(cell["bound_kappa1_1"]))
        boot[draw] = max(ratios)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def _d3_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    seed = seed_of(config)
    study_config = _study_config(config)
    study = run_excess_risk_study(study_config)
    cells = list(study["cells"])
    trend = fit_excess_ratio_trend(
        cells,
        level=float(study_config["level"]),
        n_boot=int(study_config["n_boot"]),
        seed=seed,
    )
    kappa = float(study["kappa_hat"])
    boot_low, boot_high = _kappa_interval(cells, int(config.get("n_boot", 500)), seed)
    kappa_interval = (min(boot_low, kappa), max(boot_high, kappa))

    rows: list[dict[str, Any]] = []
    for cell in cells:
        rows.append(
            {
                "panel": "ratio_heatmap",
                "series": "ratio_grid",
                "x": float(cell["n"]),
                "y": float(cell["e_m"]),
                "z": float(cell["ratio"]),
                "n": int(cell["n_replicates"]),
            }
        )
    log_n = trend["log_n"]
    label = (
        f"kappa_hat {kappa:.2f} [{kappa_interval[0]:.2f}, {kappa_interval[1]:.2f}]; "
        f"trend log n {log_n['slope']:+.3f} [{log_n['low']:.3f}, {log_n['high']:.3f}]"
    )
    rows.append(
        {
            "panel": "ratio_heatmap",
            "series": "kappa_hat",
            "y": kappa,
            "y_low": kappa_interval[0],
            "y_high": kappa_interval[1],
            "n": int(config.get("n_boot", 500)),
            "label": label,
        }
    )
    for cell in cells:
        mean = float(cell["mean_excess_risk"])
        half = 1.96 * float(cell["se_excess_risk"])
        group = f"{float(cell['e_m']):g}"
        rows.append(
            {
                "panel": "excess_curves",
                "series": "excess_curve",
                "group": group,
                "x": float(cell["n"]),
                "y": mean,
                "y_low": max(mean - half, mean * 0.05),
                "y_high": mean + half,
                "n": int(cell["n_replicates"]),
            }
        )
        rows.append(
            {
                "panel": "excess_curves",
                "series": "bound_kappa1",
                "group": group,
                "x": float(cell["n"]),
                "y": float(cell["bound_kappa1_1"]),
                "n": int(cell["n_replicates"]),
            }
        )
    summary = {
        "kappa_hat": kappa,
        "kappa_low": kappa_interval[0],
        "kappa_high": kappa_interval[1],
        "log_n_slope": float(log_n["slope"]),
        "log_n_low": float(log_n["low"]),
        "log_n_high": float(log_n["high"]),
        "inv_e_m_slope": float(trend["inv_e_m"]["slope"]),
        "cells": len(cells),
    }
    return rows, summary


def _caption_lines() -> list[str]:
    return list(CAPTION_LINES)


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
    by = {}
    for row in rows:
        by.setdefault((row["panel"], row["series"]), []).append(row)

    def series(panel: str, name: str) -> list[dict[str, Any]]:
        return by.get((panel, name), [])

    def xy(panel: str, name: str) -> tuple[np.ndarray, np.ndarray]:
        data = series(panel, name)
        return (
            np.asarray([row["x"] for row in data], dtype=float),
            np.asarray([row["y"] for row in data], dtype=float),
        )

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
        width_ratios=[1.0, 0.82, 1.2],
        height_ratios=[1.0, 1.0],
        hspace=0.34,
        wspace=0.16,
    )

    left = top[:, 0].subgridspec(2, 1, hspace=0.22)
    ax_e = fig.add_subplot(left[0])
    ax_r = fig.add_subplot(left[1])

    ex, ey = xy("propensity", "e_true")
    fx, fy = xy("propensity", "e_m_floor")
    bx, byv = xy("propensity", "labeled_fraction")
    bar_width = float(np.median(np.diff(bx))) * 0.9 if bx.size > 1 else 0.3
    ax_e.bar(bx, byv, width=bar_width, color=SKY, alpha=0.85, label="labelled fraction per bin")
    ax_e.plot(ex, ey, color=BLUE, lw=2.2, label="true e(x)")
    ax_e.plot(fx, fy, color=GREY, ls="--", lw=1.5, label="floor e_m")
    ax_e.set(xlabel="covariate x (standardised)", ylabel="labelling propensity", ylim=(0.0, 1.02))
    _style(ax_e)
    _head(
        ax_e,
        "e(x) varies with x and never falls below e_m",
        [
            Patch(color=SKY, alpha=0.85, label="labelled fraction per bin"),
            _line("true e(x)", BLUE, lw=2.2),
            _line("floor e_m", GREY, "--", 1.5),
        ],
        3,
    )

    lx, ly = xy("positive_rates", "latent_positive_rate")
    ox, oy = xy("positive_rates", "observed_positive_rate")
    ax_r.plot(lx, ly, color=BLUE, marker="o", ms=5, lw=1.8, label="latent P(Y = 1 | x bin)")
    ax_r.plot(ox, oy, color=VERM, marker="s", ms=5, lw=1.8, label="observed P(Y = 1 and labelled | x bin)")
    ax_r.set(xlabel="covariate x bin centre (standardised)", ylabel="positive rate")
    _style(ax_r)
    _head(
        ax_r,
        "label selection pulls the observed positive rate down",
        [_line("latent positive rate", BLUE, lw=1.8), _line("observed (positive and labelled)", VERM, lw=1.8)],
        2,
    )

    ax_d2 = fig.add_subplot(top[:, 1])
    colours = {"correct_propensity": BLUE, "halved_propensity": VERM}
    labels = {"correct_propensity": "correct propensity", "halved_propensity": "halved propensity (sabotage)"}
    for index, name in enumerate(("correct_propensity", "halved_propensity")):
        _, values = xy("unbiasedness", name)
        violin = ax_d2.violinplot([values], positions=[index], widths=0.72, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(colours[name])
            body.set_alpha(0.32)
            body.set_edgecolor(colours[name])
        jitter = np.linspace(-0.13, 0.13, values.size)
        ax_d2.scatter(index + jitter, values, s=11, color=colours[name], alpha=0.5, linewidths=0)
        ax_d2.scatter([index], [values.mean()], s=60, color="white", edgecolor=colours[name], linewidth=2, zorder=5)
    truth = series("unbiasedness", "truth")[0]
    ax_d2.axhspan(truth["y_low"], truth["y_high"], color=GREY, alpha=0.4, lw=0)
    ax_d2.axhline(truth["y"], color=INK, lw=1.2, ls="--")
    ax_d2.annotate(
        f"truth {truth['y']:.4f} +/- 3 SE ({truth['y_low']:.4f}, {truth['y_high']:.4f})",
        xy=(0.5, truth["y"]),
        xycoords=("axes fraction", "data"),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8.5,
        color=INK,
    )
    span = np.concatenate([xy("unbiasedness", "correct_propensity")[1], xy("unbiasedness", "halved_propensity")[1]])
    low = min(float(span.min()), float(truth["y_low"]))
    high = max(float(span.max()), float(truth["y_high"]))
    pad = 0.06 * (high - low)
    ax_d2.set_ylim(low - pad, high + pad)
    ax_d2.set_xticks([0, 1], ["correct\npropensity", "halved propensity\n(sabotage)"])
    ax_d2.set_xlim(-0.62, 1.62)
    ax_d2.set(ylabel="SAR risk estimate (probability of error)")
    _style(ax_d2)
    _head(
        ax_d2,
        "SAR against an independent large-sample truth",
        [
            _line("independent truth R(g)", INK, "--"),
            Patch(color=GREY, alpha=0.4, label="truth +/- 3 SE"),
            Line2D([0], [0], marker="o", color="w", markeredgecolor=INK, markerfacecolor="white", ms=7, label="replicate mean"),
        ],
        3,
    )

    right = top[:, 2].subgridspec(2, 1, hspace=0.42)
    ax_heat = fig.add_subplot(right[0])
    ax_curve = fig.add_subplot(right[1])

    grid = series("ratio_heatmap", "ratio_grid")
    ns = sorted({float(row["x"]) for row in grid})
    ems = sorted({float(row["y"]) for row in grid})
    z = np.full((len(ems), len(ns)), np.nan, dtype=float)
    for row in grid:
        z[ems.index(float(row["y"])), ns.index(float(row["x"]))] = float(row["z"])
    ax_heat.set_xscale("log")
    mesh = ax_heat.pcolormesh(
        ns,
        ems,
        z,
        cmap="Blues",
        shading="nearest",
        vmin=0.0,
        vmax=float(np.nanmax(z)) * 1.25,
    )
    for (row_index, column_index), ratio in np.ndenumerate(z):
        ax_heat.text(ns[column_index], ems[row_index], f"{ratio:.2f}", ha="center", va="center", fontsize=8, color=INK)
    ax_heat.set_xticks(ns)
    ax_heat.set_xticklabels([f"{int(value)}" for value in ns])
    ax_heat.set_xlim(ns[0] / 1.7, ns[-1] * 1.7)
    ax_heat.set_yticks(ems)
    ax_heat.set_yticklabels([f"{value:g}" for value in ems])
    ax_heat.set(xlabel="training size n (log axis)", ylabel="propensity floor e_m")
    ax_heat.spines[:].set_visible(False)
    fig.colorbar(mesh, ax=ax_heat, pad=0.02, label="mean excess / Eq. 15 at kappa_1 = 1")
    kappa = series("ratio_heatmap", "kappa_hat")[0]
    _head(ax_heat, "measured excess over the Eq. 15 form\n" + kappa["label"])

    groups = []
    for row in series("excess_curves", "excess_curve"):
        if row["group"] not in groups:
            groups.append(row["group"])
    handles = []
    for group, colour in zip(groups, BLUES3):
        measured = [row for row in series("excess_curves", "excess_curve") if row["group"] == group]
        bound = [row for row in series("excess_curves", "bound_kappa1") if row["group"] == group]
        xs = np.asarray([row["x"] for row in measured], dtype=float)
        ys = np.asarray([row["y"] for row in measured], dtype=float)
        lows = np.asarray([row["y_low"] for row in measured], dtype=float)
        highs = np.asarray([row["y_high"] for row in measured], dtype=float)
        ax_curve.plot(xs, ys, color=colour, lw=2.0, marker="o", ms=4.5)
        ax_curve.fill_between(xs, lows, highs, color=colour, alpha=0.25, lw=0)
        ax_curve.plot(
            np.asarray([row["x"] for row in bound], dtype=float),
            np.asarray([row["y"] for row in bound], dtype=float),
            color=colour,
            lw=1.5,
            ls=":",
        )
        handles.append(_line(f"e_m = {group}", colour, lw=2.0))
    handles.append(_line("Eq. 15 bound at kappa_1 = 1", GREY, ":"))
    ax_curve.set(xscale="log", yscale="log", xlabel="training size n (log axis)", ylabel="mean excess risk (log axis)")
    ax_curve.set_xticks(ns)
    ax_curve.set_xticklabels([f"{int(value)}" for value in ns])
    ax_curve.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(ax_curve, "both")
    _head(ax_curve, "mean excess risk against the Eq. 15 bound", handles, 4)

    return fig


def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the sar_truth_and_bound sidecar and figure."""
    d1_rows, d1_summary = _d1_rows(config)
    d2_rows, d2_summary = _d2_rows(config)
    d3_rows, d3_summary = _d3_rows(config)
    rows = [*d1_rows, *d2_rows, *d3_rows]

    sidecar = write_study_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = save_study_figure(_figure(rows), NAME, out_dir=out_dir)
    lines = _caption_lines()
    print(f"{NAME}: {len(rows)} rows -> {sidecar.name} and {figure.name}")
    print("How to read this chart: " + lines[0])
    print(lines[1])
    print(
        "instances={instances} e range [{e_min:.3f}, {e_max:.3f}] latent={latent_share:.3f} "
        "labelled={labelled_share:.3f} observed-positive={observed_positive_share:.3f}".format(**d1_summary)
    )
    print(
        "truth={truth_risk:.4f} band=+/-{band:.4f} correct mean={correct_mean:.4f} (sd {correct_sd:.4f}) "
        "halved mean={halved_mean:.4f} (sd {halved_sd:.4f})".format(**d2_summary)
    )
    print(
        "kappa_hat={kappa_hat:.4f} [{kappa_low:.4f}, {kappa_high:.4f}] log-n slope={log_n_slope:+.4f} "
        "[{log_n_low:.4f}, {log_n_high:.4f}] inv-e_m slope={inv_e_m_slope:+.4f} cells={cells}".format(**d3_summary)
    )
    return {"figure": figure, "sidecar": sidecar, "rows": len(rows), "name": NAME}
