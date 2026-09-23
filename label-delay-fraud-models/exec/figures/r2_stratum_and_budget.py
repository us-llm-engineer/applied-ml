"""E7--E8 of NB3: SAR per propensity stratum and the label-budget policy comparison.

Exec-side renderer for ``viz/spec/R2_NB3_stratum_and_budget.md`` (claims C20 and
C21). Every number is measured on this run's own shared streams; no mock value is
read or reused.

* E7 (mid) reuses the frozen C20 shared-stream study
  (:func:`exec.recovery.run_shared_stream_study`): the deployed score's SAR risk
  is bootstrapped per propensity stratum on the canonical 60k stream. The bias
  panel shows each stratum's signed bias (mean bootstrap risk minus the
  stratum's independent truth risk) with the interval over the ``n_replicates``
  bootstrap draws against the zero line and the dashed pooled bias; the spread
  panel shows the per-stratum standard deviation of R_SAR on its own axis.
* E8 (post) reuses the frozen C21 replicated policy experiment
  (:func:`exec.policies.run_replicated_policy_experiment`): four label-buying
  policies at five equal-cost budgets over ``n_seeds`` streams, with the same
  aware trainer and the same seeds per cell. The top panel draws recall at 1%
  FPR against the number of labels bought (mean and 95% t band over seeds, direct
  labels at the right end); the bottom panel draws each policy minus random on
  the same seeds with a zero line. Comparative only: no policy is declared
  optimal.
"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from exec.config import seed_of
from exec.policies import run_replicated_policy_experiment
from exec.recovery import run_shared_stream_study
from exec.vizlib import ROOT, save_r2_figure, write_r2_sidecar

NAME = "R2_NB3_stratum_and_budget"
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
Z_95 = 1.96

#: One colour per policy, held fixed across every panel of the figure.
POLICY_COLORS = {
    "random": GREY,
    "risk_first": BLUE,
    "uncertainty_first": PURPLE,
    "diverse_typology": GREEN,
}
POLICY_LABELS = {
    "random": "random",
    "risk_first": "risk-first",
    "uncertainty_first": "uncertainty-first",
    "diverse_typology": "diverse-typology",
}

_CACHE: dict[str, dict[str, Any]] = {}


# ------------------------------------------------------------------ data
def _caption_lines() -> list[str]:
    return [
        line[2:].strip()
        for line in SPEC_PATH.read_text(encoding="utf-8").splitlines()
        if line.startswith("> ")
    ]


def _stratum_rows(study: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata = list(study["sar_by_propensity_stratum"])
    replicates = int(study["n_replicates"])
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"strata": len(strata), "replicates": replicates}
    for index, stratum in enumerate(strata):
        standard_error = math.sqrt(max(float(stratum["variance"]), 0.0) / max(replicates, 1))
        half = Z_95 * standard_error
        group = f"stratum_{index + 1}"
        label = f"[{float(stratum['propensity_low']):.3f}, {float(stratum['propensity_high']):.3f}]"
        rows.append(
            {
                "panel": "E7_sar_bias",
                "series": "sar_bias",
                "group": group,
                "x": int(index + 1),
                "y": float(stratum["bias"]),
                "y_low": float(stratum["bias"]) - half,
                "y_high": float(stratum["bias"]) + half,
                "n": int(stratum["n"]),
                "label": label,
            }
        )
        rows.append(
            {
                "panel": "E7_sar_spread",
                "series": "sar_sd",
                "group": group,
                "x": int(index + 1),
                "y": math.sqrt(max(float(stratum["variance"]), 0.0)),
                "n": replicates,
                "label": label,
            }
        )
        summary[f"bias_{index + 1}"] = float(stratum["bias"])
        summary[f"sd_{index + 1}"] = math.sqrt(max(float(stratum["variance"]), 0.0))
    pooled = study["sar_pooled"]
    rows.append(
        {
            "panel": "E7_sar_bias",
            "series": "pooled_bias",
            "group": "pooled",
            "y": float(pooled["bias"]),
            "n": int(pooled["n"]),
            "label": "pooled bias",
        }
    )
    summary["pooled_bias"] = float(pooled["bias"])
    return rows, summary


def _budget_rows(experiment: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"n_seeds": int(experiment["n_seeds"])}
    for entry in experiment["performance_vs_budget"]:
        rows.append(
            {
                "panel": "E8_budget_curve",
                "series": entry["policy"],
                "x": float(entry["budget"]),
                "y": float(entry["mean"]),
                "y_low": float(entry["low"]),
                "y_high": float(entry["high"]),
                "n": int(entry["n"]),
            }
        )
        summary[f"{entry['policy']}_b{entry['budget']}"] = float(entry["mean"])
    for entry in experiment["paired_differences"]:
        rows.append(
            {
                "panel": "E8_paired_diff",
                "series": entry["policy_a"],
                "x": float(entry["budget"]),
                "y": float(entry["mean_difference"]),
                "y_low": float(entry["low"]),
                "y_high": float(entry["high"]),
                "n": int(entry["n_pairs"]),
            }
        )
        summary[f"{entry['policy_a']}_diff_b{entry['budget']}"] = float(entry["mean_difference"])
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

    width, height = 20.5, 10.4
    fig = plt.figure(figsize=(width, height), layout="constrained")
    shell = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.17])
    caption = fig.add_subplot(shell[1])
    caption.axis("off")
    caption.text(
        0.0,
        1.0,
        "\n".join(
            textwrap.fill(("How to read this chart: " if index == 0 else "") + line, width=240)
            for index, line in enumerate(_caption_lines())
        ),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        transform=caption.transAxes,
    )
    top = shell[0].subgridspec(2, 2, width_ratios=[1.0, 1.15], hspace=0.42, wspace=0.24)

    # --- column 1: E7 ------------------------------------------------------
    ax_bias = fig.add_subplot(top[0, 0])
    bias = sorted(series("E7_sar_bias", "sar_bias"), key=lambda r: r["x"])
    pooled = series("E7_sar_bias", "pooled_bias")[0]
    ax_bias.axhline(0.0, color=INK, lw=1.2)
    ax_bias.axhline(pooled["y"], color=GREY, ls="--", lw=1.4)
    ax_bias.errorbar(
        [r["x"] for r in bias],
        [r["y"] for r in bias],
        yerr=[[r["y"] - r["y_low"] for r in bias], [r["y_high"] - r["y"] for r in bias]],
        fmt="o",
        ms=6,
        lw=1.5,
        capsize=4,
        color=BLUE,
    )
    ax_bias.set(ylabel="mean R_SAR minus truth risk (signed)")
    _style(ax_bias)
    _head(
        ax_bias,
        f"E7 (mid) per-stratum bias, {summary['replicates']} bootstrap replicates",
        [
            _line("zero (truth)", INK, lw=1.2),
            _line(f"pooled bias {pooled['y']:+.4f}", GREY, "--", 1.4),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, ms=6, label="stratum bias"),
        ],
        3,
    )

    ax_spread = fig.add_subplot(top[1, 0], sharex=ax_bias)
    spread = sorted(series("E7_sar_spread", "sar_sd"), key=lambda r: r["x"])
    ax_spread.bar(
        [r["x"] for r in spread],
        [r["y"] for r in spread],
        width=0.62,
        color=SKY,
        edgecolor=BLUE,
        linewidth=0.8,
    )
    ax_spread.set(xlabel="propensity stratum (1 = lowest fitted propensity)", ylabel="sd of R_SAR across replicates")
    _style(ax_spread)
    _head(ax_spread, "E7 (mid) per-stratum spread (own axis)", None)
    ax_bias.set_xticks([r["x"] for r in bias])
    ax_bias.set_xticklabels([str(int(r["x"])) for r in bias], fontsize=8)
    xs = [r["x"] for r in bias]
    ax_spread.set_xticks(xs)
    ax_spread.set_xticklabels([f"{int(r['x'])}\n{r['label']}" for r in bias], fontsize=8)

    # --- column 2: E8 ------------------------------------------------------
    ax_budget = fig.add_subplot(top[0, 1])
    policies = list(summary["policies"])
    ends: list[tuple[float, str, float]] = []
    for policy in policies:
        data = sorted(series("E8_budget_curve", policy), key=lambda r: r["x"])
        xs = [r["x"] for r in data]
        ys = [r["y"] for r in data]
        colour = POLICY_COLORS[policy]
        ax_budget.fill_between(
            xs, [r["y_low"] for r in data], [r["y_high"] for r in data], color=colour, alpha=0.18, lw=0
        )
        ax_budget.plot(xs, ys, color=colour, lw=2.0, marker="o", ms=4)
        ends.append((float(ys[-1]), policy, float(xs[-1])))
    ends.sort()
    minimum_gap = 0.07
    adjusted: list[tuple[float, str, float]] = []
    for y, policy, x in ends:
        placed = y
        if adjusted and placed - adjusted[-1][0] < minimum_gap:
            placed = adjusted[-1][0] + minimum_gap
        adjusted.append((placed, policy, x))
    for y, policy, x in adjusted:
        ax_budget.annotate(
            POLICY_LABELS[policy],
            xy=(x, y),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            color=POLICY_COLORS[policy],
        )
    budget_all = [
        r for policy in policies for r in series("E8_budget_curve", policy)
    ] + [r for policy in policies for r in series("E8_paired_diff", policy)]
    ax_budget.set(
        xscale="log",
        xlabel="labels bought (log axis)",
        ylabel="recall at 1% FPR on the latent truth",
        ylim=(min(0.0, min(r["y_low"] for r in budget_all)) * 1.02, 1.0),
    )
    _style(ax_budget, "both")
    _head(
        ax_budget,
        f"E8 (post) recall vs label budget, mean over {summary['n_seeds']} seeds",
        None,
    )

    ax_paired = fig.add_subplot(top[1, 1], sharex=ax_budget)
    ax_paired.axhline(0.0, color=INK, lw=1.2)
    handles = [_line("zero (no difference)", INK, lw=1.2)]
    for policy in policies:
        if policy == "random":
            continue
        data = sorted(series("E8_paired_diff", policy), key=lambda r: r["x"])
        xs = [r["x"] for r in data]
        ys = [r["y"] for r in data]
        colour = POLICY_COLORS[policy]
        ax_paired.fill_between(
            xs, [r["y_low"] for r in data], [r["y_high"] for r in data], color=colour, alpha=0.18, lw=0
        )
        ax_paired.plot(xs, ys, color=colour, lw=2.0, marker="o", ms=4)
        handles.append(_line(POLICY_LABELS[policy], colour, lw=2.0))
    ax_paired.set(
        xlabel="labels bought (log axis)",
        ylabel="recall difference to random (paired)",
    )
    _style(ax_paired, "both")
    _head(ax_paired, "E8 (post) paired against random on the same seeds", handles, 4)
    return fig


# ------------------------------------------------------------------ entry
def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Build the R2_NB3_stratum_and_budget sidecar and figure."""
    key = "study:" + str(seed_of(config))
    if key not in _CACHE:
        _CACHE[key] = run_shared_stream_study(config)
    study = _CACHE[key]
    policy_key = "policies:" + str(seed_of(config))
    if policy_key not in _CACHE:
        # `exec.policies._R2_CACHE` is keyed by the config alone and is shared by
        # both R2 experiments, so the canonical config would collide with the
        # naive-versus-aware experiment when both run in one process (the
        # renderer does). The isolation tag changes only that cache key; every
        # experiment reads its sizes from the config it is handed and the tag is
        # ignored by the experiment (and never reaches the sidecar stamp).
        _CACHE[policy_key] = run_replicated_policy_experiment({**dict(config), "_r2_figure": NAME})
    experiment = _CACHE[policy_key]

    stratum_rows, stratum_summary = _stratum_rows(study)
    budget_rows, budget_summary = _budget_rows(experiment)
    rows = [*stratum_rows, *budget_rows]
    summary = {**stratum_summary, **budget_summary, "policies": list(experiment["policies"])}

    sidecar = write_r2_sidecar(NAME, rows, config, out_dir=out_dir)
    figure = save_r2_figure(_figure(rows, summary), NAME, out_dir=out_dir)
    plt.close()
    lines = _caption_lines()
    print(f"{NAME}: {len(rows)} rows -> {sidecar.name} and {figure.name}")
    print("How to read this chart: " + lines[0])
    print(lines[1])
    print(
        "E7: strata={strata} replicates={replicates} pooled bias={pooled_bias:+.5f}; per-stratum bias "
        "[{bias_1:+.4f}, {bias_2:+.4f}, {bias_3:+.4f}, {bias_4:+.4f}, {bias_5:+.4f}, {bias_6:+.4f}], "
        "sd [{sd_1:.4f}, {sd_2:.4f}, {sd_3:.4f}, {sd_4:.4f}, {sd_5:.4f}, {sd_6:.4f}]".format(
            **stratum_summary
        )
    )
    for policy in ("random", "risk_first", "uncertainty_first", "diverse_typology"):
        means = " ".join(
            f"{budget_summary[f'{policy}_b{budget}']:.3f}" for budget in experiment["budgets"]
        )
        print(f"E8 {POLICY_LABELS[policy]} recall at budgets {experiment['budgets']}: {means}")
    for policy in ("risk_first", "uncertainty_first", "diverse_typology"):
        diffs = " ".join(
            f"{budget_summary[f'{policy}_diff_b{budget}']:+.3f}" for budget in experiment["budgets"]
        )
        print(f"E8 {POLICY_LABELS[policy]} minus random: {diffs}")
    return {"figure": str(figure), "sidecar": str(sidecar), "rows": int(len(rows)), "name": NAME}
