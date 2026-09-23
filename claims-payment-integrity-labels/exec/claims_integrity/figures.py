"""Real figures: sidecar CSV + rendered PNG for each viz contract.

The CSV contract lives in ``tests/test_visualization_contract.py`` and
``viz/spec/``; these builders write ``exec/figures/<name>.csv`` and ``.png``
with real study numbers and exec provenance (never mock data).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .ccem import run_ccem_identifiability_experiment
from .config import CONFIG, PROVENANCE
from .experiments import capacity_study, dollars_study

IDENT_INTERVAL = "95% item-bootstrap interval within seed; notebook shows the across-seed spread"
CAPACITY_INTERVAL = "Wilson 95% interval on held-out risk; acceptance is design-determined except for the certificate branch (Clopper-Pearson floor)"
DOLLARS_INTERVAL = "95% bootstrap interval over reviewed claims"
DERIVED = "derived here; synthetic illustration only"

FIGURE_CAPTIONS = {
    "01_identifiability": (
        "How to read this chart: higher error means less identifiable recovery; the specialist "
        "condition is an intentional stress case under a reviewer x class coverage break, not a "
        "theorem claim; both series are aligned error after the Hungarian permutation."
    ),
    "02_capacity_risk": (
        "How to read this chart: the reviewer-selected calibration comparison is an assumption "
        "falsifier, not silently theorem-valid certification. SCRC/SCoRC language is qualified by "
        "the displayed calibration/exchangeability condition; infeasible finite-grid points are "
        "shown, not dropped."
    ),
    "03_dollars_recovered": (
        "How to read this chart: utility subtracts review cost; dollars are heavy-tailed "
        "(mean and median both reported); any P3 q10 numerical claim is pending/preprint, "
        "not a verified result, and no population guarantee follows from this synthetic fixture."
    ),
}


def identifiability_rows(seeds: Sequence[int], n: int = 2000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for regime in ("anchored", "specialist_weak_anchor"):
            result = run_ccem_identifiability_experiment(seed=seed, regime=regime, n=n)
            rows.append(
                {
                    "reviewer_regime": regime,
                    "seed": int(seed),
                    "sample_size": int(n),
                    "aligned_error": f"{result['aligned_identifiability_error']:.5f}",
                    "error_low": f"{result['error_low']:.5f}",
                    "error_high": f"{result['error_high']:.5f}",
                    "confusion_error": f"{result['confusion_error']:.5f}",
                    "confusion_low": f"{result['confusion_low']:.5f}",
                    "confusion_high": f"{result['confusion_high']:.5f}",
                    "unresolved_mass": f"{result['unresolved_mass']:.5f}",
                    "setting": "synthetic claims; same seed and sample size per paired regime",
                    "interval_method": IDENT_INTERVAL,
                    "source_status": "derived here; CCEM equations source-backed (q2), generator derived here",
                    "provenance": f"{PROVENANCE}; stream=claims_integrity-v1; figure=01",
                    "legend_label": "aligned error",
                    "caption": FIGURE_CAPTIONS["01_identifiability"],
                }
            )
    return rows


def capacity_rows(seeds: Sequence[int], n: int | None = None, capacities: Sequence[int] | None = None) -> list[dict[str, Any]]:
    studies = capacity_study(seeds, n=n, capacities=capacities)
    rows = []
    for row in studies:
        rows.append(
            {
                "capacity_claims": int(row["capacity_claims"]),
                "policy": row["policy"],
                "seed": int(row["seed"]),
                "risk": f"{row['risk']:.5f}",
                "risk_low": f"{row['risk_low']:.5f}",
                "risk_high": f"{row['risk_high']:.5f}",
                "acceptance": f"{row['acceptance']:.5f}",
                "acceptance_low": f"{row['acceptance_low']:.5f}",
                "acceptance_high": f"{row['acceptance_high']:.5f}",
                "feasible": "true" if row["feasible"] else "false",
                "acceptance_floor": f"{1.0 - row['capacity_claims'] / (n or CONFIG['n_claims']):.4f}",
                "certified_score": row.get("certified_score", ""),
                "certified_threshold": f"{row['certified_threshold']:.4f}" if np.isfinite(row.get("certified_threshold", float("nan"))) else "nan",
                "risk_ucb": f"{row['risk_ucb']:.4f}" if np.isfinite(row.get("risk_ucb", float("nan"))) else "nan",
                "utility_lcb": f"{row['utility_lcb']:.4f}" if np.isfinite(row.get("utility_lcb", float("nan"))) else "nan",
                "setting": "synthetic claims; equal review capacity; finite grid includes infeasible points",
                "interval_method": CAPACITY_INTERVAL,
                "calibration_condition": (
                    "reviewer-selected calibration comparison is an assumption falsifier; SCRC/SCoRC qualified by "
                    "displayed calibration/exchangeability condition; certificate arm uses the disjoint certification split"
                ),
                "source_status": "derived here; P2/SCRC and P3/SCoRC machinery source-backed (q5-q9), capacity branch derived here",
                "provenance": f"{PROVENANCE}; stream=claims_integrity-v1; figure=02",
                "x_label": "review capacity (claims)",
                "y_label": "selected risk / acceptance",
                "caption": FIGURE_CAPTIONS["02_capacity_risk"],
            }
        )
    return rows


def dollars_rows(seeds: Sequence[int], minutes_grid: Sequence[int] = (30, 90, 180)) -> list[dict[str, Any]]:
    studies = dollars_study(seeds, minutes_grid=minutes_grid)
    rows = []
    for row in studies:
        rows.append(
            {
                "reviewed_minutes": int(row["reviewed_minutes"]),
                "policy": row["policy"],
                "seed": int(row["seed"]),
                "dollars_recovered": f"{row['dollars_recovered']:.2f}",
                "recovered_low": f"{row['recovered_low']:.2f}",
                "recovered_high": f"{row['recovered_high']:.2f}",
                "net_utility": f"{row['net_utility']:.2f}",
                "utility_low": f"{row['utility_low']:.2f}",
                "utility_high": f"{row['utility_high']:.2f}",
                "summary": row["summary"],
                "claims_reviewed": int(row["claims_reviewed"]),
                "review_cost": f"{row['review_cost']:.2f}",
                "setting": "synthetic claim stream with heavy-tailed recoverable dollars; same seeded stream per policy",
                "interval_method": DOLLARS_INTERVAL,
                "source_status": "derived here; synthetic dollar proxy and review-cost accounting",
                "provenance": f"{PROVENANCE}; stream=claims_integrity-v1; figure=03",
                "x_label": "reviewer minutes",
                "y_label": "dollars recovered / net utility",
                "caption": FIGURE_CAPTIONS["03_dollars_recovered"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# rendering


def render_identifiability(rows: list[dict[str, Any]]):
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    colors = {"anchored": "#0072B2", "specialist_weak_anchor": "#D55E00"}
    labels = {"anchored": "anchored control", "specialist_weak_anchor": "specialist/weak-anchor stress case"}
    for regime in ("anchored", "specialist_weak_anchor"):
        subset = [row for row in rows if row["reviewer_regime"] == regime]
        x = np.arange(len(subset))
        y = np.array([float(row["aligned_error"]) for row in subset])
        lo = np.array([float(row["error_low"]) for row in subset])
        hi = np.array([float(row["error_high"]) for row in subset])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color=colors[regime], label=f"aligned error — {labels[regime]}")
        ax.plot(x, [float(row["confusion_error"]) for row in subset], "s--", alpha=0.55, color=colors[regime], label=f"aligned confusion error — {labels[regime]}")
    seeds = sorted({row["seed"] for row in rows})
    ax.set_xticks(np.arange(len(seeds)), [str(s) for s in seeds])
    ax.set(xlabel="repeated seed", ylabel="aligned error", title="Aligned identifiability by reviewer regime (synthetic, derived here)")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, FIGURE_CAPTIONS["01_identifiability"], ha="left", va="bottom", fontsize=7, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return fig


def render_capacity(rows: list[dict[str, Any]]):
    seeds = sorted({int(row["seed"]) for row in rows})
    focus = rows if len(seeds) == 1 else [row for row in rows if int(row["seed"]) == seeds[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {"naive": "#D55E00", "noise-aware": "#0072B2", "selective": "#009E73"}
    for policy in ("naive", "noise-aware", "selective"):
        subset = sorted([row for row in focus if row["policy"] == policy], key=lambda r: int(r["capacity_claims"]))
        x = [int(row["capacity_claims"]) for row in subset]
        risk = [float(row["risk"]) for row in subset]
        low = [float(row["risk_low"]) for row in subset]
        high = [float(row["risk_high"]) for row in subset]
        axes[0].errorbar(x, risk, yerr=[np.array(risk) - np.array(low), np.array(high) - np.array(risk)], fmt="o-", capsize=4, color=colors[policy], label=policy)
        feasible_x = [xi for xi, row in zip(x, subset) if row["feasible"] == "true"]
        feasible_y = [ri for ri, row in zip(risk, subset) if row["feasible"] == "true"]
        infeasible_x = [xi for xi, row in zip(x, subset) if row["feasible"] != "true"]
        infeasible_y = [ri for ri, row in zip(risk, subset) if row["feasible"] != "true"]
        axes[0].plot(feasible_x, feasible_y, "o", color=colors[policy], markersize=7)
        axes[0].plot(infeasible_x, infeasible_y, "x", color=colors[policy], markersize=11, markeredgewidth=2)
        acceptance = [float(row["acceptance"]) for row in subset]
        acc_low = [float(row["acceptance_low"]) for row in subset]
        acc_high = [float(row["acceptance_high"]) for row in subset]
        axes[1].errorbar(x, acceptance, yerr=[np.array(acceptance) - np.array(acc_low), np.array(acc_high) - np.array(acceptance)], fmt="o-", capsize=4, color=colors[policy], label=policy)
    axes[0].set(xlabel="review capacity (claims)", ylabel="selected risk (auto-pay error)", title="Selected risk at equal capacity")
    axes[1].set(xlabel="review capacity (claims)", ylabel="acceptance (auto-pay share)", title="Acceptance at equal capacity")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[0].text(0.02, 0.98, "x = certificate INFEASIBLE", transform=axes[0].transAxes, fontsize=7, va="top", color="#444444")
    fig.text(0.01, 0.005, FIGURE_CAPTIONS["02_capacity_risk"], ha="left", va="bottom", fontsize=7, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def render_dollars(rows: list[dict[str, Any]]):
    seeds = sorted({int(row["seed"]) for row in rows})
    focus = rows if len(seeds) == 1 else [row for row in rows if int(row["seed"]) == seeds[0]]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    colors = {"naive": "#D55E00", "noise-aware": "#0072B2"}
    for policy in ("naive", "noise-aware"):
        subset = sorted([row for row in focus if row["policy"] == policy], key=lambda r: int(r["reviewed_minutes"]))
        x = [int(row["reviewed_minutes"]) for row in subset]
        recovered = np.array([float(row["dollars_recovered"]) for row in subset])
        low = np.array([float(row["recovered_low"]) for row in subset])
        high = np.array([float(row["recovered_high"]) for row in subset])
        ax.errorbar(x, recovered, yerr=[recovered - low, high - recovered], fmt="o-", capsize=4, color=colors[policy], label=f"{policy} — recovered dollars")
        net = np.array([float(row["net_utility"]) for row in subset])
        ax.plot(x, net, "s--", color=colors[policy], label=f"{policy} — net utility (after review cost)")
    ax.set(xlabel="reviewer minutes", ylabel="dollars", title="Dollars recovered and net utility versus review cost (synthetic, derived here)")
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, FIGURE_CAPTIONS["03_dollars_recovered"], ha="left", va="bottom", fontsize=7, color="#444444", wrap=True)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return fig


def write_figure(name: str, rows: list[dict[str, Any]], fig, out_dir: str | Path) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / f"{name}.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    png_path = target / f"{name}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    return csv_path, png_path


def build_all(out_dir: str | Path, seeds: Sequence[int] = (11, 23), ident_seeds: Sequence[int] | None = None, n_ident: int = 2000) -> dict[str, tuple[Path, Path]]:
    out = {}
    ident = identifiability_rows(ident_seeds or (11, 23, 37, 53), n=n_ident)
    out["01_identifiability"] = write_figure("01_identifiability", ident, render_identifiability(ident), out_dir)
    cap = capacity_rows(seeds)
    out["02_capacity_risk"] = write_figure("02_capacity_risk", cap, render_capacity(cap), out_dir)
    dol = dollars_rows(seeds)
    out["03_dollars_recovered"] = write_figure("03_dollars_recovered", dol, render_dollars(dol), out_dir)
    return out
