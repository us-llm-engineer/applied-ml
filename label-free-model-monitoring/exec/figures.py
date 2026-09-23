"""Render the five required figures from the study's own generated data.

Each figure writes a CSV with the spec's columns plus seed/run_id provenance
and a PNG rendered from a separate mock format.  Never plots mock data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dataclasses import replace
from typing import Sequence

from .audit import DelayedLabelQueue
from .estimators import estimate_sjs_gap, sequential_tail_monitor
from .policy import _trigger, compare_policies, compute_signals, policy_metric_rows, policy_timeline_rows, run_stream
from .retraining_economics import (
    brute_force_schedule,
    improving_loss_table,
    stepwise_improvement_table,
    observed_adjacent_gap_max,
    optimal_retraining_schedule,
    regol_count_bound,
    simulate_retraining_queue,
    sweep_policy_latency_budget,
)
from .synthetic import D, ShiftScenario, make_shift_scenario

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "exec" / "figures"

REGIME_ORDER = (
    "none",
    "benign_covariate",
    "sparse_joint",
    "harmful_concept",
    "dense_joint",
    "d3m_regime_2",
)
REGIME_COLORS = {
    "none": "#7f7f7f",
    "benign_covariate": "#1f77b4",
    "sparse_joint": "#2ca02c",
    "harmful_concept": "#d62728",
    "dense_joint": "#9467bd",
    "d3m_regime_2": "#ff7f0e",
}

CAPTIONS = {
    "fig-a1": (
        "How to read this chart: each row is one feature and the colour is its "
        "standardised deployment-minus-reference mean.\nRead the second panel "
        "against the first: benign drift moves three noise features hard while "
        "latent risk stays put, and dense joint shift moves every feature while "
        "risk barely changes. Drift magnitude is not a retraining oracle."
    ),
    "fig-a2": (
        "How to read this chart: the point is the estimated target-source gap, "
        "the bar is its 90% bootstrap interval, and the dashed vertical line "
        "is latent truth.\nFilled markers are certified SJS estimates and "
        "open markers are refused diagnostics; a dense joint shift must appear "
        "as a refusal and a label-rule change must not be silently certified."
    ),
    "fig-a3": (
        "How to read this chart: the solid line is the time-uniform lower bound "
        "on the deployment error-propensity rate and the dashed line is the "
        "source rate plus tolerance.\nThe alarm is the first crossing; the "
        "dotted line is the latent high-loss rate that the label-free monitor "
        "is trying to track."
    ),
    "fig-a4": (
        "How to read this chart: the colour band at the top of each panel is "
        "the controller state (green stable, pink audit, blue candidate, grey "
        "cooldown); the curves are the deployed model's risk, the original "
        "champion's risk and the candidate's risk.\nTrace stable -> audit -> "
        "candidate -> promote/rollback -> cooldown: benign drift should not "
        "promote, and a harmful shift should recover only after a validated "
        "candidate."
    ),
    "fig-a6": (
        "How to read this chart: the top strip is the serving model epoch per "
        "latency case, the middle strip is queue occupancy with the pending "
        "audited-label count, and the bottom strip is nominal budget and "
        "completed retrains.\nThe serving epoch must stay flat through request "
        "and training completion, then advance only at deployment; a request "
        "rejected while work is in flight must not move the epoch or the budget."
    ),
    "fig-a7": (
        "How to read this chart: each facet is one drift regime and one update "
        "mode, x is the combined training + deployment latency, y is the risk "
        "served after the drift onset, colour is the policy and marker size is "
        "the number of completed retrains.\nHigher nominal budget does not "
        "imply more completed retrains when latency leaves jobs in flight, and "
        "the companion panel shows the stranded share of that budget."
    ),
    "fig-a8-bound": (
        "How to read this chart: each setting shows the optimized retrain count "
        "beside the conditional ceiling computed from its checked uniform "
        "adjacent-model gap.\nA setting whose observed adjacent gap exceeds its "
        "declared L is annotated as outside the theorem's scope and carries no "
        "ceiling bar, so a guarantee is never drawn where the assumption fails."
    ),
    "delayed-audit-timeline": (
        "How to read this chart: pending and released audited-label counts per "
        "stream step, with the running audited-loss estimate on the right axis.\n"
        "Labels stay hidden until their availability boundary and then release "
        "exactly once, so the evidence a controller can act on always lags the "
        "arrivals it describes."
    ),
    "fig-a8": (
        "How to read this chart: the left panel is cumulative total cost per "
        "schedule kind (solid optimized, dashed exhaustive reference, dotted "
        "no-action) for each measured setting; the right panel places the "
        "optimized/reference retrain count against the published conditional "
        "ceiling for the settings whose adjacent-model gaps were checked.\nThe "
        "ceiling only applies where the observed adjacent gap respects the "
        "declared uniform L; the excluded diagnostic row is drawn without a "
        "guarantee."
    ),
    "fig-a9": (
        "How to read this chart: the left panel plots detection delay after "
        "onset for each paired seed against the label delay, with the black "
        "line the seed median; the right panel bars the miss rate and "
        "false-retrain rate at each label delay, with mean cost annotated.\n"
        "A longer label delay is not simply 'later': on some seeds the first "
        "audit resolves during a worse stretch of drift and detects sooner, "
        "and on others it resolves too late within the horizon and the "
        "deterioration is missed outright rather than merely delayed."
    ),
    "fig-a5": (
        "How to read this chart: each point is one policy at its mean over "
        "paired seeds, with 90% bootstrap seed intervals on both axes.\nNo "
        "policy dominates: cheaper policies miss more deterioration and "
        "reliable recovery pays audit and compute cost, so the trade-off is "
        "shown rather than scored away."
    ),
}


@dataclass
class FigureResult:
    figure: str
    csv_path: str
    png_path: str
    rows: int
    caption: str
    stats: dict[str, Any]


def _stamp() -> str:
    return f"t5-r01-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def _with_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach an explicit provenance string to every row (seed + exec run id)."""
    frame = frame.copy()
    frame["provenance"] = [
        f"seed={seed}|run_id={run}|producer=exec/figures.py"
        for seed, run in zip(frame["seed"], frame["run_id"])
    ]
    return frame


def _write(frame: pd.DataFrame, stem: str) -> str:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{stem}.csv"
    _with_provenance(frame).to_csv(path, index=False)
    return str(path)


def _save(fig: plt.Figure, stem: str) -> str:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{stem}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Fig A1
# --------------------------------------------------------------------------- #
def figure_a1(run_id: str, seed: int = 7) -> FigureResult:
    rows = []
    for regime in REGIME_ORDER:
        scenario = make_shift_scenario(seed=seed, regime=regime, n_source=2400, n_target=1600)
        for j in range(D):
            for window, xs in (("reference", scenario.source_x), ("deployment", scenario.target_x)):
                rows.append(
                    {
                        "regime": regime,
                        "feature": f"x{j}",
                        "window": window,
                        "feature_mean": float(xs[:, j].mean()),
                        "feature_std": float(xs[:, j].std()),
                        "score_mean": float(
                            np.mean(scenario.source_score if window == "reference" else scenario.target_score)
                        ),
                        "latent_risk": scenario.source_risk if window == "reference" else scenario.target_risk,
                        "seed": scenario.seed,
                        "run_id": run_id,
                    }
                )
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.55})
    stats = {}
    ax = axes[0]
    diffs = np.zeros((len(REGIME_ORDER), D))
    for i, regime in enumerate(REGIME_ORDER):
        g = frame[frame.regime == regime]
        ref = g[g.window == "reference"].sort_values("feature")
        dep = g[g.window == "deployment"].sort_values("feature")
        diffs[i] = (dep.feature_mean.values - ref.feature_mean.values) / np.maximum(ref.feature_std.values, 1e-9)
    im = ax.imshow(diffs, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
    ax.set_xticks(range(D), [f"x{j}" for j in range(D)])
    ax.set_yticks(range(len(REGIME_ORDER)), REGIME_ORDER, fontsize=8)
    ax.set_title("Standardised feature mean shift (deployment - reference)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="standard deviations")
    for i in range(len(REGIME_ORDER)):
        for j in range(D):
            ax.text(j, i, f"{diffs[i, j]:+.1f}", ha="center", va="center", fontsize=6)

    ax = axes[1]
    for i, regime in enumerate(REGIME_ORDER):
        g = frame[frame.regime == regime]
        ref = g[g.window == "reference"].latent_risk.iloc[0]
        dep = g[g.window == "deployment"].latent_risk.iloc[0]
        ax.plot([ref, dep], [i, i], color=REGIME_COLORS[regime], lw=3, solid_capstyle="round")
        ax.plot(ref, i, "o", color="white", mec=REGIME_COLORS[regime], mew=2, ms=8)
        ax.plot(dep, i, "o", color=REGIME_COLORS[regime], ms=8)
        stats[regime] = {"reference_risk": ref, "deployment_risk": dep, "gap": dep - ref}
    ax.set_yticks(range(len(REGIME_ORDER)), REGIME_ORDER, fontsize=8)
    ax.set_xlabel("Brier risk (latent)")
    ax.set_title("Latent risk: reference (open) -> deployment (filled)")
    ax.grid(axis="x", alpha=0.3)
    fig.text(0.01, -0.02, CAPTIONS["fig-a1"], fontsize=7.5, va="top", wrap=True)
    fig.suptitle("Fig A1 — shift regimes before monitoring", y=1.02, fontsize=11)
    png = _save(fig, "fig-a1")
    csv = _write(frame, "fig-a1")
    return FigureResult("fig-a1", csv, png, len(frame), CAPTIONS["fig-a1"], stats)


# --------------------------------------------------------------------------- #
# Fig A2
# --------------------------------------------------------------------------- #
def _naive_gap(scenario: ShiftScenario) -> tuple[float, float, float]:
    """Covariate reweighting with no identifiability diagnostics (baseline).

    Two-fold cross-fitting keeps the density-ratio classifier honest: weights on
    each source half come from a classifier that did not see that half.
    """
    mu, sd = scenario.source_x.mean(axis=0), scenario.source_x.std(axis=0)
    sd[sd == 0] = 1.0

    def phi(a: np.ndarray) -> np.ndarray:
        z = (a - mu) / sd
        return np.hstack([z, z**2])

    x_all, sl = scenario.source_x, scenario.source_loss
    pt = phi(scenario.target_x)
    n = len(x_all)
    folds = np.array_split(np.arange(n), 5)
    diffs = []
    for fold in folds:
        fit_idx = np.setdiff1d(np.arange(n), fold)
        ps_fit, ps_eval = phi(x_all[fit_idx]), phi(x_all[fold])
        clf = LogisticRegression(max_iter=4000, C=1.0)
        clf.fit(np.vstack([ps_fit, pt]), np.r_[np.zeros(len(ps_fit)), np.ones(len(pt))])
        r = np.clip(clf.predict_proba(ps_eval)[:, 1], 1e-6, 1 - 1e-6)
        w = np.clip((r / (1 - r)) * (len(pt) / len(ps_fit)), 0, 20)
        diffs.append(float(np.sum(w * sl[fold]) / np.sum(w) - sl[fold].mean()))
    est = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    return est, est - 1.65 * max(se, 1e-4), est + 1.65 * max(se, 1e-4)


def figure_a2(run_id: str, seed: int = 11) -> FigureResult:
    rows = []
    stats = {}
    for regime in REGIME_ORDER:
        scenario = make_shift_scenario(seed=seed, regime=regime, n_source=2400, n_target=1600)
        result = estimate_sjs_gap(
            scenario.source_x, scenario.source_y, scenario.target_x, scenario.predictions
        )
        rows.append(
            {
                "regime": regime,
                "estimator": "sees_style",
                "true_gap": scenario.true_gap,
                "estimate": result.estimate,
                "lower": result.lower,
                "upper": result.upper,
                "certified": result.certified,
                "seed": scenario.seed,
                "run_id": run_id,
                "status": result.status,
            }
        )
        naive, lo, hi = _naive_gap(scenario)
        rows.append(
            {
                "regime": regime,
                "estimator": "covariate_reweight",
                "true_gap": scenario.true_gap,
                "estimate": naive,
                "lower": lo,
                "upper": hi,
                "certified": True,
                "seed": scenario.seed,
                "run_id": run_id,
                "status": "not_diagnosed",
            }
        )
        stats[regime] = {
            "true_gap": scenario.true_gap,
            "sees_estimate": result.estimate,
            "sees_status": result.status,
            "covariate_estimate": naive,
        }
    frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    offsets = {"sees_style": 0.16, "covariate_reweight": -0.16}
    markers = {"sees_style": "o", "covariate_reweight": "s"}
    finite = frame[np.isfinite(frame["estimate"].astype(float))]
    xlo = min(-0.04, float(finite["lower"].min()) - 0.02)
    xhi = max(0.30, float(finite["upper"].max()) + 0.04)
    positions = {}
    for i, regime in enumerate(REGIME_ORDER):
        base_y = len(REGIME_ORDER) - 1 - i
        positions[regime] = base_y
        true_gap = float(frame[frame.regime == regime].true_gap.iloc[0])
        ax.plot([true_gap, true_gap], [base_y - 0.34, base_y + 0.34], color="k", ls="--", lw=1.0, alpha=0.75)
        for estimator in ("sees_style", "covariate_reweight"):
            row = frame[(frame.regime == regime) & (frame.estimator == estimator)].iloc[0]
            y = base_y + offsets[estimator]
            if np.isfinite(row["estimate"]):
                ax.plot([row.lower, row.upper], [y, y], color=REGIME_COLORS[regime], lw=2.2, alpha=0.85)
                ax.plot(
                    row.estimate,
                    y,
                    markers[estimator],
                    color=REGIME_COLORS[regime],
                    mfc="white" if not row.certified else REGIME_COLORS[regime],
                    ms=7,
                )
            else:
                ax.plot([xlo + 0.01, xhi - 0.01], [y, y], color="#cccccc", lw=7, alpha=0.6, zorder=1)
                ax.plot(xhi - 0.01, y, "x", color=REGIME_COLORS[regime], ms=9, mew=2)
                ax.text(xhi - 0.02, y, " refused", fontsize=6.5, va="center", ha="right", color="#555555")
    ax.axvline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_yticks(range(len(REGIME_ORDER)), REGIME_ORDER[::-1], fontsize=8)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Brier risk gap (target - source); dashed vertical = latent truth")
    ax.set_title("Fig A2 — label-free risk estimate against latent truth")
    ax.grid(alpha=0.25, axis="x")
    ax.plot([], [], "o", color="k", label="sees_style (open = refused diagnostics)")
    ax.plot([], [], "s", color="k", label="covariate_reweight (no diagnostics)")
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.01, -0.04, CAPTIONS["fig-a2"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a2")
    csv = _write(frame, "fig-a2")
    return FigureResult("fig-a2", csv, png, len(frame), CAPTIONS["fig-a2"], stats)


# --------------------------------------------------------------------------- #
# Fig A3
# --------------------------------------------------------------------------- #
def figure_a3(run_id: str, seeds: tuple[int, ...] = (31, 32, 33)) -> FigureResult:
    rows = []
    stats = {}
    regimes = ("benign_covariate", "harmful_concept")
    for regime in regimes:
        stats[regime] = []
        for seed in seeds:
            scenario = make_shift_scenario(seed=seed, regime=regime, n_source=1600, n_target=1400)
            result = sequential_tail_monitor(scenario, alpha=0.10, tolerance=0.02)
            stats[regime].append(
                {
                    "seed": seed,
                    "ever_alarm": result.ever_alarm,
                    "alarm_time": result.alarm_time,
                    "source_rate": result.source_tail_rate,
                    "selector_power": result.selector_power,
                    "selector_fdp": result.selector_fdp,
                }
            )
            step = max(1, len(result.lower_bounds) // 220)
            for t in range(0, len(result.lower_bounds), step):
                rows.append(
                    {
                        "regime": regime,
                        "time": t + 1,
                        "lower_bound": float(result.lower_bounds[t]),
                        "source_upper": float(result.source_upper),
                        "tolerance": float(result.tolerance),
                        "alarm": bool(result.alarms[t]),
                        "latent_tail_rate": float(result.latent_tail_rates[t]),
                        "seed": seed,
                        "run_id": run_id,
                        "hit_rate": float(result.hit_rates[t]),
                    }
                )
    frame = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, regime in zip(axes, regimes):
        g = frame[frame.regime == regime]
        for k, seed in enumerate(seeds):
            gs = g[g.seed == seed].sort_values("time")
            label_bound = "lower bound (3 seeds)" if k == 1 else None
            label_latent = "latent high-loss (3 seeds)" if k == 1 else None
            ax.plot(gs.time, gs.lower_bound, lw=1.5, alpha=0.9, color="#1f77b4", label=label_bound)
            ax.plot(gs.time, gs.latent_tail_rate, lw=1.1, ls=":", alpha=0.9, color="#d62728", label=label_latent)
            alarm = gs[gs.alarm]
            if len(alarm):
                ax.scatter(alarm.time.iloc[:1], alarm.lower_bound.iloc[:1], marker="v", s=90, color="red", zorder=5)
            ref = gs.source_upper.iloc[0]
            ax.axhline(ref, color="k", ls="--", lw=1.2, label="source rate" if k == 0 else None)
            ax.axhline(ref + gs.tolerance.iloc[0], color="k", ls="-.", lw=0.9, label="source + tolerance" if k == 0 else None)
        ax.set_title(f"{regime}")
        ax.set_xlabel("arrival index")
        ax.set_ylim(-0.15, 0.75)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("rate")
    axes[0].legend(fontsize=7.0, loc="lower right")
    fig.suptitle("Fig A3 — time-uniform sequential evidence (dashed = source rate, dash-dot = + tolerance)", y=1.02, fontsize=10.5)
    fig.text(0.01, -0.05, CAPTIONS["fig-a3"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a3")
    csv = _write(frame, "fig-a3")
    return FigureResult("fig-a3", csv, png, len(frame), CAPTIONS["fig-a3"], stats)


# --------------------------------------------------------------------------- #
# Fig A4
# --------------------------------------------------------------------------- #
def figure_a4(run_id: str, stream_seed: int = 0) -> FigureResult:
    scenario = make_shift_scenario(seed=stream_seed, regime="harmful_concept", n_source=1800, n_target=1800)
    stream = scenario.sample_deployment(seed=stream_seed)
    policies = ["no_action", "drift_only", "corroborated"]
    report = compare_policies(
        scenario, policies=policies, seeds=[stream_seed], audit_size=256, cooldown=8, audit_budget=512
    )
    rows = policy_timeline_rows(report, policies)
    frame = pd.DataFrame(rows)
    frame["run_id"] = run_id
    frame["source_run_id"] = stream.run_id

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.6), sharex=True)
    state_colors = {"stable": "#c7e9c0", "audit": "#fee0d2", "candidate": "#deebf7", "cooldown": "#f0f0f0"}
    state_order = ["stable", "audit", "candidate", "cooldown"]
    for ax, policy in zip(axes, policies):
        g = frame[frame.policy == policy].sort_values("time")
        ax.plot(g.time, g.champion_risk, color="#7f7f7f", lw=1.2, ls="--", label="original champion risk")
        ax.plot(g.time, g.latent_risk, color="#d62728", lw=1.6, label="deployed model risk")
        cand = g[np.isfinite(g.candidate_risk)]
        if len(cand):
            ax.plot(cand.time, cand.candidate_risk, color="#1f77b4", lw=1.2, alpha=0.9, label="candidate risk")
        for event, color, marker in (
            ("audit_clear", "#17becf", "o"),
            ("promote", "#2ca02c", "^"),
            ("rollback", "#9467bd", "s"),
        ):
            ev = g[g.event == event]
            if len(ev):
                ax.scatter(ev.time, ev.latent_risk, color=color, marker=marker, s=45, zorder=5, label=event)
        ax.axvline(stream.onset_step, color="k", lw=1, alpha=0.5)
        ax.set_ylabel("Brier risk")
        ax.set_title(f"policy: {policy}", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, ncol=3, loc="upper left")
        states = g.state.fillna("stable").tolist()
        top = ax.get_ylim()[1]
        band = np.array([[state_order.index(s) if s in state_order else 0 for s in states]])
        ax.imshow(
            band,
            aspect="auto",
            cmap=matplotlib.colors.ListedColormap([state_colors[s] for s in state_order]),
            extent=[g.time.min(), g.time.max(), top * 0.94, top],
            vmin=-0.5,
            vmax=len(state_order) - 0.5,
            zorder=4,
        )
        ax.set_ylim(0.0, top * 1.04)
    axes[-1].set_xlabel("deployment step (64 arrivals per step)")
    fig.suptitle("Fig A4 — retraining controller lifecycle on a harmful shift (dashed: original champion risk)", fontsize=10.5)
    fig.text(0.01, 0.0, CAPTIONS["fig-a4"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a4")
    csv = _write(frame, "fig-a4")
    return FigureResult(
        "fig-a4", csv, png, len(frame), CAPTIONS["fig-a4"], {"onset_step": stream.onset_step}
    )


# --------------------------------------------------------------------------- #
# Fig A5
# --------------------------------------------------------------------------- #
def figure_a5(run_id: str, seeds: range = range(12)) -> FigureResult:
    scenario = make_shift_scenario(seed=53, regime="harmful_concept", n_source=1800, n_target=1800)
    policies = ["no_action", "drift_only", "threshold", "abstain_route", "corroborated"]
    report = compare_policies(scenario, policies=policies, seeds=seeds)
    rows = policy_metric_rows(report)
    frame = pd.DataFrame(rows)
    frame["run_id"] = run_id

    def _interval(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        boots = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
        return float(np.percentile(boots, 5)), float(np.percentile(boots, 95))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    for ax, metric, title in (
        (axes[0], "miss_rate", "cost vs missed deterioration"),
        (axes[1], "recovery_rate", "cost vs recovery"),
    ):
        for policy in policies:
            g = frame[frame.policy == policy]
            cx = float(g.total_cost.mean())
            cy = float(g[metric].mean())
            clo, chi = _interval(g.total_cost.values)
            ylo, yhi = _interval(g[metric].values)
            ax.errorbar(
                cx,
                cy,
                xerr=[[cx - clo], [chi - cx]],
                yerr=[[cy - ylo], [yhi - cy]],
                fmt="o",
                ms=8,
                capsize=3,
                lw=1.2,
                label=policy,
            )
            offset = {
                "no_action": (7, 5),
                "drift_only": (7, -13),
                "threshold": (-10, 9),
                "abstain_route": (-38, -16),
                "corroborated": (-30, -16),
            }[policy]
            ax.annotate(policy, (cx, cy), fontsize=7.5, xytext=offset, textcoords="offset points")
        ax.set_xlabel("mean intervention + exposure cost per stream (90% seed interval)")
        ax.set_ylabel(f"{metric.replace('_', ' ')} (90% seed interval)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.1, 1.1)
    axes[0].legend(fontsize=7.5, loc="center right")
    fig.suptitle("Fig A5 — policy utility frontier on the harmful shift (policy means, 90% seed intervals)", fontsize=10.5)
    fig.text(0.01, -0.05, CAPTIONS["fig-a5"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a5")
    csv = _write(frame, "fig-a5")
    means = {
        name: {
            "false_retrain_rate": row["false_retrain_rate"],
            "miss_rate": row["miss_rate"],
            "median_detection_delay": row["median_detection_delay"],
            "recovery_rate": row["recovery_rate"],
            "total_cost": row["total_cost"],
            "retrain_count": row["retrain_count"],
            "audit_budget": row["audit_budget"],
        }
        for name, row in report.items()
    }
    return FigureResult("fig-a5", csv, png, len(frame), CAPTIONS["fig-a5"], means)



# --------------------------------------------------------------------------- #
# Fig A6 — request, audit, and deployment event time
# --------------------------------------------------------------------------- #
def _busy_intervals(run, train_latency: int, deploy_latency: int) -> list[tuple[int, int]]:
    return [
        (event.time, event.time + train_latency + deploy_latency)
        for event in run.events
        if event.kind == "request_accepted"
    ]


def figure_a6(run_id: str, seed: int = 0, horizon: int = 8) -> FigureResult:
    scenario = make_shift_scenario(seed=seed, regime="harmful_concept", n_source=320, n_target=256)
    stream = scenario.sample_deployment(seed=seed + 7, n_steps=horizon, batch_size=16)
    rows: list[dict[str, Any]] = []
    cases = (("immediate", 0, 0), ("delayed", 2, 1))
    budget = 2
    audit_delay = 2
    for case, train_latency, deploy_latency in cases:
        run = simulate_retraining_queue(
            request_times=[1, 2],
            horizon=horizon,
            retrain_budget=budget,
            train_latency=train_latency,
            deploy_latency=deploy_latency,
        )
        busy = _busy_intervals(run, train_latency, deploy_latency)
        queue = DelayedLabelQueue()
        released_losses: list[float] = []
        for t in range(horizon):
            _, _, _, loss_t = stream.batch(t)
            if t >= 1:
                queue.submit(f"{case}-s{t}", float(loss_t[0]), available_at=t + audit_delay, submitted_at=t)
            released = queue.release(now=t)
            released_losses.extend(float(value) for _, value in released)
            events = run.events_at(t)
            event_text = "|".join(event.kind for event in events) if events else "monitor"
            serving = run.model_epoch_at(t)
            in_flight = any(lo <= t < hi for lo, hi in busy)
            rows.append(
                {
                    "latency_case": case,
                    "time": t,
                    "event": event_text,
                    "model_epoch_serving": serving,
                    "queue_busy": int(in_flight),
                    "nominal_budget_remaining": budget - run.completed_by_time(t),
                    "effective_completed_retrains": run.completed_by_time(t),
                    "audit_pending_labels": queue.pending_count,
                    "audit_released_labels": queue.released_count,
                    "audit_loss_mean": float(np.mean(released_losses)) if released_losses else np.nan,
                    "seed": seed,
                    "run_id": run_id,
                }
            )
    frame = _with_provenance(pd.DataFrame(rows))

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 6.9), sharex="col")
    colors = {"immediate": "#1f77b4", "delayed": "#d62728"}
    for col, (case, train_latency, deploy_latency) in enumerate(cases):
        g = frame[frame.latency_case == case].sort_values("time")
        color = colors[case]
        axes[0, col].step(g.time, g.model_epoch_serving, where="post", marker="o", color=color)
        axes[0, col].set_ylabel("serving epoch")
        axes[0, col].set_title(f"{case} latency (train {train_latency} + deploy {deploy_latency})", fontsize=9)
        label_slot = 0
        for row in g.itertuples():
            if row.event != "monitor":
                # Stagger label height by occurrence order (not just x-position)
                # so events landing close in time never stack on top of each
                # other or reach the figure's suptitle above the top row.
                offset_y = 8 + 10 * (label_slot % 3)
                axes[0, col].annotate(
                    row.event.replace("|", "+").replace("_", " "),
                    (row.time, row.model_epoch_serving),
                    xytext=(3, offset_y),
                    textcoords="offset points",
                    fontsize=6.2,
                    rotation=0,
                    ha="left",
                    va="bottom",
                )
                label_slot += 1
        top = max(1.0, float(g.model_epoch_serving.max())) + 1.9
        axes[0, col].set_ylim(-0.4, top)
        axes[1, col].step(g.time, g.queue_busy, where="post", color="darkorange", label="queue busy")
        axes[1, col].plot(g.time, g.audit_pending_labels, "o-", color="#555555", ms=3, label="pending audit labels")
        axes[1, col].set_ylabel("busy / pending")
        axes[1, col].legend(fontsize=6, loc="upper right")
        axes[2, col].step(g.time, g.nominal_budget_remaining, where="post", color="seagreen", label="budget left")
        axes[2, col].step(
            g.time, g.effective_completed_retrains, where="post", color="#9467bd", label="completed retrains"
        )
        axes[2, col].set_ylabel("counts")
        axes[2, col].set_xlabel("stream step")
        axes[2, col].legend(fontsize=6, loc="center right")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.subplots_adjust(top=0.90, hspace=0.55)
    fig.suptitle("Fig A6 — retraining and audit event time (real run data)", fontsize=10.5)
    fig.text(0.01, 0.0, CAPTIONS["fig-a6"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a6")
    csv = _write(frame, "fig-a6")
    return FigureResult("fig-a6", csv, png, len(frame), CAPTIONS["fig-a6"], {})


# --------------------------------------------------------------------------- #
# Fig A7 — latency and budget interaction across policies
# --------------------------------------------------------------------------- #
def figure_a7(run_id: str, seeds: tuple[int, ...] = (0, 1, 2)) -> FigureResult:
    sweep = sweep_policy_latency_budget(
        policies=("no_action", "threshold", "periodic", "cost_aware"),
        drift_regimes=("abrupt", "gradual", "recurring"),
        update_modes=("static", "incremental"),
        latency_settings=((0, 0), (1, 1), (2, 1)),
        budgets=(1, 3),
        seeds=seeds,
        horizon=24,
        retrain_charge=0.05,
        run_id=run_id,
    )
    frame = _with_provenance(pd.DataFrame([row.as_dict() for row in sweep]))

    policy_colors = {
        "no_action": "#7f7f7f",
        "threshold": "#1f77b4",
        "periodic": "#2ca02c",
        "cost_aware": "#d62728",
    }
    fig = plt.figure(figsize=(12.4, 9.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.3], hspace=0.62, wspace=0.22)
    regimes = ("abrupt", "gradual", "recurring")
    modes = ("static", "incremental")
    for r, regime in enumerate(regimes):
        for c, mode in enumerate(modes):
            ax = fig.add_subplot(gs[r, c])
            cell = frame[(frame.drift_regime == regime) & (frame.update_mode == mode)]
            grouped = (
                cell.groupby(["policy", "train_latency", "deploy_latency", "retrain_budget"])[
                    ["post_drift_risk", "completed_retrains", "stranded_budget", "requested_retrains"]
                ]
                .mean()
                .reset_index()
            )
            budgets_here = sorted(grouped.retrain_budget.unique())
            # budget=1 and budget=3 previously landed on identical (x, y) for
            # the same policy/latency and hid each other; give each budget a
            # distinct marker shape and a small x-jitter so both are visible.
            budget_markers = {b: m for b, m in zip(budgets_here, ("o", "^", "s", "D"))}
            jitter_step = 0.09
            for policy, g in grouped.groupby("policy"):
                for budget, gb in g.groupby("retrain_budget"):
                    slot = budgets_here.index(budget) - (len(budgets_here) - 1) / 2
                    x = gb.train_latency + gb.deploy_latency + slot * jitter_step
                    ax.scatter(
                        x,
                        gb.post_drift_risk,
                        s=25 + 45 * gb.completed_retrains,
                        color=policy_colors[policy],
                        alpha=0.8,
                        label=policy,
                        marker=budget_markers[budget],
                        edgecolor="k",
                        linewidth=0.3,
                    )
            ax.set_title(f"{regime} / {mode}", fontsize=8.5)
            ax.grid(alpha=0.25)
            if r == 2:
                ax.set_xlabel("train + deploy latency (steps)", fontsize=8)
            if c == 0:
                ax.set_ylabel("post-drift risk", fontsize=8)
            ax.tick_params(labelsize=7)
            if r == 0 and c == 0:
                policy_handles = [
                    plt.Line2D([], [], marker="o", color=policy_colors[p], linestyle="", label=p)
                    for p in policy_colors
                ]
                policy_legend = ax.legend(
                    handles=policy_handles, fontsize=6, loc="lower right", ncol=2, title="policy (color)", title_fontsize=6
                )
                ax.add_artist(policy_legend)
                budget_handles = [
                    plt.Line2D([], [], marker=m, color="#444444", linestyle="", label=f"budget={b}")
                    for b, m in budget_markers.items()
                ]
                ax.legend(handles=budget_handles, fontsize=6, loc="upper left", title="budget (shape)", title_fontsize=6)
    ax = fig.add_subplot(gs[3, :])
    bars = (
        frame[frame.retrain_budget == 3]
        .groupby(["policy", "train_latency", "deploy_latency"])[
            ["completed_retrains", "in_flight_at_horizon", "unrequested_capacity",
             "busy_rejected_requests", "budget_rejected_requests"]
        ]
        .mean()
        .reset_index()
    )
    labels = []
    positions = np.arange(len(bars))
    for i, row in enumerate(bars.itertuples()):
        labels.append(f"{row.policy}\n{row.train_latency}+{row.deploy_latency}")
        ax.bar(i, row.completed_retrains, color="#2ca02c", alpha=0.9, width=0.6, label="completed" if i == 0 else None)
        ax.bar(
            i, row.in_flight_at_horizon, bottom=row.completed_retrains, color="#f2a154",
            width=0.6, edgecolor="k", linewidth=0.3, label="in flight at horizon" if i == 0 else None,
        )
        base = row.completed_retrains + row.in_flight_at_horizon
        ax.bar(
            i, row.unrequested_capacity, bottom=base, color="#dddddd", width=0.6, edgecolor="k",
            linewidth=0.3, label="never requested" if i == 0 else None,
        )
    mean_rejected = float((bars.busy_rejected_requests + bars.budget_rejected_requests).mean())
    ax.set_xticks(positions, labels, fontsize=6.5)
    ax.set_ylim(0, 3.6)
    ax.set_ylabel("nominal budget 3:\ncompleted / in-flight / unrequested", fontsize=7.5)
    ax.set_title(
        "companion: budget 3 disaggregated — completed vs. still in flight at the horizon vs. never requested "
        f"(mean {mean_rejected:.2f} rejected requests per bar, not counted as budget loss)",
        fontsize=8,
    )
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Fig A7 — latency and budget interaction across policies (paired seeds, real sweep)", fontsize=10.5)
    fig.subplots_adjust(top=0.93, bottom=0.14)
    fig.text(0.01, 0.0, CAPTIONS["fig-a7"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a7")
    csv = _write(frame, "fig-a7")
    stats = {
        "rows": len(frame),
        "policies": sorted(frame.policy.unique().tolist()),
        "drift_regimes": sorted(frame.drift_regime.unique().tolist()),
    }
    return FigureResult("fig-a7", csv, png, len(frame), CAPTIONS["fig-a7"], stats)


# --------------------------------------------------------------------------- #
# Fig A8 — cumulative loss, retraining cost, and the conditional count bound
# --------------------------------------------------------------------------- #
@dataclass
class _BoundSetting:
    setting: str
    table: np.ndarray
    alpha: float
    declared_L: float
    theorem_applicable: bool
    note: str


def _a8_settings() -> list[_BoundSetting]:
    """Measured settings for the cumulative-cost and count-bound panels."""
    one_shot = improving_loss_table(8, stale_loss=0.30, fresh_loss=0.18)
    slow = improving_loss_table(12, stale_loss=0.24, fresh_loss=0.20)
    stepwise = stepwise_improvement_table(8, stale_loss=0.20, fresh_loss=0.16, epoch_gain=0.004)
    violating = improving_loss_table(8, stale_loss=0.30, fresh_loss=0.05)
    return [
        _BoundSetting("A: one retrain, low charge", one_shot, 0.02, observed_adjacent_gap_max(one_shot) + 1e-9,
                      True, "uniform adjacent-model gap checked on the loss table"),
        _BoundSetting("B: long horizon, low charge", slow, 0.05, observed_adjacent_gap_max(slow) + 1e-9,
                      True, "uniform adjacent-model gap checked on the loss table"),
        _BoundSetting("C: stepwise gain, small charge", stepwise, 0.001,
                      observed_adjacent_gap_max(stepwise) + 1e-9, True,
                      "uniform adjacent-model gap checked on the loss table"),
        _BoundSetting("D: high charge, no retrain", one_shot, 2.0, observed_adjacent_gap_max(one_shot) + 1e-9,
                      True, "high charge makes the no-retrain schedule optimal"),
        _BoundSetting("E: declared L violated", violating, 0.05, 0.02, False,
                      "observed adjacent gap exceeds the declared uniform L; no bound is claimed"),
    ]


def _schedule_rows(setting: _BoundSetting, kind: str, times: Sequence[int], cost: float, run_id: str, seed: int) -> list[dict[str, Any]]:
    losses = setting.table
    horizon = losses.shape[1]
    rows = []
    model = 0
    chosen = set(times)
    performance = 0.0
    charged = 0.0
    count = 0
    for t in range(horizon):
        if t in chosen:
            model = t
            count += 1
            charged += setting.alpha
        performance += float(losses[model, t])
        rows.append(
            {
                "time": t,
                "model_epoch": model,
                "cumulative_performance_cost": performance,
                "cumulative_retraining_cost": charged,
                "total_cost": performance + charged,
                "retrain_count": count,
                "horizon": horizon,
                "alpha": setting.alpha,
                "L": setting.declared_L,
                "observed_adjacent_gap_max": observed_adjacent_gap_max(losses),
                "bound_count": regol_count_bound(horizon, setting.alpha, setting.declared_L)
                if setting.theorem_applicable
                else float("nan"),
                "schedule_kind": kind,
                "setting": setting.setting,
                "theorem_applicable": bool(setting.theorem_applicable),
                "note": setting.note,
                "seed": seed,
                "run_id": run_id,
            }
        )
    return rows


def _bound_table_rows(run_id: str, seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    for setting in _a8_settings():
        losses = setting.table
        horizon = losses.shape[1]
        optimal_times, optimal_cost = optimal_retraining_schedule(losses, retrain_cost=setting.alpha)
        oracle_times, oracle_cost = brute_force_schedule(losses, retrain_cost=setting.alpha)
        none_cost = float(losses[0].sum())
        rows.append(
            {
                "setting": setting.setting,
                "horizon": horizon,
                "alpha": setting.alpha,
                "L": setting.declared_L,
                "observed_adjacent_gap_max": observed_adjacent_gap_max(losses),
                "theorem_applicable": bool(setting.theorem_applicable),
                "bound_count": regol_count_bound(horizon, setting.alpha, setting.declared_L)
                if setting.theorem_applicable
                else float("nan"),
                "optimized_count": len(optimal_times),
                "oracle_count": len(oracle_times),
                "no_action_total": none_cost,
                "optimized_total": optimal_cost,
                "oracle_total": oracle_cost,
                "note": setting.note,
                "seed": seed,
                "run_id": run_id,
            }
        )
    return rows


def figure_a8(run_id: str, seed: int = 0) -> FigureResult:
    rows: list[dict[str, Any]] = []
    for setting in _a8_settings():
        losses = setting.table
        horizon = losses.shape[1]
        optimal_times, optimal_cost = optimal_retraining_schedule(losses, retrain_cost=setting.alpha)
        oracle_times, oracle_cost = brute_force_schedule(losses, retrain_cost=setting.alpha)
        rows.extend(_schedule_rows(setting, "no_action", (), float(losses[0].sum()), run_id, seed))
        rows.extend(_schedule_rows(setting, "optimized", optimal_times, optimal_cost, run_id, seed))
        rows.extend(_schedule_rows(setting, "brute_force_oracle", oracle_times, oracle_cost, run_id, seed))
        bound_row = _schedule_rows(setting, "regol_bound", optimal_times, optimal_cost, run_id, seed)[-1].copy()
        bound_row["time"] = horizon - 1
        bound_row["retrain_count"] = float("nan")
        bound_row["cumulative_performance_cost"] = float("nan")
        bound_row["cumulative_retraining_cost"] = float("nan")
        bound_row["total_cost"] = float("nan")
        rows.append(bound_row)
    frame = _with_provenance(pd.DataFrame(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    palette = {
        "A: one retrain, low charge": "#1f77b4",
        "B: long horizon, low charge": "#2ca02c",
        "C: stepwise gain, small charge": "#ff7f0e",
        "D: high charge, no retrain": "#9467bd",
        "E: declared L violated": "#d62728",
    }
    ax = axes[0]
    for setting, color in palette.items():
        for kind, style, width in (("optimized", "-", 1.8), ("brute_force_oracle", "--", 1.0), ("no_action", ":", 1.0)):
            g = frame[(frame.setting == setting) & (frame.schedule_kind == kind)].sort_values("time")
            if not len(g):
                continue
            ax.plot(g.time, g.total_cost, style, color=color, lw=width)
    for setting, color in palette.items():
        ax.plot([], [], "-", color=color, label=setting)
    ax.plot([], [], "k-", label="solid optimized")
    ax.plot([], [], "k--", label="dashed exhaustive reference")
    ax.plot([], [], "k:", label="dotted no-action")
    ax.set_xlabel("horizon step")
    ax.set_ylabel("cumulative total cost (loss + charges)")
    ax.set_title("cumulative cost trajectories", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    table = pd.DataFrame(_bound_table_rows(run_id, seed))
    for setting, color in palette.items():
        g = table[table.setting == setting]
        if not len(g):
            continue
        applicable = bool(g.theorem_applicable.iloc[0])
        x = float(g.alpha.iloc[0])
        ax.scatter(x, g.optimized_count.iloc[0], s=70, color=color, marker="o" if applicable else "x",
                   label=f"{setting.split(':')[0]}: optimized")
        if applicable:
            ax.scatter(x, g.bound_count.iloc[0], s=110, facecolor="none", edgecolor=color, marker="s",
                       label=f"{setting.split(':')[0]}: ceiling")
    ax.set_xlabel("retraining charge alpha")
    ax.set_ylabel("retrain count")
    ax.set_title("count versus the conditional Regol-style ceiling", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Fig A8 — cumulative cost and the conditional count bound (real run data)", fontsize=10.5)
    fig.text(0.01, -0.06, CAPTIONS["fig-a8"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a8")
    csv = _write(frame, "fig-a8")
    stats = {"settings": [s.setting for s in _a8_settings()], "violating_rows": int((~frame.theorem_applicable).sum())}
    return FigureResult("fig-a8", csv, png, len(frame), CAPTIONS["fig-a8"], stats)


# --------------------------------------------------------------------------- #
# extra evidence artifacts (bound table, delayed audit, observable invariance)
# --------------------------------------------------------------------------- #
def bound_table_artifact(run_id: str, seed: int = 0) -> FigureResult:
    frame = _with_provenance(pd.DataFrame(_bound_table_rows(run_id, seed)))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = np.arange(len(frame))
    width = 0.35
    applicable = frame.theorem_applicable.astype(bool)
    ax.bar(x - width / 2, frame.optimized_count, width, color="#1f77b4", label="optimized retrain count")
    ax.bar(x + width / 2, frame.bound_count.fillna(0.0), width, color="#dddddd", edgecolor="k",
           linewidth=0.5, label="conditional ceiling (verified L)")
    top = float(pd.concat([frame.optimized_count, frame.bound_count.fillna(0.0)]).max())
    ax.set_ylim(0, top + 1.4)
    for i, row in enumerate(frame.itertuples()):
        if not row.theorem_applicable:
            bar_top = max(row.optimized_count, 0.0)
            ax.text(
                i, bar_top + 0.25, "outside\ntheorem scope", ha="center", va="bottom",
                fontsize=6.5, color="#d62728",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
            )
    ax.set_xticks(x, [s.split(":")[0] for s in frame.setting], fontsize=8)
    ax.set_ylabel("retrain count")
    ax.set_title("bound table: optimized count and conditional ceiling per setting", fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7)
    png = _save(fig, "fig-a8-bound")
    csv = _write(frame, "fig-a8-bound-table")
    return FigureResult("fig-a8-bound", csv, png, len(frame), "bound table", {})


def delayed_audit_artifact(run_id: str, seed: int = 914, delay: int = 4, horizon: int = 24) -> FigureResult:
    scenario = make_shift_scenario(seed=seed, regime="harmful_concept", n_source=320, n_target=512)
    stream = scenario.sample_deployment(seed=2718, n_steps=horizon, batch_size=32)
    queue = DelayedLabelQueue()
    released_losses: list[float] = []
    rows: list[dict[str, Any]] = []
    for t in range(horizon):
        _, _, _, loss_t = stream.batch(t)
        if t >= 1:
            queue.submit(f"audit-{t:03d}", float(loss_t[0]), available_at=t + delay, submitted_at=t)
        released = queue.release(now=t)
        released_losses.extend(float(value) for _, value in released)
        rows.append(
            {
                "step": t,
                "time": t,
                "submitted_this_step": int(t >= 1),
                "released_this_step": len(released),
                "released_sample_ids": "|".join(sample_id for sample_id, _ in released),
                "pending_labels": queue.pending_count,
                "total_released": queue.released_count,
                "audited_loss_mean": float(np.mean(released_losses)) if released_losses else float("nan"),
                "queue_delay": delay,
                "seed": seed,
                "run_id": run_id,
            }
        )
    frame = _with_provenance(pd.DataFrame(rows))
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.step(frame.step, frame.pending_labels, where="post", color="#555555", label="pending labels")
    ax.step(frame.step, frame.total_released, where="post", color="#1f77b4", label="released (usable) labels")
    ax.set_xlabel("stream step")
    ax.set_ylabel("labels")
    ax2 = ax.twinx()
    ax2.plot(frame.step, frame.audited_loss_mean, "o-", color="#d62728", ms=3, label="audited loss mean")
    ax2.set_ylabel("audited loss mean", color="#d62728")
    ax.set_title(f"delayed audit queue (release delay {delay} steps): hidden until available", fontsize=9.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
    png = _save(fig, "delayed-audit-timeline")
    csv = _write(frame, "delayed-audit-timeline")
    return FigureResult("delayed-audit-timeline", csv, png, len(frame), "delayed audit timeline", {})


def observable_invariance_artifact(run_id: str, seed: int = 914, stream_seed: int = 2718, signal_seed: int = 43) -> FigureResult:
    scenario = make_shift_scenario(seed=seed, regime="harmful_concept", n_source=320, n_target=512)
    stream = scenario.sample_deployment(seed=stream_seed, n_steps=8, batch_size=64)
    counterfactual_y = 1 - stream.y.copy()
    counterfactual_loss = np.square(stream.champion_score - counterfactual_y)
    counterfactual = replace(stream, y=counterfactual_y, loss=counterfactual_loss)
    observed = compute_signals(scenario, stream, seed=signal_seed)
    changed = compute_signals(scenario, counterfactual, seed=signal_seed)
    signals_equal = all(
        bool(np.array_equal(getattr(observed, name), getattr(changed, name)))
        for name in ("drift", "gap", "seq_rate", "seq_corroborated")
    )
    threshold = float(np.max(observed.gap) - 1.0)
    observed_policy = replace(observed, gap_threshold=threshold)
    changed_policy = replace(changed, gap_threshold=threshold)
    observed_triggers = [t for t in range(stream.n_steps) if _trigger("threshold", observed_policy, t)]
    changed_triggers = [t for t in range(stream.n_steps) if _trigger("threshold", changed_policy, t)]
    record = {
        "run_id": run_id,
        "provenance": f"seed={seed}|stream_seed={stream_seed}|run_id={run_id}|producer=exec/figures.py",
        "n_steps": int(stream.n_steps),
        "signals_equal": bool(signals_equal),
        "trigger_schedule_equal": bool(observed_triggers == changed_triggers),
        "observed_triggers": observed_triggers,
        "counterfactual_triggers": changed_triggers,
        "x_unchanged": bool(np.array_equal(stream.x, counterfactual.x)),
        "y_changed": bool(not np.array_equal(stream.y, counterfactual.y)),
        "loss_changed": bool(not np.array_equal(stream.loss, counterfactual.loss)),
        "gap_threshold_used": threshold,
    }
    path = FIGDIR / "observable-invariance.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return FigureResult("observable-invariance", str(path), "", 1, "observable invariance record", record)


# --------------------------------------------------------------------------- #
# Fig A9 — immediate vs. delayed retraining decisions (C14, derived here)
# --------------------------------------------------------------------------- #
def figure_a9(
    run_id: str, seeds: tuple[int, ...] = tuple(range(10)), delays: tuple[int, ...] = (0, 2, 4)
) -> FigureResult:
    """Connect the delayed-audit queue to the actual policy decision loop.

    A6's request-time timeline fixed ``request_times=[1, 2]`` and never
    compared an immediate-decision arm against a delayed one; this figure
    runs the real ``threshold`` policy through ``run_stream(label_delay=...)``
    across paired seeds so the immediate-vs-delayed trade-off is measured,
    not assumed.
    """
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        scenario = make_shift_scenario(seed=seed, regime="harmful_concept", n_source=320, n_target=256)
        stream = scenario.sample_deployment(seed=seed + 7)
        signals = compute_signals(scenario, stream, seed=seed)
        for delay in delays:
            outcome = run_stream(scenario, stream, "threshold", signals, seed=seed, label_delay=delay)
            rows.append(
                {
                    "label_delay": delay,
                    "seed": seed,
                    "run_id": run_id,
                    "detection_delay": outcome.detection_delay,
                    "missed": int(outcome.missed),
                    "false_retrain": int(outcome.false_retrain),
                    "recovered": int(outcome.recovered),
                    "retrains": outcome.retrains,
                    "cost": outcome.cost,
                }
            )
    frame = _with_provenance(pd.DataFrame(rows))

    agg = (
        frame.groupby("label_delay")
        .agg(
            median_detection_delay=("detection_delay", "median"),
            miss_rate=("missed", "mean"),
            false_retrain_rate=("false_retrain", "mean"),
            recovery_rate=("recovered", "mean"),
            mean_cost=("cost", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax = axes[0]
    for seed, g in frame.groupby("seed"):
        g = g.sort_values("label_delay")
        ax.plot(g.label_delay, g.detection_delay, "-o", color="#1f77b4", alpha=0.3, ms=3, lw=1)
    ax.plot(agg.label_delay, agg.median_detection_delay, "-o", color="#000000", lw=2.2, label="median across paired seeds")
    for row in agg.itertuples():
        y = row.median_detection_delay if np.isfinite(row.median_detection_delay) else 0.0
        ax.text(row.label_delay, y + 1.1, f"miss={row.miss_rate:.0%}", ha="center", fontsize=6.5, color="#d62728")
    ax.set_xlabel("label delay (steps)")
    ax.set_ylabel("detection delay (steps after onset)")
    ax.set_title("per-seed detection timing (gaps = missed within horizon)", fontsize=8.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    width = 0.35
    x = np.arange(len(agg))
    ax.bar(x - width / 2, agg.miss_rate, width, color="#d62728", label="miss rate")
    ax.bar(x + width / 2, agg.false_retrain_rate, width, color="#9467bd", label="false-retrain rate")
    for i, row in enumerate(agg.itertuples()):
        ax.text(i, max(row.miss_rate, row.false_retrain_rate) + 0.03, f"cost={row.mean_cost:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [f"delay={int(d)}" for d in agg.label_delay], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("rate across paired seeds")
    ax.set_title("miss / false-retrain rate by label delay (mean cost annotated)", fontsize=8.5)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=7)

    fig.suptitle("Fig A9 — immediate vs. delayed retraining decisions (C14, real paired-seed run)", fontsize=10.5)
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.01, 0.02, CAPTIONS["fig-a9"], fontsize=7.5, va="top", wrap=True)
    png = _save(fig, "fig-a9")
    csv = _write(frame, "fig-a9")
    stats = {"delays": list(delays), "seeds": len(seeds)}
    return FigureResult("fig-a9", csv, png, len(frame), CAPTIONS["fig-a9"], stats)


def render_all(run_id: str | None = None) -> dict[str, FigureResult]:
    run_id = run_id or _stamp()
    results = {
        "fig-a1": figure_a1(run_id),
        "fig-a2": figure_a2(run_id),
        "fig-a3": figure_a3(run_id),
        "fig-a4": figure_a4(run_id),
        "fig-a5": figure_a5(run_id),
        "fig-a6": figure_a6(run_id),
        "fig-a7": figure_a7(run_id),
        "fig-a9": figure_a9(run_id),
        "fig-a8": figure_a8(run_id),
        "fig-a8-bound": bound_table_artifact(run_id),
        "delayed-audit-timeline": delayed_audit_artifact(run_id),
        "observable-invariance": observable_invariance_artifact(run_id),
    }
    manifest = {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figures": {
            key: {
                "csv": value.csv_path,
                "png": value.png_path,
                "rows": value.rows,
                "stats": value.stats,
            }
            for key, value in results.items()
        },
    }
    (FIGDIR / "run-manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return results
