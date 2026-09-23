"""R2_NB2_stream_anatomy: E1 (pre) + E2 (pre) + E3 (pre) of the shared realistic stream.

Panels
------
E1 (three stacked axes sharing the day axis): daily fraud prevalence, the
fraud-typology mix inside the fraud, and the daily standardised mean of the
drifting feature per typology, with the drift day marked on all three.
E2: the realised label confirmation per fraud typology against label age.
E3: two feature-space scatters for the two ambiguous regimes (first-party fraud
versus plain default; synthetic identity versus thin file), each with the
out-of-fold AUC of a strong learner printed on it.

Everything is computed from ``exec.data.generate_realistic_stream`` (audited by
``audit_stream``); nothing is read from ``viz/mock``. The strong-learner AUC is
the same out-of-fold HistGradientBoostingClassifier measurement the C17 contract
test uses, on the latent-free features only.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from exec.config import config_hash, seed_of
from exec.data import audit_stream, generate_realistic_stream
from exec.vizlib import save_r2_figure, write_r2_sidecar

NAME = "R2_NB2_stream_anatomy"

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

#: One colour per fraud typology, held fixed across every panel of the figure.
TYPOLOGY_COLORS = {
    "first_party_fraud": BLUE,
    "synthetic_identity": GREEN,
    "account_takeover": VERM,
}
PREVALENCE_COLOR = INK

#: Day bins across the stream (one "day" is one bin of the arrival order), the
#: centred rolling window used to smooth the mix/feature series, the maturation
#: age grid, and the cap on confuser points drawn in the E3 scatters.
DAY_BINS = 30
ROLLING_WINDOW = 9
MATURATION_AGES = tuple(range(0, 91, 2))
CONFUSER_PLOT_SAMPLE = 300

#: Verbatim two-line "How to read this chart" block from
#: ``viz/spec/R2_NB2_stream_anatomy.md`` (wrapped for the PNG footer only).
HOW_TO_READ_LINE_1 = (
    "Left: daily fraud prevalence, typology mix and the standardised mean of the "
    "drifting feature against time, with a dashed line at the drift day. Middle: "
    "fraction of each typology's fraud confirmed as label age grows (days). Right: "
    "two feature-space scatters, each of two look-alike classes, with the best AUC a "
    "strong learner can reach printed on each."
)
HOW_TO_READ_LINE_2 = (
    "Look for: a step at the drift day in one typology, curves that differ by "
    "typology, and overlapping clouds with best AUC well under 0.97. For the job: it "
    "shows the stream is hard in the way real fraud data is (rare, drifting, "
    "late-labelled, ambiguous), so later naive-versus-aware gaps are earned, not "
    "handed out."
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


def _rolling_mean(values: Any, window: int) -> np.ndarray:
    """Centred rolling mean that ignores NaN entries (all-NaN window -> NaN)."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    half = window // 2
    for index in range(values.size):
        window_values = values[max(0, index - half) : index + half + 1]
        if np.isfinite(window_values).any():
            out[index] = float(np.nanmean(window_values))
    return out


def _oof_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Out-of-fold probability of latent Y from a strong learner (C17 definition)."""
    learner = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, max_depth=4, random_state=0
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_predict(
        learner, features, labels, cv=cv, method="predict_proba"
    )[:, 1]


# ------------------------------------------------------------------ aggregation
def _aggregate(stream: Mapping[str, Any], audit: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transactions = stream["transactions"]
    n = int(stream["n"])
    t_drift = int(stream["t_drift"])
    names = list(stream["feature_names"])
    frauds = [str(t) for t in stream["fraud_typologies"]]
    drift_typology = str(stream["drift_typology"])
    seed = seed_of(stream)

    arrival = np.asarray([int(t["arrival_index"]) for t in transactions], dtype=np.int64)
    labels = np.asarray([int(t["Y"]) for t in transactions], dtype=np.int64)
    typology = np.asarray([str(t["typology"]) for t in transactions], dtype=object)
    delay = np.asarray([int(t["label_delay"]) for t in transactions], dtype=np.int64)
    features = np.asarray(
        [[float(t["features"][name]) for name in names] for t in transactions], dtype=float
    )

    day = np.minimum(arrival * DAY_BINS // n, DAY_BINS - 1)
    t_drift_day = int(t_drift * DAY_BINS // n)
    pre = arrival < t_drift
    post = ~pre

    rows: list[dict[str, Any]] = []

    # --- E1 prevalence ------------------------------------------------------
    day_count = np.bincount(day, minlength=DAY_BINS)
    prevalence = np.asarray(
        [float(labels[day == k].mean()) for k in range(DAY_BINS)], dtype=float
    )
    for k in range(DAY_BINS):
        rows.append(
            {
                "panel": "E1_prevalence",
                "series": "prevalence",
                "group": "",
                "x": int(k),
                "y": round(float(prevalence[k]), 8),
                "n": int(day_count[k]),
            }
        )
    rows.append({"panel": "E1_prevalence", "series": "t_drift", "x": int(t_drift_day)})

    # --- E1 typology mix (centred rolling share over the fixed day grid) -----
    mix_counts = np.asarray(
        [
            np.bincount(day[(typology == tp) & (labels == 1)], minlength=DAY_BINS)
            for tp in frauds
        ],
        dtype=float,
    )
    mix_roll = np.asarray([_rolling_mean(counts, ROLLING_WINDOW) for counts in mix_counts])
    mix_total = mix_roll.sum(axis=0)
    for index, tp in enumerate(frauds):
        shares = np.divide(
            mix_roll[index], mix_total, out=np.full(DAY_BINS, np.nan), where=mix_total > 0
        )
        for k in range(DAY_BINS):
            if not np.isfinite(shares[k]):
                continue
            rows.append(
                {
                    "panel": "E1_typology_mix",
                    "series": "typology_share",
                    "group": tp,
                    "x": int(k),
                    "y": round(float(shares[k]), 8),
                    "n": int(((labels == 1) & (day == k)).sum()),
                }
            )

    # --- E1 drifting feature (measured choice, pre-drift standardisation) ----
    effect = []
    for j in range(len(names)):
        taken = features[(typology == drift_typology) & pre, j]
        after = features[(typology == drift_typology) & post, j]
        pooled = math.sqrt((taken.var(ddof=1) + after.var(ddof=1)) / 2.0)
        effect.append(abs(after.mean() - taken.mean()) / pooled if pooled > 0 else 0.0)
    feature_index = int(np.argmax(effect))
    feature_name = str(names[feature_index])
    anchored = features[(typology == drift_typology) & pre, feature_index]
    mu = float(anchored.mean())
    sd = float(anchored.std())
    z = (features[:, feature_index] - mu) / (sd + 1e-12)

    for tp in frauds:
        mask = typology == tp
        per_day = np.full(DAY_BINS, np.nan, dtype=float)
        for k in range(DAY_BINS):
            selected = mask & (day == k)
            if bool(selected.any()):
                per_day[k] = float(z[selected].mean())
        curve = _rolling_mean(per_day, ROLLING_WINDOW)
        for k in range(DAY_BINS):
            if not np.isfinite(curve[k]):
                continue
            rows.append(
                {
                    "panel": "E1_feature_drift",
                    "series": "feature_mean",
                    "group": tp,
                    "x": int(k),
                    "y": round(float(curve[k]), 8),
                    "n": int((mask & (day == k)).sum()),
                }
            )

    # --- E2 realised maturation ---------------------------------------------
    for tp in frauds:
        mask = typology == tp
        cases = int(mask.sum())
        for age in MATURATION_AGES:
            confirmed = int((delay[mask] <= age).sum())
            low, high = _wilson(confirmed, cases)
            rows.append(
                {
                    "panel": "E2_maturation",
                    "series": "maturation",
                    "group": tp,
                    "x": int(age),
                    "y": round(confirmed / cases, 8),
                    "y_low": round(low, 8),
                    "y_high": round(high, 8),
                    "n": cases,
                }
            )

    # --- E3 ambiguous regimes -------------------------------------------------
    oof = _oof_scores(features, labels)
    rng = np.random.default_rng(seed)
    regime_meta: dict[str, Any] = {}
    for regime in stream["ambiguous_regimes"]:
        regime_name = str(regime["name"])
        fraud_typology = str(regime["fraud_typology"])
        confuser_typology = str(regime["confuser_typology"])
        mask = np.isin(typology, [fraud_typology, confuser_typology])
        pair_labels = labels[mask]
        single_aucs = [roc_auc_score(pair_labels, features[mask, j]) for j in range(len(names))]
        top = sorted(
            range(len(names)), key=lambda j: -abs(single_aucs[j] - 0.5)
        )[:2]
        x_index, y_index = int(top[0]), int(top[1])
        x_mu, x_sd = features[mask, x_index].mean(), features[mask, x_index].std()
        y_mu, y_sd = features[mask, y_index].mean(), features[mask, y_index].std()
        xs = (features[:, x_index] - x_mu) / (x_sd + 1e-12)
        ys = (features[:, y_index] - y_mu) / (y_sd + 1e-12)

        panel = (
            "E3_first_party_vs_default"
            if fraud_typology == "first_party_fraud"
            else "E3_synthetic_vs_thin_file"
        )
        fraud_mask = typology == fraud_typology
        confuser_mask = typology == confuser_typology
        confuser_indices = np.flatnonzero(confuser_mask)
        if confuser_indices.size > CONFUSER_PLOT_SAMPLE:
            confuser_indices = rng.choice(
                confuser_indices, size=CONFUSER_PLOT_SAMPLE, replace=False
            )
        for index in np.flatnonzero(fraud_mask):
            rows.append(
                {
                    "panel": panel,
                    "series": fraud_typology,
                    "x": round(float(xs[index]), 8),
                    "y": round(float(ys[index]), 8),
                }
            )
        for index in confuser_indices:
            rows.append(
                {
                    "panel": panel,
                    "series": confuser_typology,
                    "x": round(float(xs[index]), 8),
                    "y": round(float(ys[index]), 8),
                }
            )
        best_auc = float(roc_auc_score(pair_labels, oof[mask]))
        rows.append(
            {
                "panel": panel,
                "series": "best_auc",
                "y": round(best_auc, 8),
                "label": f"best achievable AUC {best_auc:.2f} (HistGB, 5-fold OOF)",
            }
        )
        regime_meta[panel] = {
            "name": regime_name,
            "fraud_typology": fraud_typology,
            "confuser_typology": confuser_typology,
            "x_feature": str(names[x_index]),
            "y_feature": str(names[y_index]),
            "best_auc": best_auc,
            "n_fraud": int(fraud_mask.sum()),
            "n_confuser": int(confuser_mask.sum()),
            "n_confuser_plotted": int(confuser_indices.size),
        }

    meta = {
        "panel": NAME,
        "n": n,
        "t_drift": t_drift,
        "t_drift_day": t_drift_day,
        "day_bins": DAY_BINS,
        "records_per_day": int(day_count[0]),
        "rolling_window": ROLLING_WINDOW,
        "fraud_typologies": frauds,
        "drift_typology": drift_typology,
        "drift_feature": feature_name,
        "drift_cohen_d": round(float(effect[feature_index]), 6),
        "prevalence_mean": round(float(labels.mean()), 8),
        "prevalence_min": round(float(prevalence.min()), 8),
        "prevalence_max": round(float(prevalence.max()), 8),
        "typology_counts": {
            tp: int((typology == tp).sum()) for tp in frauds
        },
        "regimes": regime_meta,
        "data_hash": str(audit["data_hash"]),
        "n_observed_labels": int(audit["n_observed_labels"]),
    }
    return rows, meta


def _cached(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = config_hash(config)
    if key not in _CACHE:
        stream = generate_realistic_stream(config)
        audit = audit_stream(stream)
        _CACHE[key] = _aggregate(stream, audit)
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


def _caption_text(width: int) -> str:
    first = textwrap.fill("How to read this chart: " + HOW_TO_READ_LINE_1, width=width)
    second = textwrap.fill(HOW_TO_READ_LINE_2, width=width)
    return first + "\n" + second


def _new_figure(width: float, height: float, cap_height: float) -> tuple[Any, Any]:
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
    fig = plt.figure(figsize=(width, height), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, cap_height])
    caption = fig.add_subplot(grid[1])
    caption.axis("off")
    caption.text(
        0.0,
        1.0,
        _caption_text(int(width * 13)),
        ha="left",
        va="top",
        fontsize=9.5,
        color=INK,
        transform=caption.transAxes,
    )
    return fig, grid[0]


def _render(rows: list[dict[str, Any]], meta: Mapping[str, Any]) -> Any:
    frauds = list(meta["fraud_typologies"])
    by = lambda panel, series: [r for r in rows if r["panel"] == panel and r["series"] == series]
    fig, top = _new_figure(21.0, 10.8, 0.15)
    grid = top.subgridspec(1, 3, width_ratios=[1.18, 1.0, 1.0], wspace=0.26)

    # --- column 1: E1 ------------------------------------------------------
    e1 = [fig.add_subplot(spec) for spec in grid[0].subgridspec(3, 1, hspace=0.12)]
    for index in range(1, 3):
        e1[index].sharex(e1[0])
    t_drift_day = float(meta["t_drift_day"])

    prevalence = sorted(by("E1_prevalence", "prevalence"), key=lambda r: r["x"])
    e1[0].plot(
        [r["x"] for r in prevalence],
        [100.0 * r["y"] for r in prevalence],
        color=PREVALENCE_COLOR,
        lw=1.8,
    )
    e1[0].set(ylabel="fraud prevalence (%)", ylim=(0, None))
    _style(e1[0])
    _head(
        e1[0],
        f"E1 (pre) daily fraud prevalence (n = {meta['records_per_day']:,} records/day)",
        [_line("prevalence", PREVALENCE_COLOR, lw=1.8), _line("drift day", GREY, "--")],
        2,
    )

    shares = {
        tp: sorted(by("E1_typology_mix", "typology_share"), key=lambda r: r["x"])
        for tp in frauds
    }
    xs = [r["x"] for r in shares[frauds[0]]]
    e1[1].stackplot(
        xs,
        [[r["y"] for r in shares[tp]] for tp in frauds],
        colors=[TYPOLOGY_COLORS[tp] for tp in frauds],
        alpha=1.0,
        lw=0,
    )
    e1[1].set(ylabel="share of fraud cases", ylim=(0, 1), xlim=(min(xs), max(xs)))
    _style(e1[1])
    _head(
        e1[1],
        f"E1 (pre) typology mix ({meta['rolling_window']}-day centred share)",
        [Patch(color=TYPOLOGY_COLORS[tp], label=tp) for tp in frauds],
        3,
    )

    feature_rows = by("E1_feature_drift", "feature_mean")
    for tp in frauds:
        series = sorted((r for r in feature_rows if r["group"] == tp), key=lambda r: r["x"])
        e1[2].plot([r["x"] for r in series], [r["y"] for r in series], color=TYPOLOGY_COLORS[tp], lw=1.8)
    e1[2].axhline(0.0, color=INK, lw=1.0, ls=":", zorder=0)
    e1[2].set(
        xlabel="day (arrival-order bin)",
        ylabel=f"{meta['drift_feature']} (pre-drift z-units)",
        xlim=(min(xs), max(xs)),
    )
    _style(e1[2])
    _head(
        e1[2],
        f"E1 (pre) drifting feature: {meta['drift_feature']} (Cohen's d = {meta['drift_cohen_d']:.2f})",
        [_line(tp, TYPOLOGY_COLORS[tp]) for tp in frauds],
        3,
    )
    for axis in e1:
        axis.axvline(t_drift_day, color=GREY, ls="--", lw=1.3)
    for axis in e1[:2]:
        axis.tick_params(labelbottom=False)

    # --- column 2: E2 ------------------------------------------------------
    maturation = by("E2_maturation", "maturation")
    axis = fig.add_subplot(grid[1])
    for tp in frauds:
        series = sorted((r for r in maturation if r["group"] == tp), key=lambda r: r["x"])
        xs = [r["x"] for r in series]
        ys = [r["y"] for r in series]
        axis.fill_between(xs, [r["y_low"] for r in series], [r["y_high"] for r in series],
                          color=TYPOLOGY_COLORS[tp], alpha=0.18, lw=0)
        axis.plot(xs, ys, color=TYPOLOGY_COLORS[tp], lw=2.2)
        axis.text(
            max(xs) + 2,
            ys[-1],
            f"{tp} (n={series[-1]['n']})",
            color=INK,
            fontsize=8.5,
            va="center",
        )
    axis.set(
        xlabel="label age (days since transaction)",
        ylabel="fraction of fraud confirmed",
        ylim=(0, 1.02),
        xlim=(0, 132),
    )
    _style(axis)
    _head(
        axis,
        "E2 (pre) realised maturation by typology",
        [_line(tp, TYPOLOGY_COLORS[tp], lw=2.2) for tp in frauds],
        3,
    )

    # --- column 3: E3 ------------------------------------------------------
    scatter_axes = [fig.add_subplot(spec) for spec in grid[2].subgridspec(2, 1, hspace=0.24)]
    panels = ("E3_first_party_vs_default", "E3_synthetic_vs_thin_file")
    titles = (
        "E3 (pre) first-party fraud vs plain default",
        "E3 (pre) synthetic identity vs thin file",
    )
    for axis, panel, title in zip(scatter_axes, panels, titles):
        regime = meta["regimes"][panel]
        for series_name, color in (
            (regime["fraud_typology"], BLUE),
            (regime["confuser_typology"], VERM),
        ):
            series = by(panel, series_name)
            axis.scatter(
                [r["x"] for r in series],
                [r["y"] for r in series],
                s=10,
                alpha=0.45,
                color=color,
                linewidths=0,
            )
        best = by(panel, "best_auc")[0]
        axis.text(
            0.98,
            0.03,
            best["label"],
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            bbox=dict(fc="white", ec="#BBBBBB", pad=3),
        )
        axis.set(
            xlabel=f"{regime['x_feature']} (standardised)",
            ylabel=f"{regime['y_feature']} (standardised)",
        )
        _style(axis, "both")
        _head(
            axis,
            f"{title}\n"
            f"{regime['fraud_typology']} (n = {regime['n_fraud']}) vs "
            f"{regime['confuser_typology']} ({regime['n_confuser_plotted']} of {regime['n_confuser']:,} shown)",
            [
                Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, ms=6,
                       label=f"{regime['fraud_typology']} (plotted)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor=VERM, ms=6,
                       label=f"{regime['confuser_typology']} (subsample)"),
            ],
            2,
        )
    return fig


def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Compute the shared stream, write the PNG + sidecar, and report the run."""
    rows, meta = _cached(config)
    sidecar_path = write_r2_sidecar(NAME, rows, config, out_dir=out_dir)
    fig = _render(rows, meta)
    figure_path = save_r2_figure(fig, NAME, out_dir=out_dir)
    plt.close(fig)
    print("How to read this chart: " + HOW_TO_READ_LINE_1)
    print(HOW_TO_READ_LINE_2)
    print(
        f"{NAME}: {len(rows)} rows; prevalence {meta['prevalence_mean']:.4%}; "
        f"drift feature {meta['drift_feature']} (d = {meta['drift_cohen_d']:.2f}); "
        f"best AUCs "
        + ", ".join(
            f"{panel} {meta['regimes'][panel]['best_auc']:.3f}" for panel in meta["regimes"]
        )
    )
    return {
        "figure": str(figure_path),
        "sidecar": str(sidecar_path),
        "rows": int(len(rows)),
        "name": NAME,
    }
