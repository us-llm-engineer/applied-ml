"""NB2 selection-delay audit figure.

Decision rules used throughout this audit:
* model flag rule -- a transaction is flagged (would be declined) iff
  ``risk_score >= FLAG_THRESHOLD`` (0.50);
* approval boundary -- the business approves a transaction iff
  ``risk_score < data["approval_boundary"]`` (0.70 in the canonical run), so the
  accepts-only evaluation view is exactly that approved subpopulation and the
  boundary is annotated on the pre panel's risk-decile axis.

Panels
------
pre  label maturation (share of an age/risk bucket whose label has arrived by
     the audit age) across audit ages and risk deciles, approval boundary shown;
mid  each evaluation view's operational metric against its latent-truth value;
post operational-minus-latent metric gaps with a seeded percentile bootstrap
     interval over transactions (400 resamples by default).

The audit reads only the real ``generate_synthetic_data`` stream -- never
``viz/mock`` -- and the two orders in the pre panel are the audit ages a
transaction has survived. Latent ``Y`` and the true delay are audit-only
ground truth; the metrics cannot see them through the operational views.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exec.config import seed_of
from exec.data import generate_synthetic_data
from exec.vizlib import save_figure, write_sidecar

NAME = "NB2_selection_delay_audit"

#: Model decision rule (stated in the module docstring).
FLAG_THRESHOLD = 0.50

PANEL_AGES = (3, 7, 14, 21)
PANEL_DECILES = (1, 3, 5, 7, 9)
EVALUATION_VIEWS = ("latent_truth", "unresolved_as_negative", "accepts_only")
METRICS = ("precision", "recall", "calibration")

DEFAULT_BOOTSTRAP_RESAMPLES = 400
DEFAULT_BOOTSTRAP_LEVEL = 0.95

#: Colour map mirrors the frozen mock palette in ``viz/mock/render_mock_figures.py``.
VIEW_COLORS = {
    "latent_truth": "#0072B2",
    "unresolved_as_negative": "#D55E00",
    "accepts_only": "#009E73",
}
VIEW_LABELS = {
    "latent_truth": "latent truth",
    "unresolved_as_negative": "unresolved as negative",
    "accepts_only": "accepts only",
}

HOW_TO_READ = (
    "How to read: pre colours each age/risk bucket by the share of its labels that have arrived by "
    "that age -- high-decile cases resolve fast, low-decile cases stay unresolved -- and the dashed "
    "line marks the approval boundary.\n"
    "Mid and post compare each evaluation view with the latent-truth metric over the same population; "
    "post is the operational-minus-latent gap with a 95% bootstrap interval over transactions, so an "
    "interval crossing zero means that view is compatible with latent truth."
)


def _metric_values(
    risk: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    approved: np.ndarray,
    view: str,
) -> dict[str, float | None]:
    """Compute precision, recall, and calibration for one evaluation view."""
    flagged = risk >= FLAG_THRESHOLD
    if view == "latent_truth":
        truth = y
        selected = np.ones_like(y, dtype=bool)
    elif view == "unresolved_as_negative":
        truth = y * observed
        selected = np.ones_like(y, dtype=bool)
    elif view == "accepts_only":
        truth = y
        selected = approved & (observed == 1)
    else:
        raise ValueError(f"unknown evaluation view: {view!r}")

    flagged_true = flagged & selected & (truth == 1)
    n_flagged = int((flagged & selected).sum())
    n_positive = int(((truth == 1) & selected).sum())
    n_selected = int(selected.sum())
    return {
        "precision": float(flagged_true.sum()) / n_flagged if n_flagged else None,
        "recall": float(flagged_true.sum()) / n_positive if n_positive else None,
        "calibration": (
            float(risk[selected].mean()) - float(truth[selected].mean())
            if n_selected
            else None
        ),
    }


def _bootstrap_intervals(
    risk: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    approved: np.ndarray,
    seed: int,
    resamples: int,
    level: float,
) -> dict[tuple[str, str], tuple[float, float, int] | None]:
    """Paired percentile bootstrap of every (view, metric) gap over transactions."""
    n = int(risk.size)
    rng = np.random.default_rng(seed + 7919)
    draws: dict[tuple[str, str], list[float]] = {
        (view, metric): [] for view in EVALUATION_VIEWS for metric in METRICS
    }
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        latent = _metric_values(risk[idx], y[idx], observed[idx], approved[idx], "latent_truth")
        for view in EVALUATION_VIEWS:
            values = (
                latent
                if view == "latent_truth"
                else _metric_values(risk[idx], y[idx], observed[idx], approved[idx], view)
            )
            for metric in METRICS:
                if latent[metric] is None or values[metric] is None:
                    continue
                draws[(view, metric)].append(float(values[metric]) - float(latent[metric]))

    low_pct = 100.0 * (1.0 - level) / 2.0
    high_pct = 100.0 - low_pct
    intervals: dict[tuple[str, str], tuple[float, float, int] | None] = {}
    for key, values in draws.items():
        if len(values) >= resamples // 2:
            low, high = np.percentile(np.asarray(values), [low_pct, high_pct])
            intervals[key] = (float(low), float(high), len(values))
        else:
            intervals[key] = None
    return intervals


def _audit_rows(data: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    transactions = data["transactions"]
    approval_boundary = float(data["approval_boundary"])
    risk = np.array([float(t["risk_score"]) for t in transactions])
    y = np.array([int(t["Y"]) for t in transactions], dtype=np.int64)
    observed = np.array([int(t["label_observed"]) for t in transactions], dtype=np.int64)
    delay = np.array([float(t["label_delay_days"]) for t in transactions])
    age = np.array([float(t["age_days"]) for t in transactions])
    decile = np.array([int(t["risk_decile"]) for t in transactions], dtype=np.int64)
    approved = risk < approval_boundary

    rows: list[dict[str, Any]] = []

    for audit_age in PANEL_AGES:
        arrived = delay <= audit_age
        eligible = age >= audit_age
        for bucket_decile in PANEL_DECILES:
            bucket = eligible & (decile == bucket_decile)
            if not bool(bucket.any()):
                continue
            rows.append(
                {
                    "panel": "pre",
                    "transaction_age_days": int(audit_age),
                    "risk_decile": int(bucket_decile),
                    "maturation_rate": round(float(arrived[bucket].mean()), 6),
                    "approval_boundary": approval_boundary,
                    "evaluation_view": "",
                    "metric": "",
                    "latent_value": "",
                    "operational_value": "",
                    "gap": "",
                    "interval_low": "",
                    "interval_high": "",
                }
            )

    full = {
        view: _metric_values(risk, y, observed, approved, view) for view in EVALUATION_VIEWS
    }
    latent = full["latent_truth"]
    resamples = int(config.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES))
    level = float(config.get("bootstrap_level", DEFAULT_BOOTSTRAP_LEVEL))
    intervals = _bootstrap_intervals(
        risk, y, observed, approved, seed_of(config), resamples, level
    )

    for view in EVALUATION_VIEWS:
        for metric in METRICS:
            latent_value = float(latent[metric]) if latent[metric] is not None else ""
            operational_value = (
                float(full[view][metric]) if full[view][metric] is not None else ""
            )
            gap = (
                float(operational_value) - float(latent_value)
                if latent_value != "" and operational_value != ""
                else ""
            )
            rows.append(
                {
                    "panel": "mid",
                    "transaction_age_days": "",
                    "risk_decile": "",
                    "maturation_rate": "",
                    "approval_boundary": approval_boundary,
                    "evaluation_view": view,
                    "metric": metric,
                    "latent_value": latent_value,
                    "operational_value": operational_value,
                    "gap": round(gap, 6) if gap != "" else "",
                    "interval_low": "",
                    "interval_high": "",
                }
            )

    for view in EVALUATION_VIEWS:
        for metric in METRICS:
            latent_value = float(latent[metric]) if latent[metric] is not None else ""
            operational_value = (
                float(full[view][metric]) if full[view][metric] is not None else ""
            )
            gap = (
                float(operational_value) - float(latent_value)
                if latent_value != "" and operational_value != ""
                else ""
            )
            interval = intervals[(view, metric)]
            if interval is None:
                interval_low = interval_high = round(gap, 6) if gap != "" else ""
                valid = 0
            else:
                interval_low, interval_high, valid = interval
                interval_low = round(float(interval_low), 6)
                interval_high = round(float(interval_high), 6)
            rows.append(
                {
                    "panel": "post",
                    "transaction_age_days": "",
                    "risk_decile": "",
                    "maturation_rate": "",
                    "approval_boundary": approval_boundary,
                    "evaluation_view": view,
                    "metric": metric,
                    "latent_value": latent_value,
                    "operational_value": operational_value,
                    "gap": round(gap, 6) if gap != "" else "",
                    "interval_low": interval_low,
                    "interval_high": interval_high,
                    "bootstrap_resamples": int(valid),
                    "bootstrap_level": level,
                }
            )

    return rows


def _approval_boundary_position(data: Mapping[str, Any]) -> float:
    """Risk-decile coordinate of the approval boundary for the pre-panel overlay."""
    approval_boundary = float(data["approval_boundary"])
    above = [
        int(t["risk_decile"])
        for t in data["transactions"]
        if float(t["risk_score"]) >= approval_boundary
    ]
    return float(min(above)) - 0.5 if above else 10.5


def _render(rows: Sequence[Mapping[str, Any]], data: Mapping[str, Any]) -> plt.Figure:
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold", "figure.dpi": 150})
    approval_boundary = float(data["approval_boundary"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    pre = [r for r in rows if r["panel"] == "pre"]
    scatter = axes[0].scatter(
        [int(r["transaction_age_days"]) for r in pre],
        [int(r["risk_decile"]) for r in pre],
        c=[float(r["maturation_rate"]) for r in pre],
        cmap="viridis",
        s=130,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].axhline(
        _approval_boundary_position(data),
        color="#333333",
        linestyle="--",
        linewidth=1,
        label=f"approval boundary (risk_score >= {approval_boundary:.2f})",
    )
    axes[0].set(
        title="Pre: label maturation by age and risk decile",
        xlabel="transaction age at audit (days)",
        ylabel="risk decile",
        xticks=sorted({int(r["transaction_age_days"]) for r in pre}),
        yticks=list(PANEL_DECILES),
        ylim=(0.4, 10.6),
    )
    axes[0].legend(fontsize=7, loc="lower right")
    fig.colorbar(scatter, ax=axes[0], label="maturation rate")

    mid = [r for r in rows if r["panel"] == "mid"]
    axes[1].plot([0.0, 1.0], [0.0, 1.0], color="#BBBBBB", linestyle=":", linewidth=1, label="parity")
    for view in EVALUATION_VIEWS:
        subset = [r for r in mid if r["evaluation_view"] == view]
        axes[1].plot(
            [float(r["latent_value"]) for r in subset],
            [float(r["operational_value"]) for r in subset],
            marker="o",
            color=VIEW_COLORS[view],
            label=VIEW_LABELS[view],
        )
        for row in subset:
            axes[1].annotate(
                str(row["metric"])[:4],
                (float(row["latent_value"]), float(row["operational_value"])),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    axes[1].set(
        title="Mid: operational vs latent-truth metrics",
        xlabel="latent-truth metric value",
        ylabel="operational metric value",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    axes[1].legend(fontsize=6.5, loc="lower right")

    post = [r for r in rows if r["panel"] == "post"]
    axes[2].axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    for i, row in enumerate(post):
        gap = float(row["gap"])
        low = float(row["interval_low"])
        high = float(row["interval_high"])
        color = VIEW_COLORS[str(row["evaluation_view"])]
        axes[2].plot([0.0, gap], [i, i], color=color, linewidth=2, solid_capstyle="round")
        axes[2].errorbar(
            gap,
            i,
            xerr=[[gap - low], [high - gap]],
            fmt="o",
            color="#333333",
            capsize=3,
            markersize=4,
        )
    axes[2].set(
        title="Post: operational minus latent metric",
        xlabel="metric gap (operational - latent)",
        yticks=range(len(post)),
        yticklabels=[
            f"{r['metric']} . {VIEW_LABELS[str(r['evaluation_view'])]}" for r in post
        ],
    )
    axes[2].tick_params(axis="y", labelsize=7)

    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    fig.text(0.5, 0.015, HOW_TO_READ, ha="center", va="bottom", fontsize=7.5, linespacing=1.5)
    return fig


def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Run the real selection-delay audit and write its PNG + sidecar."""
    data = generate_synthetic_data(config)
    rows = _audit_rows(data, config)
    sidecar_path = write_sidecar(NAME, rows, config, out_dir=out_dir)
    fig = _render(rows, data)
    figure_path = save_figure(fig, NAME, out_dir=out_dir)
    plt.close(fig)
    print(HOW_TO_READ)
    return {
        "figure": str(figure_path),
        "sidecar": str(sidecar_path),
        "rows": int(len(rows)),
        "name": NAME,
    }
