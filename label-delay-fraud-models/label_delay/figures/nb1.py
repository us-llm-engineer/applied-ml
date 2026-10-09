"""NB1 estimator-calibration triptych figure (pre / mid / post training).

Panels
------
pre   observed-positive rate by propensity floor ``e_min``, coloured by latent
      risk stratum: how often the positive label is actually seen in each
      stratum as the propensity floor falls (rows from
      ``run_recovery_experiment(...)["observed_positive_by_stratum"]``).
mid   per-replicate ``SAR - latent 0-1 risk`` for each estimator variant
      (``oracle``, ``estimated``, ``misspecified``) at the oracle ``e_min``,
      against the dashed zero reference line (rows from ``replicates``).
post   ``absolute_bias`` (points) and the 95% interval width (dotted lines) per
      ``e_min`` and variant (rows from ``post_summary``).

All rows come from the experiment in :mod:`label_delay.recovery`.

How to read
-----------
pre shows the observation channel itself: the high-risk stratum is confirmed
far more often than the low-risk one, and both rise with the propensity floor.
Mid shows the oracle wobbling around the zero line (its Monte-Carlo error
interval is declared in the experiment report) while the derived-here
``estimated`` and ``misspecified`` variants carry visible bias; post turns that
into ``|bias|`` and interval width per ``e_min``, where a lower floor must
widen the interval.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from label_delay.recovery import run_recovery_experiment
from label_delay.vizlib import save_figure, write_sidecar

NAME = "NB1_estimator_calibration_triptych"
VARIANTS: tuple[str, ...] = ("oracle", "estimated", "misspecified")
STRATA: tuple[str, ...] = ("low", "high")

#: Colour map for the figure.
PALETTE: dict[str, str] = {
    "oracle": "#0072B2",
    "estimated": "#E69F00",
    "misspecified": "#CC79A7",
    "low": "#0072B2",
    "high": "#D55E00",
}

HOW_TO_READ = (
    "How to read: pre shows how often a positive is actually observed in each latent-risk "
    "stratum as the propensity floor e_min moves -- the high-risk stratum is confirmed far "
    "more often than the low-risk one.\n"
    "Mid plots each estimator's SAR-minus-latent risk per replicate against the dashed zero "
    "line (the oracle fluctuates around it, the two derived-here variants do not); post turns "
    "that into |bias| (points) and 95% interval width (dotted) per e_min, where a lower e_min "
    "must widen the interval."
)


def _rows_from_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for row in report["observed_positive_by_stratum"]:
        rows.append(
            {
                "panel": "pre",
                "e_min": float(row["e_min"]),
                "latent_risk_stratum": str(row["latent_risk_stratum"]),
                "observed_positive_rate": float(row["observed_positive_rate"]),
                "estimator_variant": "",
                "replicate": "",
                "sar_minus_truth_risk": "",
                "absolute_bias": "",
                "interval_width": "",
                "stratum_n": int(row["n"]),
            }
        )

    # The mid panel uses the oracle's e_min (the largest configured floor) so
    # the oracle benchmark shown there is the one the oracle report declares.
    oracle_e_min = max(float(value) for value in report["e_min_values"])
    for row in report["replicates"]:
        if float(row["e_min"]) != oracle_e_min:
            continue
        rows.append(
            {
                "panel": "mid",
                "e_min": float(row["e_min"]),
                "latent_risk_stratum": "",
                "observed_positive_rate": "",
                "estimator_variant": str(row["estimator_variant"]),
                "replicate": int(row["replicate"]),
                "sar_minus_truth_risk": float(row["sar_minus_truth_risk"]),
                "absolute_bias": "",
                "interval_width": float(row["interval_width"]),
            }
        )

    for row in report["post_summary"]:
        rows.append(
            {
                "panel": "post",
                "e_min": float(row["e_min"]),
                "latent_risk_stratum": "",
                "observed_positive_rate": "",
                "estimator_variant": str(row["estimator_variant"]),
                "replicate": "",
                "sar_minus_truth_risk": "",
                "absolute_bias": float(row["absolute_bias"]),
                "interval_width": float(row["interval_width"]),
            }
        )

    return rows


def _render(rows: Sequence[Mapping[str, Any]]) -> plt.Figure:
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold", "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    pre = [row for row in rows if row["panel"] == "pre"]
    for stratum in STRATA:
        subset = [row for row in pre if row["latent_risk_stratum"] == stratum]
        axes[0].scatter(
            [float(row["e_min"]) for row in subset],
            [float(row["observed_positive_rate"]) for row in subset],
            s=65,
            color=PALETTE[stratum],
            label=f"latent risk: {stratum}",
            zorder=3,
        )
    axes[0].set(
        title="Pre: observation by propensity floor",
        xlabel="minimum propensity e_min",
        ylabel="observed-positive rate",
        xticks=sorted({float(row["e_min"]) for row in pre}),
    )
    axes[0].legend(fontsize=8)

    mid = [row for row in rows if row["panel"] == "mid"]
    axes[1].axhline(0.0, color="#333333", linewidth=1, linestyle="--", label="zero error")
    for variant in VARIANTS:
        subset = [row for row in mid if row["estimator_variant"] == variant]
        axes[1].scatter(
            [int(row["replicate"]) for row in subset],
            [float(row["sar_minus_truth_risk"]) for row in subset],
            s=9,
            alpha=0.55,
            color=PALETTE[variant],
            label=variant,
            linewidths=0,
            zorder=3,
        )
    axes[1].set(
        title="Mid: SAR error by estimator",
        xlabel="replicate",
        ylabel="SAR minus latent 0-1 risk",
    )
    axes[1].legend(fontsize=7, markerscale=2.2, loc="upper right")

    post = [row for row in rows if row["panel"] == "post"]
    for variant in VARIANTS:
        subset = sorted(
            (row for row in post if row["estimator_variant"] == variant),
            key=lambda row: float(row["e_min"]),
        )
        axes[2].scatter(
            [float(row["e_min"]) for row in subset],
            [float(row["absolute_bias"]) for row in subset],
            s=65,
            color=PALETTE[variant],
            label=f"{variant}: bias",
            zorder=3,
        )
        axes[2].plot(
            [float(row["e_min"]) for row in subset],
            [float(row["interval_width"]) for row in subset],
            color=PALETTE[variant],
            alpha=0.65,
            linestyle=":",
            marker="o",
            markersize=4,
            label=f"{variant}: width",
        )
    axes[2].set(
        title="Post: bias (points), interval width (dotted)",
        xlabel="minimum propensity e_min",
        ylabel="absolute value",
        xticks=sorted({float(row["e_min"]) for row in post}),
    )
    axes[2].legend(fontsize=6, ncol=2)

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.text(0.5, 0.015, HOW_TO_READ, ha="center", va="bottom", fontsize=7.5, linespacing=1.5)
    return fig


def build(config: Mapping[str, Any], out_dir: str | None = None) -> dict[str, Any]:
    """Run the real recovery experiment and write the NB1 PNG + sidecar."""
    report = run_recovery_experiment(config)
    rows = _rows_from_report(report)
    sidecar_path = write_sidecar(NAME, rows, config, out_dir=out_dir)
    fig = _render(rows)
    figure_path = save_figure(fig, NAME, out_dir=out_dir)
    plt.close(fig)
    print(HOW_TO_READ)
    return {
        "figure": str(figure_path),
        "sidecar": str(sidecar_path),
        "rows": int(len(rows)),
        "name": NAME,
    }
