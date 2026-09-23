"""Shared figure-sidecar and rendering helpers for the frozen viz contract.

Every final figure is written as ``exec/figures/<name>.png`` plus
``exec/figures/<name>.csv`` whose rows carry provenance (seed, config hash, run
id) back to the execution run that produced them. Mock assets under ``viz/``
must never be copied or promoted to result data.
"""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from exec.config import config_hash, seed_of

ROOT = Path(__file__).resolve().parents[1]

#: Required sidecar columns, mirroring the frozen specs in ``viz/spec/``.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "NB1_estimator_calibration_triptych": (
        "panel",
        "e_min",
        "latent_risk_stratum",
        "observed_positive_rate",
        "estimator_variant",
        "replicate",
        "sar_minus_truth_risk",
        "absolute_bias",
        "interval_width",
    ),
    "NB2_selection_delay_audit": (
        "panel",
        "transaction_age_days",
        "risk_decile",
        "maturation_rate",
        "approval_boundary",
        "evaluation_view",
        "metric",
        "latent_value",
        "operational_value",
        "gap",
        "interval_low",
        "interval_high",
    ),
    "NB3_monitoring_validity_panel": (
        "panel",
        "update_index",
        "label_status",
        "variant",
        "log_wealth",
        "threshold_c",
        "false_alarm_incidence",
        "interval_low",
        "interval_high",
        "detection_delay",
    ),
}

PROVENANCE_COLUMNS = ("seed", "config_hash", "run_id")


def figure_dir(out_dir: str | Path | None = None) -> Path:
    """Directory for final figures; defaults to ``exec/figures``."""
    path = Path(out_dir) if out_dir is not None else ROOT / "exec" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path

def run_id(config: Mapping[str, Any]) -> str:
    """Deterministic identifier of the run that produced a figure."""
    return (
        f"r1-{config_hash(config)[:10]}-seed{seed_of(config)}"
        f"-n{int(config.get('n_transactions', 0))}"
        f"-rep{int(config.get('n_replicates', 0))}"
    )


def provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Provenance fields stamped onto every sidecar row."""
    return {
        "seed": seed_of(config),
        "config_hash": config_hash(config),
        "run_id": run_id(config),
    }


def write_sidecar(
    name: str,
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    out_dir: str | Path | None = None,
    required: Sequence[str] | None = None,
) -> Path:
    """Write ``exec/figures/<name>.csv`` with required columns plus provenance.

    Missing required values become empty strings; extra columns are preserved.
    """
    required_cols = tuple(required if required is not None else REQUIRED_COLUMNS[name])
    materialised = [dict(row) for row in rows]
    if not materialised:
        raise ValueError(f"{name}: refusing to write an empty sidecar")
    seen: list[str] = []
    for row in materialised:
        for key in row:
            if key not in seen:
                seen.append(key)
    extra_cols = [c for c in seen if c not in required_cols and c not in PROVENANCE_COLUMNS]
    fields = [*required_cols, *extra_cols, *PROVENANCE_COLUMNS]
    stamp = provenance(config)
    path = figure_dir(out_dir) / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialised:
            full = {col: row.get(col, "") for col in fields}
            full.update(stamp)
            writer.writerow(full)
    return path


def png_dimensions(path: str | Path) -> tuple[int, int]:
    """Return (width, height) of a PNG without external image libraries."""
    payload = Path(path).read_bytes()
    width, height = struct.unpack(">II", payload[16:24])
    return int(width), int(height)


def save_figure(fig: Any, name: str, out_dir: str | Path | None = None) -> Path:
    """Save a matplotlib figure as the non-trivial PNG the contract requires."""
    path = figure_dir(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    width, height = png_dimensions(path)
    if width < 900 or height < 300:
        raise AssertionError(f"{name}.png is too small: {width}x{height}")
    if path.stat().st_size <= 2_000:
        raise AssertionError(f"{name}.png is too small in bytes: {path.stat().st_size}")
    return path


def mock_path(name: str) -> Path:
    """Path of the frozen mock sidecar (read-only; never a result source)."""
    return ROOT / "viz" / "mock" / f"{name}.csv"


# --------------------------------------------------------------------------- #
# Round-2 long-format sidecars (one schema for all seven R2 figures)


R2_DATA_COLUMNS = (
    "panel",
    "series",
    "group",
    "x",
    "y",
    "y_low",
    "y_high",
    "z",
    "n",
    "label",
)


def write_r2_sidecar(
    name: str,
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    out_dir: str | Path | None = None,
) -> Path:
    """Write an R2 long-format sidecar with the shared provenance stamp."""
    from exec.config import r2_run_identity

    materialised = [dict(row) for row in rows]
    if not materialised:
        raise ValueError(f"{name}: refusing to write an empty sidecar")
    seen: list[str] = []
    for row in materialised:
        for key in row:
            if key not in seen:
                seen.append(key)
    extra_cols = [
        c for c in seen if c not in R2_DATA_COLUMNS and c not in PROVENANCE_COLUMNS
    ]
    fields = [*R2_DATA_COLUMNS, *extra_cols, *PROVENANCE_COLUMNS]
    identity = r2_run_identity(config)
    stamp = {
        "seed": identity["seed"],
        "config_hash": identity["config_hash"],
        "run_id": identity["run_id"],
    }
    path = figure_dir(out_dir) / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialised:
            full = {col: row.get(col, "") for col in fields}
            full.update(stamp)
            writer.writerow(full)
    return path


def save_r2_figure(fig: Any, name: str, out_dir: str | Path | None = None) -> Path:
    """Save an R2 figure: dpi 150, at least 2000 px wide, 2 kB floor."""
    path = figure_dir(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    width, height = png_dimensions(path)
    if width < 2000 or height < 300:
        raise AssertionError(f"{name}.png is too small: {width}x{height}")
    if path.stat().st_size <= 2_000:
        raise AssertionError(f"{name}.png is too small in bytes: {path.stat().st_size}")
    return path
