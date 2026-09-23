#!/usr/bin/env python3
"""Render the frozen figures from the canonical execution runs.

Round-1 figures (3) are rendered from the Round-1 canonical config; Round-2
figures (7) from `exec.config.canonical_r2_config()`. Writes
``exec/results/run_manifest.json`` (config_name / run_id shared by the R2
sidecars and the notebooks).

Run from the project root: ``python3 -m exec.render_all_figures``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from exec.config import (
    canonical_config,
    canonical_r2_config,
    config_hash,
    r2_run_identity,
    seed_of,
)
from exec.vizlib import ROOT, run_id

R1_MODULES = {
    "NB1_estimator_calibration_triptych": "nb1",
    "NB2_selection_delay_audit": "nb2",
    "NB3_monitoring_validity_panel": "nb3",
}

R2_MODULES = {
    "R2_NB1_sar_truth_and_bound": "r2_sar_truth_and_bound",
    "R2_NB1_wctm_validity": "r2_wctm_validity",
    "R2_NB1_alg1_prior_corruption": "r2_alg1_prior_corruption",
    "R2_NB2_stream_anatomy": "r2_stream_anatomy",
    "R2_NB2_naive_vs_aware": "r2_naive_vs_aware",
    "R2_NB3_stratum_and_budget": "r2_stratum_and_budget",
    "R2_NB3_monitoring_and_ledger": "r2_monitoring_and_ledger",
}

R3_MODULES = {
    "R3_naive_vs_aware_restricted": "r3_naive_vs_aware_restricted",
    "R3_monitor_comparison": "r3_monitor_comparison",
}


def _render(module_name: str, config: dict) -> dict:
    module = importlib.import_module(f"exec.figures.{module_name}")
    return module.build(config)


def _entry(name: str, result: dict) -> dict:
    return {
        "name": name,
        "figure": str(Path(result["figure"]).relative_to(ROOT)),
        "sidecar": str(Path(result["sidecar"]).relative_to(ROOT)),
        "rows": int(result["rows"]),
    }


def main() -> int:
    r1_config = canonical_config()
    r2_config = canonical_r2_config()
    identity = r2_run_identity(r2_config)
    manifest: dict[str, object] = {
        "config_name": identity["config_name"],
        "run_id": identity["run_id"],
        "seed": identity["seed"],
        "config_hash": identity["config_hash"],
        "config": r2_config,
        "round1_run_id": run_id(r1_config),
        "round1_seed": seed_of(r1_config),
        "round1_config_hash": config_hash(r1_config),
        "figures": [],
        "missing": [],
    }
    for name, module_name in R1_MODULES.items():
        result = _render(module_name, r1_config)
        manifest["figures"].append(_entry(name, result))
        print(f"rendered {name}: {result['rows']} rows -> {result['figure']}")
    for name, module_name in R2_MODULES.items():
        try:
            result = _render(module_name, r2_config)
        except ModuleNotFoundError:
            manifest["missing"].append(name)
            print(f"SKIP {name}: exec/figures/{module_name}.py not present yet")
            continue
        manifest["figures"].append(_entry(name, result))
        print(f"rendered {name}: {result['rows']} rows -> {result['figure']}")
    for name, module_name in R3_MODULES.items():
        try:
            result = _render(module_name, r2_config)
        except ModuleNotFoundError:
            manifest["missing"].append(name)
            print(f"SKIP {name}: exec/figures/{module_name}.py not present yet")
            continue
        manifest["figures"].append(_entry(name, result))
        print(f"rendered {name}: {result['rows']} rows -> {result['figure']}")
    results = ROOT / "exec" / "results"
    results.mkdir(parents=True, exist_ok=True)
    manifest_path = results / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"manifest: {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
