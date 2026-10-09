#!/usr/bin/env python3
"""Render the figures from the canonical runs.

The three base figures are rendered from the canonical config; the study
figures from `label_delay.config.canonical_study_config()`. Writes
``label_delay/results/run_manifest.json`` (config_name / run_id shared by the
figure CSVs and the notebooks).

Run from the project root: ``python3 -m label_delay.render_all_figures``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from label_delay.config import (
    canonical_config,
    canonical_study_config,
    config_hash,
    run_identity,
    seed_of,
)
from label_delay.vizlib import ROOT, run_id

BASE_MODULES = {
    "NB1_estimator_calibration_triptych": "nb1",
    "NB2_selection_delay_audit": "nb2",
    "NB3_monitoring_validity_panel": "nb3",
}

STUDY_MODULES = {
    "sar_truth_and_bound": "sar_truth_and_bound",
    "wctm_validity": "wctm_validity",
    "alg1_prior_corruption": "alg1_prior_corruption",
    "stream_anatomy": "stream_anatomy",
    "naive_vs_aware": "naive_vs_aware",
    "stratum_and_budget": "stratum_and_budget",
    "monitoring_and_ledger": "monitoring_and_ledger",
}

COMPARISON_MODULES = {
    "naive_vs_aware_restricted": "naive_vs_aware_restricted",
    "monitor_comparison": "monitor_comparison",
}


def _render(module_name: str, config: dict) -> dict:
    module = importlib.import_module(f"label_delay.figures.{module_name}")
    return module.build(config)


def _entry(name: str, result: dict) -> dict:
    return {
        "name": name,
        "figure": str(Path(result["figure"]).relative_to(ROOT)),
        "sidecar": str(Path(result["sidecar"]).relative_to(ROOT)),
        "rows": int(result["rows"]),
    }


def main() -> int:
    base_config = canonical_config()
    study_config = canonical_study_config()
    identity = run_identity(study_config)
    manifest: dict[str, object] = {
        "config_name": identity["config_name"],
        "run_id": identity["run_id"],
        "seed": identity["seed"],
        "config_hash": identity["config_hash"],
        "config": study_config,
        "base_run_id": run_id(base_config),
        "base_seed": seed_of(base_config),
        "base_config_hash": config_hash(base_config),
        "figures": [],
        "missing": [],
    }
    for name, module_name in BASE_MODULES.items():
        result = _render(module_name, base_config)
        manifest["figures"].append(_entry(name, result))
        print(f"rendered {name}: {result['rows']} rows -> {result['figure']}")
    for name, module_name in STUDY_MODULES.items():
        try:
            result = _render(module_name, study_config)
        except ModuleNotFoundError:
            manifest["missing"].append(name)
            print(f"SKIP {name}: label_delay/figures/{module_name}.py not present yet")
            continue
        manifest["figures"].append(_entry(name, result))
        print(f"rendered {name}: {result['rows']} rows -> {result['figure']}")
    for name, module_name in COMPARISON_MODULES.items():
        try:
            result = _render(module_name, study_config)
        except ModuleNotFoundError:
            manifest["missing"].append(name)
            print(f"SKIP {name}: label_delay/figures/{module_name}.py not present yet")
            continue
        manifest["figures"].append(_entry(name, result))
        print(f"rendered {name}: {result['rows']} rows -> {result['figure']}")
    results = ROOT / "label_delay" / "results"
    results.mkdir(parents=True, exist_ok=True)
    manifest_path = results / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"manifest: {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
