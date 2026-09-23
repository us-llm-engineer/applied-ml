"""NB1 driver: run the three founded experiment families, publish JSON, log the ledger.

The notebook calls :func:`run_all` once from its setup cell, so every number in
``exec/foundations_results.json`` is produced by a live execution of this module
rather than copied from a previous artifact.
"""
from __future__ import annotations

from pathlib import Path

from exec import common

from . import eoss, momentum, variability

RESULTS_PATH = common.EXEC_DIR / "foundations_results.json"

FAMILIES = (
    ("nb1-eoss", eoss, 20260921, "eoss"),
    ("nb1-momentum", momentum, 20260922, "momentum"),
    ("nb1-variability", variability, 20260923, "variability"),
)


def run_all(publish: bool = True) -> dict:
    """Compute eoss + momentum + variability, publish the JSON, append the ledger."""
    results = {}
    timings = {}
    for run_id, module, seed, key in FAMILIES:
        with common.Timer() as timer:
            results[key] = module.run()
        timings[run_id] = float(timer.elapsed)
        if publish:
            row = common.make_run(
                run_id=run_id,
                notebook="01",
                seed=seed,
                data_hash=common.sha256_json(module.SPEC),
                config_hash=common.sha256_json(module.CONFIG),
                result_obj=results[key],
                wall_time_s=timer.elapsed,
            )
            common.append_runs([row])
    payload = {
        "eoss": results["eoss"],
        "momentum": results["momentum"],
        "variability": results["variability"],
        "meta": {
            "stage": "foundations-nb1",
            "produced_by": "exec.foundations.runner.run_all",
            "wall_time_s": timings,
            "cost_assumption": common.COST_ASSUMPTION,
        },
    }
    if publish:
        common.write_json(RESULTS_PATH, payload)
    return payload


if __name__ == "__main__":
    run_all()
    print(f"wrote {RESULTS_PATH}")
