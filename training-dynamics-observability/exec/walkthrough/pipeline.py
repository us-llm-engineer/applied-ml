"""The interface NB3 reuses: published results, the cached dataset, shared runs.

Nothing here regenerates anything.  ``load_dataset`` reads the rows NB2 already
published, so NB3 provably consumes NB2's artifact instead of re-deriving it,
and ``shared_run_ids`` returns the ledger run ids that carry NB2's generation
config hash and task data hash.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exec.common import load_json, load_ledger  # noqa: E402  (frozen shared helpers)
from exec.walkthrough import task  # noqa: E402

RESULTS_PATH = PROJECT_ROOT / "exec" / "walkthrough_results.json"
GENERATION_RUN_ID = "nb2-task-generation"
GENERATION_REPEAT_RUN_ID = "nb2-task-generation-repeat"
CACHED_RERUN_ID = "nb2-task-generation-cached"

# NB3 must not regenerate the task data; load_dataset returns NB2's published rows.
DATA_REGENERATED_BY_NB3 = False


def load_published() -> dict:
    """The merged walkthrough artifact (NB2 keys plus whatever NB3 added)."""
    published = load_json(RESULTS_PATH, None)
    if not isinstance(published, dict):
        raise FileNotFoundError(
            f"{RESULTS_PATH} is missing; run notebooks/02_project_walkthrough_part1.ipynb first"
        )
    return published


def load_dataset() -> list:
    """NB2's published dataset rows, read from cache and never regenerated."""
    truth = load_published().get("task_truth") or {}
    rows = truth.get("dataset_rows")
    if not rows:
        raise FileNotFoundError("walkthrough_results.json has no task_truth.dataset_rows to reuse")
    return rows


def generation_config() -> dict:
    """The generation config whose hash matches NB2's ledger rows and preprocessing."""
    return task.generation_config()


def generation_config_hash() -> str:
    return task.generation_config_hash()


def dataset_data_hash() -> str:
    """Hash of the cached rows, recomputed the same way test_30 recomputes it."""
    return task.dataset_data_hash(load_dataset())


def published_data_hash() -> str:
    published = load_published()
    return (published.get("preprocessing") or {}).get("data_hash")


def published_config_hash() -> str:
    published = load_published()
    return (published.get("preprocessing") or {}).get("config_hash")


def shared_run_ids() -> list:
    """Ledger run ids for the task-generation config that NB3 consumes."""
    config_hash = generation_config_hash()
    data_hash = dataset_data_hash()
    ids = []
    for row in load_ledger():
        if row.get("notebook") != "02":
            continue
        if row.get("config_hash") != config_hash or row.get("data_hash") != data_hash:
            continue
        if row.get("cache_status") == "cached_rerun":
            continue
        ids.append(str(row["run_id"]))
    return sorted(set(ids))


def verify_cache() -> dict:
    """Confirm the published rows hash to the published preprocessing data hash."""
    rows = load_dataset()
    recomputed = task.dataset_data_hash(rows)
    published = published_data_hash()
    return {
        "rows": len(rows),
        "recomputed_data_hash": recomputed,
        "published_data_hash": published,
        "matches": recomputed == published,
        "data_regenerated_by_nb3": DATA_REGENERATED_BY_NB3,
        "config_hash": generation_config_hash(),
        "shared_run_ids": shared_run_ids(),
    }
