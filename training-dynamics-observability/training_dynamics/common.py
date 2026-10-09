"""Frozen shared helpers for the training-dynamics observability lab.

Everything written through :func:`write_json` is strict JSON: no NaN, no
Infinity, one trailing newline.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXEC_DIR = PROJECT_ROOT / "training_dynamics"
LEDGER_PATH = EXEC_DIR / "run_ledger.json"

ASSUMED_USD_PER_GPU_HOUR = 1.50
COST_ASSUMPTION = (
    "assumed cost = wall-clock seconds x USD 1.50 per GPU-hour (a stated T4-class "
    "cloud-rate assumption, not a measured invoice)"
)


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, no whitespace, strict values."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj) -> str:
    return sha256_text(canonical_json(obj))


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json(path, obj) -> None:
    """Write strict, human-readable JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, allow_nan=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_ledger() -> list:
    value = load_json(LEDGER_PATH, [])
    rows = value.get("runs", value) if isinstance(value, dict) else value
    return list(rows or [])


def append_runs(new_rows) -> list:
    """Merge rows by ``run_id`` (later write wins per id) and persist sorted."""
    merged = {row["run_id"]: row for row in load_ledger()}
    for row in new_rows:
        merged[row["run_id"]] = row
    rows = [merged[run_id] for run_id in sorted(merged)]
    write_json(LEDGER_PATH, {"runs": rows})
    return rows


def make_run(
    run_id: str,
    notebook: str,
    seed: int,
    data_hash: str,
    config_hash: str,
    result_obj,
    wall_time_s: float,
    *,
    assumed_cost_usd: float | None = None,
    cache_status: str | None = None,
    reuses_run_id: str | None = None,
    historical_wall_time_s: float | None = None,
    historical_assumed_cost_usd: float | None = None,
) -> dict:
    """Build one ledger row; hashes the result payload for reproducibility."""
    wall = max(float(wall_time_s), 0.0)
    if assumed_cost_usd is None:
        cost = round(wall * ASSUMED_USD_PER_GPU_HOUR / 3600.0, 10)
    else:
        cost = max(float(assumed_cost_usd), 0.0)
    row = {
        "run_id": run_id,
        "notebook": notebook,
        "seed": int(seed),
        "data_hash": data_hash,
        "config_hash": config_hash,
        "result_hash": sha256_json(result_obj),
        "wall_time_s": wall,
        "assumed_cost_usd": cost,
        "cost_assumption": COST_ASSUMPTION,
    }
    if cache_status is not None:
        row["cache_status"] = cache_status
    if reuses_run_id is not None:
        row["reuses_run_id"] = reuses_run_id
    if historical_wall_time_s is not None:
        row["historical_wall_time_s"] = float(historical_wall_time_s)
    if historical_assumed_cost_usd is not None:
        row["historical_assumed_cost_usd"] = float(historical_assumed_cost_usd)
    return row


def selfcheck(check_id: int, ok: bool, message: str = "") -> bool:
    """Print the one project-wide self-check format (one verdict per line)."""
    verdict = "MET" if ok else "NOT MET"
    line = f"SELF-CHECK {int(check_id)} [{verdict}]"
    if message:
        line = f"{line} {message}"
    print(line)
    return bool(ok)


class Timer:
    """Wall-clock context manager: ``with Timer() as t: ...; t.elapsed``."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


def find_project_root(start=None) -> Path:
    """Locate the project root from a notebook's working directory."""
    path = Path(start) if start is not None else Path.cwd()
    path = path.resolve()
    while True:
        if (path / "training_dynamics").is_dir() and (path / "notebooks").is_dir():
            return path
        if path.parent == path:
            raise RuntimeError("project root with training_dynamics/ and notebooks/ not found")
        path = path.parent


def wilson_95(successes: int, attempts: int) -> tuple:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.975)
    proportion = successes / attempts
    denominator = 1.0 + z ** 2 / attempts
    center = (proportion + z ** 2 / (2.0 * attempts)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / attempts + z ** 2 / (4.0 * attempts ** 2)
    ) / denominator
    return center - radius, center + radius


def rate_object(successes: int, attempts: int) -> dict:
    """The published binomial-rate shape shared by the coverage and telemetry audits."""
    low, high = wilson_95(successes, attempts)
    return {
        "successes": int(successes),
        "attempts": int(attempts),
        "rate": successes / attempts,
        "wilson_95": {"method": "wilson_95", "low": float(low), "high": float(high)},
    }
