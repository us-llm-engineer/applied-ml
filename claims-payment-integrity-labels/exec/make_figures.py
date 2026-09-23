#!/usr/bin/env python3
"""Build the real figures under ``exec/figures/``.

Run from the project root:

    MPLBACKEND=Agg python3 exec/make_figures.py

Every sidecar CSV carries exec provenance (stream config, figure id, harness
hash pointer) and every PNG is rendered from the study data, never from the
mock fixtures under ``viz/mock/``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exec.claims_integrity.figures import build_all  # noqa: E402
from exec.claims_integrity.ledger import record_run, write_ledger  # noqa: E402


def main() -> int:
    out_dir = ROOT / "exec" / "figures"
    started = time.time()
    outputs = build_all(out_dir)
    elapsed = time.time() - started
    config = {
        "scenario": "synthetic-claims-v1",
        "figures": sorted(outputs),
        "seeds": [11, 23],
        "identification_seeds": [11, 23, 37, 53],
    }
    ledger = [
        record_run(seed=seed, config=config, capacity=capacity, runtime_seconds=elapsed)
        for seed, capacity in ((11, 0), (11, 100), (11, 300), (23, 0), (23, 100), (23, 300))
    ]
    write_ledger(ROOT / "exec" / "ledger.jsonl", ledger)
    for name, (csv_path, png_path) in sorted(outputs.items()):
        print(f"{name}: {csv_path.relative_to(ROOT)} ({csv_path.stat().st_size} B), {png_path.relative_to(ROOT)} ({png_path.stat().st_size} B)")
    print(f"elapsed_seconds: {elapsed:.1f}")
    print(json.dumps({"config_hash": ledger[0]["config_hash"]}, indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
