#!/usr/bin/env python3
"""Build the figures under ``figures/``.

Run from the project root:

    MPLBACKEND=Agg python3 make_figures.py

Every CSV carries its provenance (stream config and figure id) and every PNG
is rendered from the CSV's study data.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from claims_integrity.figures import build_all  # noqa: E402
from claims_integrity.ledger import config_hash  # noqa: E402


def main() -> int:
    out_dir = ROOT / "figures"
    started = time.time()
    outputs = build_all(out_dir)
    elapsed = time.time() - started
    config = {
        "scenario": "synthetic-claims-v1",
        "figures": sorted(outputs),
        "seeds": [11, 23],
        "identification_seeds": [11, 23, 37, 53],
    }
    for name, (csv_path, png_path) in sorted(outputs.items()):
        print(f"{name}: {csv_path.relative_to(ROOT)} ({csv_path.stat().st_size} B), {png_path.relative_to(ROOT)} ({png_path.stat().st_size} B)")
    print(f"elapsed_seconds: {elapsed:.1f}")
    print(json.dumps({"config_hash": config_hash(config)}, indent=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
