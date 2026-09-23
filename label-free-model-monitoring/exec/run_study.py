"""Study driver: regenerate every figure artifact and summarise the run.

``python3 -m exec.run_study`` writes the five figure CSVs and PNGs plus
``exec/figures/run-manifest.json``.  The notebooks call :func:`ensure_figures`
so a fresh checkout reproduces exactly the artifacts the main suite checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import figures

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "exec" / "figures"


def load_manifest() -> dict:
    path = FIGDIR / "run-manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def ensure_figures(force: bool = False) -> dict:
    """Generate the figure suite if any artifact is missing (idempotent)."""
    stems = ["fig-a1", "fig-a2", "fig-a3", "fig-a4", "fig-a5", "fig-a6", "fig-a7", "fig-a9", "fig-a8"]
    complete = all((FIGDIR / f"{s}.csv").exists() and (FIGDIR / f"{s}.png").exists() for s in stems)
    if force or not complete or not (FIGDIR / "run-manifest.json").exists():
        figures.render_all()
    return load_manifest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the T5 synthetic study figures")
    parser.add_argument("--run-id", default=None, help="override the generated run id")
    args = parser.parse_args()
    results = figures.render_all(args.run_id)
    for stem, result in results.items():
        print(f"{stem}: {result.rows} rows -> {result.csv_path} + {result.png_path}")
    print(f"manifest: {FIGDIR / 'run-manifest.json'}")


if __name__ == "__main__":
    main()
