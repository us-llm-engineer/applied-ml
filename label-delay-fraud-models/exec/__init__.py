"""Round-1 execution package for the label-delay fraud-models flex project.

Synthetic-only research code. Public callables accept a JSON-serialisable
config mapping and return JSON-serialisable mappings/lists (see
``exec/INTERFACES.md``). The frozen behavioural contract lives in ``tests/``.
"""

from __future__ import annotations

__all__ = [
    "config",
    "data",
    "evaluation",
    "monitoring",
    "policies",
    "recovery",
    "vizlib",
]

__version__ = "0.1.0"
