"""NB1 foundations: the three paper objects reproduced by simulation.

Only :mod:`exec.foundations.runner` is used by the notebook; the three
experiment modules (``eoss``, ``momentum``, ``variability``) stay importable
on their own so each family can be inspected or re-run in isolation.
"""
from . import eoss, momentum, variability

__all__ = ["eoss", "momentum", "variability"]
