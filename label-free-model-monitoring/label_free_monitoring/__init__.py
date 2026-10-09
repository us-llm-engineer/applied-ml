"""Label-free post-deployment monitoring study (synthetic data, CPU only).

Modules
-------
synthetic   : shift-regime generators, champion models, deployment streams.
estimators  : SEES-style gap estimate, sequential tail monitor, D3M disagreement monitor.
controller  : retraining state machine (derived here; the cited papers stop at the alarm).
policy      : policy comparison over paired seeds with latent truth and delayed audits.
figures     : renders the study figures into label_free_monitoring/figures/.
run_study   : end-to-end study driver used by the notebooks.
"""

__all__ = [
    "synthetic",
    "estimators",
    "controller",
    "policy",
    "figures",
    "run_study",
]
