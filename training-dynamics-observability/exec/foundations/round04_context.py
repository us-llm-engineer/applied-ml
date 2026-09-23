"""Small, trace-labelled context tables used by the R4 live NB1 figures."""
from __future__ import annotations


def formal_setup_table():
    return {"trace": "q1", "rows": [("loss", "squared local toy"), ("curvature", "Rayleigh quotient"), ("noise", "seeded minibatches")], "scope": "Definition 1 / Theorem 1 context"}


def paper_context_comparison():
    return {"trace": "q3", "rows": [("CIFAR", 800), ("SVHN", 100), ("toy", 8)], "scope": "paper settings contrasted with this toy"}


def batch_scaling_transitions():
    return {"trace": "q7", "b1": 0.40, "b2": 0.58, "b3": 0.67, "scope": "Theorem 6.3 order-level transitions"}


def round04_ks_vs_alpha_sample_size():
    return {"trace": "q10", "rows": [("this paired audit", 8), ("alpha-trim paper ensemble", 30), ("classical KS comparison", 100)]}


def audit_matrix():
    return {"trace": "q11", "rows": [("specified", 3), ("not specified", 5), ("defaults", 4)], "scope": "cross-paper reproducibility defaults"}
