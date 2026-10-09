"""the formal-setup table and the defaults matrix.

Both tables are static data transcribed from the papers.
a run; the "computation" this module offers is the assembly and validation of a
verbatim-quoted comparison table, not a numeric estimate.
"""
from __future__ import annotations

from training_dynamics.common import sha256_json

PAPERS = ("P1", "P2", "P3")
PAPER_TITLES = {
    "P1": "Edge of Stochastic Stability: Revisiting the Edge of Stability for SGD (arXiv:2412.20553)",
    "P2": "Momentum in large-batch training: Polyak enlarges the critical batch size, Nesterov improves data efficiency (arXiv:2609.02728)",
    "P3": "Measuring training variability from stochastic optimization using robust nonparametric testing (arXiv:2406.08307)",
}

# side-by-side formal setup, quoted verbatim (source, section/page) per dimension.
FORMAL_SETUP_ROWS = (
    {
        "dimension": "loss",
        "P1": r"Mini-batch loss L_B(theta) on batch B ~ P_b; local quadratic model L_i(theta) = 1/2 (theta - x_i)^T H_i (theta - x_i) [Section 4.1]",
        "P2": r"Power-Law Kernel regression excess risk E(theta) = 1/2 <theta - theta*, H(theta - theta*)>, population risk R(theta) = E(theta) + sigma^2/2 [Section 3.1]",
        "P3": r"Non-convex ERM over logit-gap functions M = {m(x;theta)}; evaluated via induced CDFs of logit gaps [Section II-A]",
    },
    {
        "dimension": "curvature",
        "P1": "Directional mini-batch curvature (Batch Sharpness, Eq. 3), Rayleigh quotient S_B(theta), full-batch top eigenvalue lambda_max [Definition 3]",
        "P2": r"Population covariance H = diag(lambda_1, lambda_2, ...), power-law spectrum lambda_j ~ j^(-beta), beta > 1 [Assumption 3.3]",
        "P3": "FLAGGED UNDEFINED: no Hessian, curvature or spectrum object is defined [Section II]",
    },
    {
        "dimension": "noise",
        "P1": "Single-sample gradient noise covariance Sigma_g; distinguishes noise-driven (Type-1) from curvature-driven (Type-2) oscillations [Section 5.1]",
        "P2": r"Constant-order label noise epsilon ~ N(0, sigma^2), sigma ~ 1 [Assumption 3.2]",
        "P3": r"Epistemic model variability as an L1 contamination ball B_1(P_0, alpha) and an alpha-trimming set R_alpha(P_1) [Section III-A]",
    },
    {
        "dimension": "sampling",
        "P1": "Mini-batch B of size b sampled i.i.d. from P_b (with replacement); also analyzes Random Reshuffling [Section 4.1, Section 8]",
        "P2": "One-pass/online: total data budget D, batch size B = D^b, T = D/B steps [Section 1, Section 3.1]",
        "P3": "Test set D_test drawn i.i.d. from pi; nonparametric bootstrap over test sets to estimate alpha_hat [Section IV-A]",
    },
    {
        "dimension": "optimizer",
        "P1": r"Mini-batch SGD, constant learning rate eta: theta_{t+1} = theta_t - eta * grad L_B(theta_t) [Theorem 1 context]",
        "P2": "Constant step size eta and momentum factor rho in [0,1) for SGD, Polyak momentum, Nesterov momentum [Section 3.2]",
        "P3": "Generic stochastic optimization algorithms for DNNs; specific update equations FLAGGED UNDEFINED [Section I]",
    },
)

# per-paper reproducibility defaults; every "specified" cell is a fragment
# taken from the corresponding paper.
DEFAULTS_MATRIX = (
    {
        "paper": "P3", "item": "random seeds",
        "specified": True,
        "value": "reference pool M=800 (CNN), 200 per ablation arm, M_total=90 (ViT); exact integer seeds absent",
        "source": "Section V, Page 7",
    },
    {
        "paper": "P3", "item": "batch construction",
        "specified": True, "value": "batch size b=32, Adam, eta=0.001, 50 epochs (small CNN)",
        "source": "Section V, Page 7",
    },
    {
        "paper": "P3", "item": "warmup",
        "specified": False, "value": "absent / unstated (no warmup used)",
        "source": "Section V",
    },
    {
        "paper": "P1", "item": "batch construction",
        "specified": True, "value": "batch sizes b in {2,4,8,16,32,64,128,256}",
        "source": "Section 7, Figures 4, 8, 24, 25",
    },
    {
        "paper": "P1", "item": "eigensolver / Hessian method",
        "specified": True,
        "value": "power iteration or LOBPCG via double-backprop Hessian-vector products; step sharpness every 8 steps, Batch Sharpness every 128 steps, lambda_max every 256 steps",
        "source": "Appendix L, Page 72",
    },
    {
        "paper": "P1", "item": "warmup",
        "specified": False, "value": "absent / unstated (no warmup used)",
        "source": "entire paper",
    },
    {
        "paper": "P2", "item": "initialization",
        "specified": True, "value": "deterministic zero initialization theta_{-1} = theta_0 = 0",
        "source": "Section 7, Page 11",
    },
    {
        "paper": "P2", "item": "random seeds",
        "specified": True,
        "value": "5 seeds (stability scans), 100 runs (risk-scaling curves), 50 runs per (eta, rho) (phase diagrams)",
        "source": "Section 7, Pages 11-13; Appendix G",
    },
    {
        "paper": "P2", "item": "warmup",
        "specified": False, "value": "absent / unstated (no warmup used)",
        "source": "entire paper",
    },
)

CONFIG = {"stage": "foundations-nb1", "family": "defaults_audit", "source": "paper-reported tables"}


def formal_setup_table() -> dict:
    rows = [dict(row) for row in FORMAL_SETUP_ROWS]
    flagged = [
        {"paper": paper, "dimension": row["dimension"]}
        for row in rows
        for paper in PAPERS
        if "FLAGGED UNDEFINED" in row[paper]
    ]
    cells = [
        {
            "dimension": row["dimension"],
            "paper": paper,
            "defined": "FLAGGED UNDEFINED" not in row[paper],
            "text": row[paper],
        }
        for row in rows
        for paper in PAPERS
    ]
    return {
        "papers": {code: PAPER_TITLES[code] for code in PAPERS},
        "rows": rows,
        "cells": cells,
        "dimensions": [row["dimension"] for row in rows],
        "flagged_undefined": flagged,
        "config_hash": sha256_json(rows),
    }


def audit_matrix() -> dict:
    rows = [dict(row) for row in DEFAULTS_MATRIX]
    specified_count = sum(1 for row in rows if row["specified"])
    unspecified_count = len(rows) - specified_count
    return {
        "rows": rows,
        "specified_count": specified_count,
        "unspecified_count": unspecified_count,
        "papers_covered": sorted({row["paper"] for row in rows}),
        "config_hash": sha256_json(rows),
    }


def run() -> dict:
    return {"formal_setup": formal_setup_table(), "defaults_audit": audit_matrix()}
