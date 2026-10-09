"""an analytic Batch-Sharpness oracle plus deliberately one-sided catapult evidence.

Paper object (P1): ``BatchSharpness(theta) = E_B[ g_B^T H_B g_B / ||g_B||^2 ]``.
The experiment below reproduces it on a quadratic whose Rayleigh quotient is known
exactly, then records what happens *above* the sufficient catapult condition
``(2 + eps) / eta`` and, separately, what happens *below* it.  The below-threshold
run is reported as INCONCLUSIVE on purpose: P1 gives a sufficient condition, so a
quiet detector is not a safety certificate.
"""
from __future__ import annotations

import numpy as np

from training_dynamics.common import sha256_json

HESSIAN = [[2.0, 0.5], [0.5, 1.0]]
GRADIENT = [1.0, -0.7]

ABOVE_ETA = 0.5
ABOVE_CURVATURE = 5.0
ABOVE_EPSILON = 0.3
ABOVE_ESCAPE_RADIUS = 5.0
ABOVE_STEPS = 15
ABOVE_TRIAL_SEEDS = tuple(range(14))
DECLARED_MIN_FREQUENCY = 0.6

BELOW_ETA = 0.6
BELOW_CURVATURE = 1.0

SPEC = {
    "hessian": HESSIAN,
    "gradient": GRADIENT,
    "above": {
        "eta": ABOVE_ETA,
        "curvature": ABOVE_CURVATURE,
        "epsilon": ABOVE_EPSILON,
        "escape_radius": ABOVE_ESCAPE_RADIUS,
        "steps": ABOVE_STEPS,
        "trial_seeds": list(ABOVE_TRIAL_SEEDS),
    },
    "below": {"eta": BELOW_ETA, "curvature": BELOW_CURVATURE},
}
CONFIG = {
    "stage": "foundations-nb1",
    "family": "eoss",
    "batch_sharpness_estimator": "full-batch Rayleigh quotient g^T H g / (g^T g)",
    "catapult_label_rule": "max(parameter_norms) >= escape_radius > parameter_norms[0]",
    "initial_condition_rule": "theta0 = 0.1 * (1 + 0.001 * standard_normal())",
}


def _batch_sharpness(hessian: np.ndarray, gradient: np.ndarray) -> float:
    """Full-batch Batch-Sharpness: the batch Hessian and gradient ARE H and g.

    With one batch covering the whole dataset and every sample sharing the same
    quadratic, ``H_B`` collapses to ``H`` and ``g_B`` to ``g``, so the estimator
    is exact rather than sampled.  That is the oracle this check is built on.
    """
    num = float(gradient @ hessian @ gradient)
    den = float(gradient @ gradient)
    return num / den


def known_quadratic() -> dict:
    """Analytic Rayleigh quotient and its full-batch estimator on one 2-D quadratic."""
    hessian = np.asarray(HESSIAN, dtype=float)
    gradient = np.asarray(GRADIENT, dtype=float)
    analytic = float(gradient @ hessian @ gradient / (gradient @ gradient))
    estimated = _batch_sharpness(hessian, gradient)
    return {
        "hessian": hessian.tolist(),
        "gradient": gradient.tolist(),
        "analytic_rayleigh": analytic,
        "estimated_batch_sharpness": estimated,
        "estimator": "full-batch Rayleigh quotient (exact by construction)",
    }


def _gradient_descent_norms(seed: int, curvature: float, eta: float, steps: int) -> list:
    rng = np.random.default_rng(1000 + int(seed))
    theta = 0.1 * (1.0 + 0.001 * float(rng.standard_normal()))
    norms = [abs(theta)]
    for _ in range(steps):
        theta = theta - eta * curvature * theta
        norms.append(abs(theta))
    return [float(value) for value in norms]


def catapult_experiment() -> dict:
    """Above-threshold quadratic: GD with eta*curvature = 2.5 > 2 + eps."""
    eta, curvature, epsilon = ABOVE_ETA, ABOVE_CURVATURE, ABOVE_EPSILON
    gradient = np.asarray([curvature * 1.0])
    hessian = np.asarray([[curvature]])
    sharpness = _batch_sharpness(hessian, gradient)
    trials = []
    for seed in ABOVE_TRIAL_SEEDS:
        norms = _gradient_descent_norms(seed, curvature, eta, ABOVE_STEPS)
        escaped = bool(max(norms) >= ABOVE_ESCAPE_RADIUS > norms[0])
        trials.append(
            {
                "seed": int(seed),
                "parameter_norms": norms,
                "escape_radius": ABOVE_ESCAPE_RADIUS,
                "catapult": escaped,
            }
        )
    frequency = float(np.mean([trial["catapult"] for trial in trials]))
    return {
        "eta": float(eta),
        "epsilon": float(epsilon),
        "curvature": float(curvature),
        "batch_sharpness": float(sharpness),
        "sufficient_threshold": float((2.0 + epsilon) / eta),
        "escape_radius": ABOVE_ESCAPE_RADIUS,
        "steps": ABOVE_STEPS,
        "declared_min_frequency": DECLARED_MIN_FREQUENCY,
        "catapult_frequency": frequency,
        "trials": trials,
    }


def below_threshold_experiment() -> dict:
    """Below-threshold quadratic: batch sharpness < 2/eta.  One-sided, so INCONCLUSIVE."""
    eta, curvature = BELOW_ETA, BELOW_CURVATURE
    gradient = np.asarray([curvature * 1.0])
    hessian = np.asarray([[curvature]])
    sharpness = _batch_sharpness(hessian, gradient)
    return {
        "eta": float(eta),
        "curvature": float(curvature),
        "batch_sharpness": float(sharpness),
        "threshold": float(2.0 / eta),
        "status": "INCONCLUSIVE",
        "status_reason": (
            "the catapult criterion is sufficient, never necessary: a quiet below-threshold "
            "run is not evidence of stability, so the outcome abstains"
        ),
    }


def run() -> dict:
    return {
        "known_quadratic": known_quadratic(),
        "above_threshold": catapult_experiment(),
        "below_threshold": below_threshold_experiment(),
    }


# ---------------------------------------------------------------------------
# the paper's own reported empirical settings,
# published next to this toy's settings so the scale gap is explicit rather
# than implied.
# ---------------------------------------------------------------------------

PAPER_CONTEXT = {
    "datasets": ["CIFAR-10-8k (8,192-sample subset)", "full CIFAR-10", "SVHN-8k (8,192-sample subset)"],
    "architectures": ["MLP (2x512, ~2M params)", "5-layer CNN", "ResNet-10", "ResNet-14", "ResNet-20"],
    "batch_sizes": [2, 4, 8, 16, 32, 64, 128, 256],
    "learning_rates_by_arch": {
        "mlp": [0.01, 0.02],
        "cnn": [0.03, 0.05],
        "resnet10_14": [0.005],
        "resnet20": [0.02],
    },
    "eval_frequency_steps": {"step_sharpness": 8, "batch_sharpness": 128, "lambda_max": 256},
    "plateau_2_over_eta": {"mlp_eta_0.01": 200.0, "mlp_eta_0.02": 100.0, "cnn_eta_0.03": 66.67, "resnet10_eta_0.005": 400.0},
    "lambda_max_gap_qualitative": (
        "as batch size b shrinks from 256 toward 2 (MLP, eta=0.01), lambda_max is suppressed from "
        "near 200 down to roughly 10-20 while Batch Sharpness stays pinned at 2/eta=200"
    ),
    "source": "Section 7, Figures 4/8/24/25, Appendix L (arXiv:2412.20553)",
}


def paper_context_comparison() -> dict:
    """This toy's actual settings, published next to the paper's reported settings."""
    this_toy = {
        "datasets": ["one 2-D analytic quadratic (known_quadratic)", "one 1-D quadratic sweep (catapult_experiment)"],
        "batch_sizes": "not applicable: full-batch Rayleigh quotient, no mini-batch sampling",
        "eta": ABOVE_ETA,
        "epsilon": ABOVE_EPSILON,
        "sufficient_threshold_2_over_eta": float((2.0 + ABOVE_EPSILON) / ABOVE_ETA),
        "trial_count": len(ABOVE_TRIAL_SEEDS),
        "steps": ABOVE_STEPS,
    }
    ratio_datasets = f"paper: {sum(PAPER_CONTEXT['batch_sizes'])/len(PAPER_CONTEXT['batch_sizes']):.1f}-avg batch sweep on 8,192-sample subsets; this toy: a single 2-D/1-D quadratic, no dataset"
    return {
        "paper": PAPER_CONTEXT,
        "this_toy": this_toy,
        "scale_gap_note": ratio_datasets,
        "config_hash": sha256_json(PAPER_CONTEXT),
    }
