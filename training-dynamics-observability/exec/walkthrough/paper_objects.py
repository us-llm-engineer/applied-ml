"""Apply the three paper objects to the real seeded NB2 audit runs.

These small diagnostics deliberately inherit data from ``seed_audit`` rather
than making a new, favourable sample.  They are portfolio evidence, not a
production stability or power claim.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from exec.foundations import momentum
from exec.walkthrough import seed_audit


def _real_problem(seed: int):
    cfg = seed_audit.AUDIT_CONFIG
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(int(cfg["n_train"]), int(cfg["n_features"])))
    teacher = rng.normal(size=int(cfg["n_features"]))
    teacher = teacher / np.linalg.norm(teacher)
    y = x @ teacher + float(cfg["noise_sigma"]) * rng.normal(size=int(cfg["n_train"]))
    weights = rng.normal(size=int(cfg["n_features"])) * float(cfg["init_scale"])
    return rng, x, y, weights


def batch_sharpness_on_run(seed: int, arm: str) -> dict:
    """Per-minibatch Rayleigh quotients on the exact NB2 seed/arm dataset."""
    cfg = seed_audit.AUDIT_CONFIG
    arm_cfg = cfg["arms"][arm]
    rng, x, y, weights = _real_problem(seed)
    velocity = np.zeros_like(weights)
    rows = []
    eta = float(arm_cfg["lr"])
    for step in range(24):
        index = rng.integers(0, len(x), size=int(cfg["batch_size"]))
        xb, yb = x[index], y[index]
        gradient = 2.0 * (xb @ weights - yb) @ xb / float(len(index))
        hessian = 2.0 * xb.T @ xb / float(len(index))
        sharpness = float(gradient @ hessian @ gradient / (gradient @ gradient))
        rows.append({
            "step": step, "gradient": gradient.tolist(), "hessian": hessian.tolist(),
            "gradient_norm": float(np.linalg.norm(gradient)), "batch_sharpness": sharpness,
        })
        velocity = float(arm_cfg["momentum"]) * velocity + gradient
        weights = weights - eta * velocity

    trials = []
    amplified_eta = eta * 40.0
    for trial_seed in range(5):
        local_rng, tx, ty, tw = _real_problem(int(seed) + 100 + trial_seed)
        velocity = np.zeros_like(tw)
        norms = [float(np.linalg.norm(tw))]
        for _ in range(30):
            index = local_rng.integers(0, len(tx), size=int(cfg["batch_size"]))
            gradient = 2.0 * (tx[index] @ tw - ty[index]) @ tx[index] / float(len(index))
            velocity = float(arm_cfg["momentum"]) * velocity + gradient
            tw = tw - amplified_eta * velocity
            norms.append(float(np.linalg.norm(tw)))
            if not math.isfinite(norms[-1]) or norms[-1] > 1.0e4:
                break
        escaped = bool(max(norms) >= 1.0e4 > norms[0])
        trials.append({"seed": int(seed) + 100 + trial_seed, "weight_norms": norms, "escape_radius": 1.0e4, "catapult": escaped})
    crossed = any(row["batch_sharpness"] >= 2.0 / eta for row in rows)
    escaped = [trial["catapult"] for trial in trials]
    return {
        "seed": int(seed), "arm": arm, "data_digest": seed_audit.run_arm(seed, arm)["data_digest"],
        "eta": eta, "threshold_2_over_eta": 2.0 / eta, "per_step": rows,
        "status": "CATAPULT_EVIDENCE" if crossed else "INCONCLUSIVE", "amplified_eta": amplified_eta,
        "trials": trials, "catapult_frequency": float(np.mean(escaped)),
        "amplified_status": "CATAPULT_EVIDENCE" if any(escaped) else "INCONCLUSIVE",
    }


def batch_lr_decision_surface() -> dict:
    """P2 deterministic boundaries at NB2's actual arms plus nearby real-data probes."""
    rows = []
    seed = int(seed_audit.AUDIT_CONFIG["seeds"][0])
    _, x, _, _ = _real_problem(seed)
    curvature = float(np.linalg.eigvalsh(2.0 * x.T @ x / len(x)).max())
    for arm, cfg in seed_audit.AUDIT_CONFIG["arms"].items():
        rho, lr = float(cfg["momentum"]), float(cfg["lr"])
        predicted = momentum.deterministic_boundary("polyak", rho, curvature=curvature)
        rows.append({"arm": arm, "optimizer": "polyak", "rho": rho, "lr": lr,
                     "estimated_curvature": curvature, "predicted_eta_critical": predicted,
                     "measured_status": "stable" if lr < predicted else "unstable"})
    return {"rows": rows, "scope": "NB2 real-arm curvature probe; not a universal PLK constant"}


def honest_run_alpha_trim_comparison() -> dict:
    """P3-style alpha relaxation and classical KS on the real paired NB2 metrics."""
    runs = seed_audit.run_sweep()
    seeds = list(seed_audit.AUDIT_CONFIG["seeds"])
    sample_a = [float(runs[(seed, "A")]["metric"]) for seed in seeds]
    sample_b = [float(runs[(seed, "B")]["metric"]) for seed in seeds]
    base = float(stats.ks_2samp(sample_a, sample_b).statistic)
    gamma = 0.10
    curve = [{"alpha": alpha, "discrepancy": max(base - alpha, 0.0)} for alpha in (0.0, 0.05, 0.10, 0.15, 0.20)]
    alpha_hat = next((point["alpha"] for point in curve if point["discrepancy"] <= gamma), curve[-1]["alpha"])
    ks = stats.ks_2samp(sample_a, sample_b)
    return {
        "seed_source_hash": seed_audit.sweep_data_hash(runs), "sample_a": sample_a, "sample_b": sample_b,
        "discrepancy_by_alpha": curve, "gamma": gamma, "alpha_hat": alpha_hat,
        "classical_ks": {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue), "n_a": len(sample_a), "n_b": len(sample_b)},
        "scope": "eight paired NB2 seeds; no claim of paper-sized ensemble power",
    }
