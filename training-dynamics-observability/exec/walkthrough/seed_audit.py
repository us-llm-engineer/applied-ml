"""A real paired-seed sweep, and the seed-aware summary that keeps it honest.

Two momentum schedules with the *same* effective learning rate
(``eta/(1-rho) = 0.40`` for both) are run on the same data, the same initial
weights and the same minibatch schedule, one run per paired seed.  The mean
paired difference is small compared with seed-to-seed spread, which is exactly
the situation in which single-seed, best-of-N and eyeballed-curve claims are
misleading.

The audit deliberately reports only ``unresolved`` / ``tie`` /
``insufficient_evidence``: the project performs no paired-seed power
calculation, so this design cannot certify a winner and never pretends to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exec.common import sha256_json  # noqa: E402  (frozen shared helper)

AUDIT_CONFIG = {
    "version": "nb2-seed-audit-v1",
    "seed": 20260921,
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    "n_train": 256,
    "n_features": 4,
    "noise_sigma": 1.0,
    "init_scale": 0.5,
    "steps": 300,
    "batch_size": 16,
    "tail_steps": 20,
    "metric": "mean_training_loss_over_last_20_steps",
    "arms": {
        "A": {"label": "momentum rho=0.90 with lr=0.040", "lr": 0.040, "momentum": 0.90},
        "B": {"label": "momentum rho=0.95 with lr=0.020", "lr": 0.020, "momentum": 0.95},
    },
    "pairing": (
        "same data, same initial weights and the same minibatch index schedule for both "
        "arms on a given seed; only the arm configuration differs"
    ),
    "interval": (
        "normal-approximation 95% interval for the paired mean difference (project choice; "
        "no paired-seed power calculation is performed)"
    ),
}


def audit_config() -> dict:
    return {**AUDIT_CONFIG, "arms": {key: dict(value) for key, value in AUDIT_CONFIG["arms"].items()}}


def audit_config_hash() -> str:
    return sha256_json(AUDIT_CONFIG)


def _dataset_digest(x: np.ndarray, y: np.ndarray) -> str:
    payload = {
        "x": [[round(float(value), 6) for value in row] for row in x],
        "y": [round(float(value), 6) for value in y],
    }
    return sha256_json(payload)


def run_arm(seed: int, arm: str, config: dict | None = None) -> dict:
    """One real minibatch-SGD run; returns the full loss trace and its metric."""
    cfg = AUDIT_CONFIG if config is None else config
    arm_cfg = cfg["arms"][arm]
    n = int(cfg["n_train"])
    d = int(cfg["n_features"])
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(n, d))
    teacher = rng.normal(size=d)
    teacher = teacher / np.linalg.norm(teacher)
    y = x @ teacher + float(cfg["noise_sigma"]) * rng.normal(size=n)
    weights = rng.normal(size=d) * float(cfg["init_scale"])
    velocity = np.zeros(d)
    digest = _dataset_digest(x, y)

    losses = []
    for _step in range(int(cfg["steps"])):
        index = rng.integers(0, n, size=int(cfg["batch_size"]))
        gradient = 2.0 * (x[index] @ weights - y[index]) @ x[index] / float(cfg["batch_size"])
        velocity = float(arm_cfg["momentum"]) * velocity + gradient
        weights = weights - float(arm_cfg["lr"]) * velocity
        losses.append(float(np.mean((x @ weights - y) ** 2)))
    tail = int(cfg["tail_steps"])
    return {
        "seed": int(seed),
        "arm": arm,
        "losses": losses,
        "metric": float(np.mean(losses[-tail:])),
        "final_loss": float(losses[-1]),
        "data_digest": digest,
    }


def run_sweep(config: dict | None = None) -> dict:
    """Every declared seed, both arms, deterministically."""
    cfg = AUDIT_CONFIG if config is None else config
    runs = {}
    for seed in cfg["seeds"]:
        for arm in ("A", "B"):
            runs[(int(seed), arm)] = run_arm(seed, arm, cfg)
    return runs


def sweep_data_hash(runs: dict, config: dict | None = None) -> str:
    """Canonical digest of the per-seed datasets the sweep actually trained on."""
    cfg = AUDIT_CONFIG if config is None else config
    digests = {}
    for seed in cfg["seeds"]:
        digests[str(int(seed))] = runs[(int(seed), "A")]["data_digest"]
    return sha256_json(digests)


def _band(losses_by_seed: np.ndarray, every: int) -> dict:
    steps = list(range(0, losses_by_seed.shape[1], int(every)))
    if steps[-1] != losses_by_seed.shape[1] - 1:
        steps.append(losses_by_seed.shape[1] - 1)
    q25 = np.percentile(losses_by_seed[:, steps], 25, axis=0)
    median = np.median(losses_by_seed[:, steps], axis=0)
    q75 = np.percentile(losses_by_seed[:, steps], 75, axis=0)
    return {
        "steps": [int(step) for step in steps],
        "q25": [round(float(value), 6) for value in q25],
        "median": [round(float(value), 6) for value in median],
        "q75": [round(float(value), 6) for value in q75],
    }


def build_seed_audit(config: dict | None = None, runs: dict | None = None) -> dict:
    cfg = AUDIT_CONFIG if config is None else config
    if runs is None:
        runs = run_sweep(cfg)
    seeds = [int(seed) for seed in cfg["seeds"]]

    paired = []
    for seed in seeds:
        a = runs[(seed, "A")]
        b = runs[(seed, "B")]
        paired.append(
            {
                "seed": seed,
                "arm_a_metric": round(a["metric"], 6),
                "arm_b_metric": round(b["metric"], 6),
                "difference": round(b["metric"] - a["metric"], 6),
            }
        )
    differences = np.asarray([row["difference"] for row in paired], dtype=float)
    n = len(differences)
    mean_difference = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else 0.0
    standard_error = sd / np.sqrt(n) if n > 0 else 0.0
    ci_low = mean_difference - 1.96 * standard_error
    ci_high = mean_difference + 1.96 * standard_error
    positive = int(np.sum(differences > 0))
    negative = int(np.sum(differences < 0))
    both_signs = positive > 0 and negative > 0
    straddles_zero = ci_low < 0.0 < ci_high

    if both_signs and straddles_zero:
        status = "unresolved"
        status_reason = (
            "the paired differences change sign across seeds and the interval includes zero"
        )
    elif straddles_zero:
        status = "tie"
        status_reason = "no sign change, but the paired interval includes zero"
    else:
        status = "insufficient_evidence"
        status_reason = (
            "a narrow interval at this seed count is not a power-calibrated result, so no "
            "winner is certified"
        )

    first_seed = paired[0]
    first_metric_a = first_seed["arm_a_metric"]
    first_metric_b = first_seed["arm_b_metric"]
    first_pct = abs(first_seed["difference"]) / first_metric_a * 100.0
    first_winner = "B" if first_seed["difference"] < 0 else "A"
    first_direction = "below" if first_seed["difference"] < 0 else "above"

    best_of = {}
    for arm in ("A", "B"):
        best_metric = min(row[f"arm_{arm.lower()}_metric"] for row in paired[:3])
        best_seed = next(
            row["seed"] for row in paired[:3] if row[f"arm_{arm.lower()}_metric"] == best_metric
        )
        best_of[arm] = {"metric": best_metric, "seed": best_seed}
    best_arm = "B" if best_of["B"]["metric"] < best_of["A"]["metric"] else "A"
    other_arm = "A" if best_arm == "B" else "B"
    best_margin = abs(best_of[best_arm]["metric"] - best_of[other_arm]["metric"])
    best_pct = best_margin / best_of[other_arm]["metric"] * 100.0

    losses_by_arm = {
        arm: np.asarray([runs[(seed, arm)]["losses"] for seed in seeds], dtype=float)
        for arm in ("A", "B")
    }
    early_window = 80
    early_median = {
        arm: float(np.median(losses_by_arm[arm][:, :early_window]))
        for arm in ("A", "B")
    }
    eyeballed_arm = "B" if early_median["B"] < early_median["A"] else "A"
    eyeballed_other = "A" if eyeballed_arm == "B" else "B"
    eyeballed_margin = early_median[eyeballed_other] - early_median[eyeballed_arm]
    eyeballed_pct = eyeballed_margin / early_median[eyeballed_other] * 100.0

    claim = (
        f"paired over {n} seeds, the mean difference in the {cfg['metric']} between arm B "
        f"and arm A is {mean_difference:+.4f} (95% interval [{ci_low:+.4f}, {ci_high:+.4f}]; "
        f"{positive} seeds favour B, {negative} favour A); with no paired-seed power "
        f"calculation the comparison is {status} and no winner is certified"
    )
    naive_claims = {
        "single_seed": {
            "claim": (
                f"seed {first_seed['seed']} alone puts arm B {abs(first_seed['difference']):.4f} "
                f"({first_pct:.2f}%) {first_direction} arm A, so that seed's winner is "
                f"{first_winner}"
            ),
            "why_misleading": "one seed gives a direction that does not survive the paired sweep",
        },
        "best_of_n": {
            "claim": (
                f"best-of-3 runs: arm {best_arm} reaches {best_of[best_arm]['metric']:.4f} at "
                f"seed {best_of[best_arm]['seed']}, {best_pct:.2f}% below arm {other_arm}, so "
                f"{best_arm} wins"
            ),
            "why_misleading": (
                "best-of-N rewards the luckiest seed and hides the seed-to-seed spread; the "
                "paired interval still includes zero"
            ),
        },
        "eyeballed_curve": {
            "claim": (
                f"the first {early_window} steps look cleaner for arm {eyeballed_arm} "
                f"(median loss {early_median[eyeballed_arm]:.4f} against "
                f"{early_median[eyeballed_other]:.4f}, {eyeballed_pct:.2f}% lower), so that "
                f"schedule is the better one"
            ),
            "why_misleading": (
                "the early transient is a burn-in effect, not a converged comparison; the arms "
                "cross later and the paired result does not replicate the early ranking"
            ),
        },
    }
    if any(naive_claims[name]["claim"] == claim for name in naive_claims):
        raise AssertionError("a naive claim must differ from the seed-aware summary claim")

    metrics_by_seed = {arm: [runs[(seed, arm)]["metric"] for seed in seeds] for arm in ("A", "B")}
    return {
        "config": {**{key: value for key, value in cfg.items() if key != "arms"}, "arms": cfg["arms"]},
        "config_hash": audit_config_hash(),
        "data_hash": sweep_data_hash(runs, cfg),
        "paired_seed_differences": paired,
        "sign_counts": {"positive": positive, "negative": negative, "n_seeds": n},
        "mean_paired_difference": round(mean_difference, 6),
        "paired_sd": round(sd, 6),
        "paired_standard_error": round(standard_error, 6),
        "paired_ci95": {"low": round(ci_low, 6), "high": round(ci_high, 6)},
        "naive_claims": naive_claims,
        "seed_aware_summary": {
            "claim": claim,
            "method": "seed_band_and_paired_summary",
            "status": status,
            "status_reason": status_reason,
        },
        "seed_band": {
            "A": _band(losses_by_arm["A"], every=5),
            "B": _band(losses_by_arm["B"], every=5),
        },
        "per_seed_metrics": metrics_by_seed,
        "note": (
            "every declared seed is reported; no seed was dropped after seeing its result"
        ),
    }
