"""Seeded incident telemetry and the no-hindsight online diagnosis.

The bank contains one incident per truth kind, plus a second ``unresolved_mixed``
case, so the ambiguous cases are not collapsed into one bucket:

- ``true_divergence``                  loss really escalates; outcome "diverged"
- ``recoverable_transient_plateau``    same-run recovery; no restart, no config change
- ``seed_only_misleading_improvement`` one seed looks better; the paired audit disagrees
- ``unresolved_mixed`` (two cases)     conflicting signals, and insufficient evidence

``diagnose_online`` is a real function of the telemetry *prefix only*.  It is
handed a list of points with ``step <= decision_step`` and it rejects any input
carrying hidden truth keys, so the no-hindsight property is structural rather
than a promise in a comment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_dynamics.common import rate_object, sha256_json, sha256_text  # noqa: E402  (shared helpers)

TRACE_SEED_BASE = 90210

# Detector thresholds: stated simulation tolerances, i.e. project choices
# derived here, not constants from any paper.
DIVERGENCE_DELTA = 0.18
CONFLICT_RISE = 0.12
CONFLICT_FALL = 0.12
PLATEAU_TOL = 0.06
PLATEAU_DESCENT = 0.05
IMPROVEMENT_DELTA = 0.12

ONLINE_LABEL_SPACE = {
    "true_divergence": {"divergence_risk", "diverged", "abstain"},
    "recoverable_transient_plateau": {"wait_for_transient", "plateau", "abstain"},
    "unresolved_mixed": {"abstain", "unresolved"},
    "seed_only_misleading_improvement": {"improving", "abstain", "unresolved"},
}

HIDDEN_TRUTH_KEYS = {
    "truth_kind",
    "observed_outcome",
    "optimizer_restarted",
    "configuration_changed",
    "truth_revealed_after_step",
    "online_diagnosis",
    "incident_id",
}


def _diagnose(telemetry_prefix: list, threshold_scale: float = 1.0) -> str:
    """Core prefix-only detector; thresholds can be scaled for replay perturbations."""
    if not telemetry_prefix:
        return "abstain"
    present = set().union(*(set(point.keys()) for point in telemetry_prefix))
    leaked = present & HIDDEN_TRUTH_KEYS
    if leaked:
        raise ValueError(f"online diagnosis was handed hidden truth fields: {sorted(leaked)}")
    losses = [float(point["loss"]) for point in telemetry_prefix]
    n = len(losses)
    if n < 6:
        return "abstain"
    third = n // 3
    early = float(np.median(losses[:third]))
    middle = float(np.median(losses[third:2 * third]))
    late = float(np.median(losses[2 * third:]))
    magnitude = max(abs(early), 1e-9)
    first_change = (middle - early) / magnitude
    late_change = (late - middle) / magnitude
    divergence_delta = DIVERGENCE_DELTA * float(threshold_scale)
    conflict_rise = CONFLICT_RISE * float(threshold_scale)
    conflict_fall = CONFLICT_FALL * float(threshold_scale)
    plateau_tol = PLATEAU_TOL * float(threshold_scale)
    plateau_descent = PLATEAU_DESCENT * float(threshold_scale)
    improvement_delta = IMPROVEMENT_DELTA * float(threshold_scale)
    if late_change > divergence_delta:
        return "divergence_risk"
    if first_change > conflict_rise and late_change < -conflict_fall:
        return "abstain"
    if abs(late_change) <= plateau_tol and first_change < -plateau_descent:
        return "wait_for_transient"
    if late_change < -improvement_delta:
        return "improving"
    return "unresolved"


def diagnose_online(telemetry_prefix: list) -> str:
    """Diagnose from the prefix only; the label vocabulary is global.

    Compares medians of the first, middle and last thirds of the prefix.  The
    thresholds above are order-level simulation tolerances: the function is
    deliberately allowed to return ``abstain`` or ``unresolved`` instead of
    forcing a call.  The input is never mutated, and any hidden-truth key raises.
    """
    return _diagnose(telemetry_prefix, 1.0)


def _telemetry(steps, losses, lr: float, noise_scale: float, seed: int) -> list:
    rng = np.random.default_rng(int(seed))
    jitter = rng.normal(scale=float(noise_scale), size=len(losses))
    points = []
    for offset, step in enumerate(steps):
        loss = float(losses[offset]) + float(jitter[offset])
        points.append(
            {
                "step": int(step),
                "loss": round(loss, 6),
                "grad_norm": round(0.4 * float(np.sqrt(max(loss, 0.0))) + 0.02, 6),
                "lr": float(lr),
            }
        )
    return points


def _divergence_trace() -> list:
    losses = []
    for step in range(65):
        if step < 28:
            losses.append(2.10 - 0.012 * step + 0.02 * np.sin(0.7 * step))
        else:
            losses.append(1.764 * (1.135 ** (step - 28)))
    return _telemetry(range(65), losses, lr=0.05, noise_scale=0.015, seed=TRACE_SEED_BASE + 1)


def _plateau_trace() -> list:
    plateau_end = 1.320 - 0.0008 * (50 - 18)
    losses = []
    for step in range(85):
        if step <= 17:
            losses.append(2.40 - 1.06 * (1.0 - np.exp(-step / 7.0)))
        elif step <= 50:
            losses.append(1.320 - 0.0008 * (step - 18))
        else:
            losses.append(plateau_end - 0.52 * (1.0 - np.exp(-(step - 51) / 8.0)))
    return _telemetry(range(85), losses, lr=0.05, noise_scale=0.012, seed=TRACE_SEED_BASE + 2)


def _seed_only_trace() -> list:
    losses = [2.05 - 1.35 * (1.0 - np.exp(-step / 18.0)) for step in range(80)]
    return _telemetry(range(80), losses, lr=0.05, noise_scale=0.010, seed=TRACE_SEED_BASE + 3)


def _mixed_conflict_trace() -> list:
    losses = []
    for step in range(71):
        if step <= 22:
            losses.append(1.55 + 0.030 * step)
        elif step <= 52:
            losses.append(2.21 - 1.10 * (1.0 - np.exp(-(step - 23) / 9.0)))
        else:
            losses.append(1.155 + 0.004 * np.sin(0.5 * step))
    return _telemetry(range(71), losses, lr=0.05, noise_scale=0.012, seed=TRACE_SEED_BASE + 4)


def _mixed_flat_trace() -> list:
    losses = [1.62 + 0.02 * np.sin(0.9 * step) for step in range(61)]
    return _telemetry(range(61), losses, lr=0.05, noise_scale=0.030, seed=TRACE_SEED_BASE + 5)


def _record(
    incident_id: str,
    truth_kind: str,
    telemetry: list,
    decision_step: int,
    truth_revealed_after_step: int,
    observed_outcome: str,
    optimizer_restarted: bool,
    configuration_changed: bool,
    description: str,
) -> dict:
    prefix = [dict(point) for point in telemetry if int(point["step"]) <= int(decision_step)]
    label = diagnose_online(prefix)
    allowed = ONLINE_LABEL_SPACE[truth_kind]
    if label not in allowed:
        raise AssertionError(
            f"{incident_id}: online detector returned {label!r}, not in {sorted(allowed)}"
        )
    if not any(int(point["step"]) > int(decision_step) for point in telemetry):
        raise AssertionError(f"{incident_id}: needs telemetry after the decision step")
    if int(truth_revealed_after_step) < int(decision_step):
        raise AssertionError(f"{incident_id}: truth cannot be revealed before the decision")
    return {
        "incident_id": incident_id,
        "truth_kind": truth_kind,
        "description": description,
        "telemetry": telemetry,
        "decision_step": int(decision_step),
        "truth_revealed_after_step": int(truth_revealed_after_step),
        "observed_outcome": observed_outcome,
        "optimizer_restarted": bool(optimizer_restarted),
        "configuration_changed": bool(configuration_changed),
        "online_diagnosis": {
            "decision_step": int(decision_step),
            "uses_truth": False,
            "telemetry_inputs": prefix,
            "n_inputs": len(prefix),
            "label": label,
            "detector": "diagnose_online (third-median comparison, prefix only)",
            "thresholds": {
                "divergence_delta": DIVERGENCE_DELTA,
                "conflict_rise": CONFLICT_RISE,
                "conflict_fall": CONFLICT_FALL,
                "plateau_tolerance": PLATEAU_TOL,
                "improvement_delta": IMPROVEMENT_DELTA,
            },
        },
    }


def build_bank() -> dict:
    """The ``incident_bank`` object published in walkthrough_results.json."""
    incidents = [
        _record(
            "inc-divergence-01",
            "true_divergence",
            _divergence_trace(),
            decision_step=36,
            truth_revealed_after_step=64,
            observed_outcome="diverged",
            optimizer_restarted=True,
            configuration_changed=True,
            description=(
                "loss creeps down, then compounds at ~13.5% per step; the online call made at "
                "step 36 is divergence_risk, and the run is confirmed diverged afterwards"
            ),
        ),
        _record(
            "inc-plateau-01",
            "recoverable_transient_plateau",
            _plateau_trace(),
            decision_step=40,
            truth_revealed_after_step=84,
            observed_outcome="recovered_without_restart",
            optimizer_restarted=False,
            configuration_changed=False,
            description=(
                "a slow mode stalls the loss on a plateau from ~step 18 to ~step 50, then the "
                "same run descends again; the correct online call is wait_for_transient"
            ),
        ),
        _record(
            "inc-seed-only-01",
            "seed_only_misleading_improvement",
            _seed_only_trace(),
            decision_step=55,
            truth_revealed_after_step=79,
            observed_outcome="apparent_improvement_not_reproducible",
            optimizer_restarted=False,
            configuration_changed=False,
            description=(
                "this single seeded curve looks like a clean win; the paired-seed audit in "
                "seed_audit reverses its sign, so the curve alone cannot support the claim"
            ),
        ),
        _record(
            "inc-mixed-conflict-01",
            "unresolved_mixed",
            _mixed_conflict_trace(),
            decision_step=34,
            truth_revealed_after_step=70,
            observed_outcome="unresolved_conflicting_signals",
            optimizer_restarted=False,
            configuration_changed=False,
            description=(
                "the prefix contains both an escalation and a subsequent fall, so the online "
                "detector abstains instead of inventing a winner"
            ),
        ),
        _record(
            "inc-mixed-flat-01",
            "unresolved_mixed",
            _mixed_flat_trace(),
            decision_step=30,
            truth_revealed_after_step=60,
            observed_outcome="unresolved_insufficient_evidence",
            optimizer_restarted=False,
            configuration_changed=False,
            description=(
                "the prefix is flat within its own noise floor; the online detector reports "
                "unresolved rather than a direction"
            ),
        ),
    ]
    return {
        "incidents": incidents,
        "truth_kinds": sorted({incident["truth_kind"] for incident in incidents}),
        "label_space": {key: sorted(value) for key, value in ONLINE_LABEL_SPACE.items()},
        "diagnosis_protocol": (
            "diagnose_online() is called with exactly the telemetry points whose step is <= "
            "decision_step; the stored telemetry_inputs are that same prefix, and the detector "
            "raises if any hidden-truth key is present in its input"
        ),
        "trace_generation": {
            "seed_base": TRACE_SEED_BASE,
            "note": "each trace is a deterministic seeded synthetic training trace",
        },
    }


def account_incidents(bank: dict) -> dict:
    """Tally the online labels, keeping abstentions visible and out of the denominator."""
    counts: dict = {}
    for incident in bank["incidents"]:
        label = incident["online_diagnosis"]["label"]
        counts[label] = counts.get(label, 0) + 1
    abstentions = counts.get("abstain", 0) + counts.get("unresolved", 0)
    scored = sum(count for label, count in counts.items() if label not in {"abstain", "unresolved"})
    return {
        "outcome_counts": counts,
        "scored_count": int(scored),
        "abstention_count": int(abstentions),
        "accuracy_denominator": int(scored),
        "accuracy_denominator_excludes_unresolved": True,
        "note": (
            "abstain and unresolved are reported as their own outcomes; they are excluded "
            "from the accuracy denominator instead of being scored as correct or incorrect"
        ),
    }


# ---------------------------------------------------------------------------
# prefix-only detector replay and perturbation audit
# ---------------------------------------------------------------------------

REPLAY_PERTURBATIONS = (-0.20, -0.10, 0.0, 0.10, 0.20)
REPLAY_RUN_ID = "r02-detector-replay"


def build_replay_audit() -> dict:
    """Replay every incident's prefix through the detector and perturb the thresholds.

    A replay stores the telemetry-only prefix, the original label and the recomputed
    label; a perturbation stores the signed threshold scaling, the resulting label
    and ``uses_hidden_truth=False``.  No accuracy, false-alarm or production field is
    produced: the sweep only shows where the boundary diagnostic abstains.
    """
    bank = build_bank()
    replays = []
    perturbations = []
    for incident in bank["incidents"]:
        cutoff = int(incident["decision_step"])
        prefix = [dict(point) for point in incident["telemetry"] if int(point["step"]) <= cutoff]
        original = incident["online_diagnosis"]["label"]
        replay_label = diagnose_online(prefix)
        prefix_hash = sha256_json(prefix)
        replays.append(
            {
                "incident_id": incident["incident_id"],
                "decision_step": cutoff,
                "telemetry_prefix": prefix,
                "original_label": original,
                "replay_label": replay_label,
                "prefix_hash": prefix_hash,
                "uses_hidden_truth": False,
            }
        )
        for perturbation in REPLAY_PERTURBATIONS:
            perturbations.append(
                {
                    "incident_id": incident["incident_id"],
                    "case": "replay" if perturbation == 0.0 else "perturb",
                    "perturbation": float(perturbation),
                    "original_label": original,
                    "replay_label": replay_label,
                    "label": _diagnose(prefix, 1.0 + float(perturbation)),
                    "decision_step": cutoff,
                    "prefix_hash": prefix_hash,
                    "uses_hidden_truth": False,
                }
            )
    return {
        "run_id": REPLAY_RUN_ID,
        "seed": int(TRACE_SEED_BASE),
        "replays": replays,
        "perturbations": perturbations,
        "protocol": (
            "each incident prefix is replayed through diagnose_online unchanged (original label "
            "must equal the replay label); perturbations scale every detector threshold by "
            "1 + perturbation and record the resulting label, including abstain/unresolved"
        ),
        "note": (
            "this is a boundary-diagnostic sweep on synthetic telemetry, not a production "
            "accuracy or false-alarm estimate"
        ),
    }


# ---------------------------------------------------------------------------
# telemetry-only perturbations and stability rates
# ---------------------------------------------------------------------------

R03_TELEMETRY_OPERATORS = ("gaussian_noise", "scale_drift", "observation_dropout")
R03_TELEMETRY_STRENGTHS = (0.0, 0.01, 0.03, 0.05)
R03_TELEMETRY_SEEDS = (501, 502, 503, 504, 505)

_R03_RULES = {
    "gaussian_noise": "add N(0, (strength * median |loss|)^2) noise to the loss column",
    "scale_drift": "multiply each loss by 1 + strength * a linear -1..1 ramp along the prefix",
    "observation_dropout": "drop round(strength * n) seeded observations from the prefix",
}


def _perturb_prefix(prefix: list, operator: str, strength: float, seed: int, incident_id: str) -> list:
    """Telemetry-only perturbation of one prefix; strength 0 is an exact identity."""
    perturbed = [dict(point) for point in prefix]
    if strength == 0.0:
        return perturbed
    key = f"{operator}|{strength}|{seed}|{incident_id}"
    rng = np.random.default_rng(int(sha256_text(key)[:12], 16))
    losses = np.asarray([float(point["loss"]) for point in perturbed], dtype=float)
    if operator == "gaussian_noise":
        scale = float(np.median(np.abs(losses))) if losses.size else 1.0
        noise = rng.normal(0.0, float(strength) * max(scale, 1e-9), size=losses.shape)
        for point, value in zip(perturbed, losses + noise):
            point["loss"] = float(value)
        return perturbed
    if operator == "scale_drift":
        n = len(losses)
        ramp = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(1)
        for point, value in zip(perturbed, losses * (1.0 + float(strength) * ramp)):
            point["loss"] = float(value)
        return perturbed
    if operator == "observation_dropout":
        n = len(losses)
        drop = min(int(round(float(strength) * n)), max(n - 1, 0))
        if drop <= 0:
            return perturbed
        dropped = set(int(index) for index in rng.choice(n, size=drop, replace=False))
        return [point for index, point in enumerate(perturbed) if index not in dropped]
    raise ValueError(f"unknown telemetry operator {operator!r}")


def telemetry_audit() -> dict:
    """All 300 telemetry-only attempts, with stability and abstention kept as separate rates."""
    bank = build_bank()
    incidents_list = bank["incidents"]
    configurations = {}
    attempt_rows = []
    for operator in R03_TELEMETRY_OPERATORS:
        for strength in R03_TELEMETRY_STRENGTHS:
            for seed in R03_TELEMETRY_SEEDS:
                for incident in incidents_list:
                    incident_id = incident["incident_id"]
                    cutoff = int(incident["decision_step"])
                    base_prefix = [
                        dict(point) for point in incident["telemetry"] if int(point["step"]) <= cutoff
                    ]
                    perturbed_prefix = _perturb_prefix(base_prefix, operator, float(strength), int(seed), incident_id)
                    base_label = diagnose_online(base_prefix)
                    label = diagnose_online(perturbed_prefix)
                    run_id = (
                        f"r03-telemetry-{operator}-s{int(round(strength * 100)):02d}-"
                        f"seed{seed}-{incident_id}"
                    )
                    config = {
                        "operator": operator,
                        "strength": float(strength),
                        "seed": int(seed),
                        "incident_id": incident_id,
                        "run_id": run_id,
                        "prefix_length": len(base_prefix),
                        "perturbation_rule": _R03_RULES[operator],
                    }
                    config_hash = sha256_json(config)
                    configurations[config_hash] = config
                    attempt_rows.append(
                        {
                            "operator": operator,
                            "strength": float(strength),
                            "seed": int(seed),
                            "incident_id": incident_id,
                            "run_id": run_id,
                            "config_hash": config_hash,
                            "base_prefix": base_prefix,
                            "perturbed_prefix": perturbed_prefix,
                            "base_label": base_label,
                            "label": label,
                            "base_prefix_hash": sha256_json(base_prefix),
                            "prefix_hash": sha256_json(perturbed_prefix),
                        }
                    )
    rates = []
    for operator in R03_TELEMETRY_OPERATORS:
        for strength in R03_TELEMETRY_STRENGTHS:
            subset = [
                row
                for row in attempt_rows
                if row["operator"] == operator and row["strength"] == float(strength)
            ]
            stable = int(sum(row["label"] == row["base_label"] for row in subset))
            abstained = int(sum(row["label"] in {"abstain", "unresolved"} for row in subset))
            rates.append(
                {
                    "operator": operator,
                    "strength": float(strength),
                    "stability_count": stable,
                    "abstention_count": abstained,
                    "stability": rate_object(stable, len(subset)),
                    "abstention": rate_object(abstained, len(subset)),
                }
            )
    return {
        "configurations": configurations,
        "attempt_rows": attempt_rows,
        "stability_rates": rates,
        "operators": list(R03_TELEMETRY_OPERATORS),
        "strengths": [float(value) for value in R03_TELEMETRY_STRENGTHS],
        "seeds": [int(value) for value in R03_TELEMETRY_SEEDS],
        "protocol": (
            "telemetry-only perturbations of every incident prefix; stability compares the "
            "perturbed label with the unperturbed online label, abstention stays a separate "
            "outcome, and both rates use all 25 seed x incident attempts per cell"
        ),
        "note": (
            "this is a synthetic telemetry robustness diagnostic, not production accuracy or "
            "false-alarm evidence"
        ),
    }
