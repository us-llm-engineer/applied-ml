"""Retraining policies over paired synthetic deployment streams.

The controller is the intervention layer (``derived here``): no source paper
defines a lifecycle, promotion gate, rollback rule, cooldown, or audit budget.
Each policy below decides *when* to open an audit campaign; the controller
decides what may happen next, and the engine charges the label, compute,
routing and unmitigated-risk costs of every decision.

Policies
--------
``no_action``     monitor only; never intervenes.
``drift_only``    intervene on a covariate-drift statistic alone.
``threshold``     intervene on a certified batch SJS gap above a threshold.
``abstain_route`` intervene by routing to human review, never retraining.
``corroborated``  intervene when a time-uniform error-propensity alarm is
                  corroborated by a windowed effect-size check.

Reported metrics keep false retraining, missed deterioration, detection delay,
recovery, and cost separate; no aggregate hides another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .controller import Controller, IllegalTransition
from .estimators import sequential_tail_monitor_arrays
from .synthetic import DeploymentStream, ShiftScenario, _brier, _stable_seed

POLICY_NAMES = ("no_action", "drift_only", "threshold", "abstain_route", "corroborated")
HARMFUL_REGIMES = ("sparse_joint", "harmful_concept", "d3m_regime_2")

# Cost units and decision thresholds, all `derived here`.
C_LABEL = 0.005
C_COMPUTE = 3.0
C_ROUTE = 1.0
C_RISK = 1.0
HARM_TOL = 0.04
RECOVERY_TOL = 0.05
GAP_MIN = 0.05
AUDIT_SIZE = 256
COOLDOWN = 8
AUDIT_BUDGET_LABELS = 512
AUDIT_BUDGET = AUDIT_BUDGET_LABELS // AUDIT_SIZE
PROMOTION_MIN_GAIN = 0.005
DRIFT_WINDOW_STEPS = 4
SEQ_WINDOW = 512
SEQ_EFFECT_MIN = 0.05
SEQ_ALPHA = 0.05
GAP_REFIT_STEPS = 12


@dataclass
class StreamSignals:
    """Policy inputs computed once per stream and shared across policies."""

    drift: np.ndarray
    drift_threshold: float
    gap: np.ndarray
    gap_threshold: float
    seq_rate: np.ndarray
    seq_corroborated: np.ndarray
    seq_source_rate: float
    onset_step: int


@dataclass
class PolicyOutcome:
    """Per-stream outcome for one policy."""

    policy: str
    seed: int
    run_id: str
    false_retrain: bool
    missed: bool
    detection_delay: float
    recovered: bool
    cost: float
    retrains: int
    audits: int
    routed: int
    events: int
    final_risk: float
    timeline: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over the leading axis with a strictly causal window."""
    c = np.cumsum(x, axis=0, dtype=float)
    out = np.empty_like(c)
    out[:window] = c[:window] / np.arange(1, window + 1)[:, None]
    out[window:] = (c[window:] - c[:-window]) / window
    return out


def _drift_threshold(source_x: np.ndarray, window: int, n_null: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    mu, sd = source_x.mean(axis=0), source_x.std(axis=0)
    sd[sd == 0] = 1.0
    stats = np.empty(n_null)
    for i in range(n_null):
        idx = rng.choice(len(source_x), size=window, replace=True)
        stats[i] = np.max(np.abs(source_x[idx].mean(axis=0) - mu) / sd)
    return float(np.quantile(stats, 0.999))


def _gap_proxy(
    source_x: np.ndarray, source_loss: np.ndarray, window_x: np.ndarray, seed: int = 0
) -> float:
    """Fast SJS-style reweighted gap on a window (quadratic domain classifier)."""
    mu, sd = source_x.mean(axis=0), source_x.std(axis=0)
    sd[sd == 0] = 1.0

    def phi(a: np.ndarray) -> np.ndarray:
        z = (a - mu) / sd
        return np.hstack([z, z**2])

    sub = min(len(source_x), 1200)
    idx = np.random.default_rng(seed).choice(len(source_x), size=sub, replace=False)
    ps, pt = phi(source_x[idx]), phi(window_x)
    pooled = np.vstack([ps, pt])
    labels = np.concatenate([np.zeros(len(ps)), np.ones(len(pt))])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(pooled, labels)
    r = np.clip(clf.predict_proba(ps)[:, 1], 1e-6, 1 - 1e-6)
    w = np.clip((r / (1 - r)) * (len(pt) / len(ps)), 0.0, 50.0)
    sl = source_loss[idx]
    return float(np.sum(w * sl) / np.sum(w) - np.mean(sl))


def _step_values(sample_values: np.ndarray, batch_size: int, n_steps: int) -> np.ndarray:
    """Reduce a per-sample series to the value at each step's last arrival."""
    idx = np.minimum((np.arange(1, n_steps + 1) * batch_size) - 1, len(sample_values) - 1)
    return np.asarray(sample_values)[idx]


def compute_signals(scenario: ShiftScenario, stream: DeploymentStream, seed: int = 0) -> StreamSignals:
    """Per-step policy inputs: drift, batch gap proxy, sequential corroboration."""
    x = stream.x
    b = stream.batch_size
    mu, sd = scenario.source_x.mean(axis=0), scenario.source_x.std(axis=0)
    sd[sd == 0] = 1.0
    roll = _rolling_mean(x, DRIFT_WINDOW_STEPS * b)
    drift_sample = np.max(np.abs(roll - mu) / sd, axis=1)
    drift = _step_values(drift_sample, b, stream.n_steps)
    drift_cut = _drift_threshold(scenario.source_x, DRIFT_WINDOW_STEPS * b, seed=seed)

    monitor = sequential_tail_monitor_arrays(
        source_x=scenario.source_x,
        source_loss=scenario.source_loss,
        source_score=scenario.source_score,
        target_x=stream.x,
        target_score=stream.champion_score,
        target_loss=stream.loss,
        alpha=SEQ_ALPHA,
    )
    flags = monitor.selector_flags.astype(float)
    c = np.cumsum(flags)
    rate_sample = np.empty_like(c)
    rate_sample[:SEQ_WINDOW] = c[:SEQ_WINDOW] / np.arange(1, SEQ_WINDOW + 1)
    rate_sample[SEQ_WINDOW:] = (c[SEQ_WINDOW:] - c[:-SEQ_WINDOW]) / SEQ_WINDOW
    n_windows = max(1, len(flags) // SEQ_WINDOW)
    w_win = float(np.sqrt(np.log(2.0 * n_windows / SEQ_ALPHA) / (2.0 * SEQ_WINDOW)))
    corrob_sample = rate_sample > (monitor.source_tail_rate + SEQ_EFFECT_MIN + w_win)
    seq_rate = _step_values(rate_sample, b, stream.n_steps)
    seq_corrob = _step_values(corrob_sample, b, stream.n_steps).astype(bool)

    gap = np.empty(stream.n_steps)
    for t in range(stream.n_steps):
        if t % GAP_REFIT_STEPS == 0 or t == stream.n_steps - 1:
            lo = max(0, (t + 1) * b - DRIFT_WINDOW_STEPS * b)
            gap[t] = _gap_proxy(scenario.source_x, scenario.source_loss, x[lo : (t + 1) * b], seed=seed)
        else:
            gap[t] = gap[t - 1] if t else 0.0

    return StreamSignals(
        drift=drift,
        drift_threshold=drift_cut,
        gap=gap,
        gap_threshold=GAP_MIN,
        seq_rate=seq_rate,
        seq_corroborated=seq_corrob,
        seq_source_rate=monitor.source_tail_rate,
        onset_step=stream.onset_step,
    )


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
def _trigger(policy: str, signals: StreamSignals, t: int) -> bool:
    if policy == "no_action":
        return False
    if policy == "drift_only":
        return bool(signals.drift[t] > signals.drift_threshold)
    if policy == "threshold":
        return bool(signals.gap[t] > signals.gap_threshold)
    if policy == "corroborated":
        return bool(signals.seq_corroborated[t])
    if policy == "abstain_route":
        return bool(signals.seq_corroborated[t])
    raise ValueError(f"unknown policy {policy!r}")


def run_stream(
    scenario: ShiftScenario,
    stream: DeploymentStream,
    policy: str,
    signals: StreamSignals,
    *,
    audit_size: int = AUDIT_SIZE,
    cooldown: int = COOLDOWN,
    audit_budget: int = AUDIT_BUDGET,
    label_delay: int = 0,
    seed: int = 0,
) -> PolicyOutcome:
    """Run one policy on one stream; the controller owns every transition.

    ``label_delay`` (`derived here`): with the default ``0`` the audit
    label is available the instant it is requested, reproducing the original
    synchronous behavior exactly. With ``label_delay > 0`` the audited sample
    is routed through :class:`~label_free_monitoring.audit.DelayedLabelQueue` and the
    candidate-train/promote/rollback decision is deferred until the label is
    actually released ``label_delay`` steps later — this is what makes an
    immediate-vs-delayed decision comparison measurable instead of assumed.
    """
    from .audit import DelayedLabelQueue

    budget_labels = audit_budget if audit_budget > 4 else audit_budget * audit_size
    controller = Controller(
        cooldown=cooldown, audit_budget=budget_labels, min_validation_gain=PROMOTION_MIN_GAIN
    )
    original = scenario.model
    champion = original
    source_risk = scenario.source_risk
    audit_x: list[np.ndarray] = []
    audit_y: list[np.ndarray] = []
    cost = 0.0
    retrains = 0
    audits = 0
    routed = 0
    false_retrain = False
    first_post_onset: int | None = None
    last_candidate: LogisticRegression | None = None
    timeline: list[dict[str, Any]] = []
    recent_risk: list[float] = []
    label_queue = DelayedLabelQueue() if label_delay > 0 else None

    def _resolve_audit(now: int, ax: np.ndarray, ay: np.ndarray) -> str:
        nonlocal champion, last_candidate, retrains, false_retrain, first_post_onset, cost
        audit_risk = float(np.mean(_brier(champion.predict_proba(ax)[:, 1], ay)))
        harm_confirmed = audit_risk > source_risk + HARM_TOL
        controller.complete_audit(labels=len(ax), harm_confirmed=harm_confirmed)
        if harm_confirmed:
            audit_x.append(ax)
            audit_y.append(ay)
            X = np.vstack(audit_x)
            Y = np.concatenate(audit_y)
            controller.train_candidate(window=f"recent_{audit_size}")
            cost += C_COMPUTE
            cut = int(0.7 * len(X))
            candidate = LogisticRegression(C=1.0, max_iter=2000)
            candidate.fit(X[:cut], Y[:cut])
            gain = float(
                np.mean(_brier(champion.predict_proba(X[cut:])[:, 1], Y[cut:]))
                - np.mean(_brier(candidate.predict_proba(X[cut:])[:, 1], Y[cut:]))
            )
            if gain >= PROMOTION_MIN_GAIN:
                controller.promote_candidate(validation_gain=gain)
                champion = candidate
                last_candidate = candidate
                retrains += 1
                resolved_event = "promote"
            else:
                controller.rollback(reason="validation_gate")
                resolved_event = "rollback"
        else:
            resolved_event = "audit_clear"
        if harm_confirmed and now >= stream.onset_step and first_post_onset is None:
            first_post_onset = now
        return resolved_event

    for t in range(stream.n_steps):
        controller.advance_time(1)
        x_t, y_t, _, _ = stream.batch(t)
        p_cur = champion.predict_proba(x_t)[:, 1]
        risk_cur = float(np.mean(_brier(p_cur, y_t)))
        p_base = original.predict_proba(x_t)[:, 1]
        risk_base = float(np.mean(_brier(p_base, y_t)))
        risk_cand = (
            float(np.mean(_brier(last_candidate.predict_proba(x_t)[:, 1], y_t)))
            if last_candidate is not None
            else float("nan")
        )
        recent_risk.append(risk_cur)

        event = ""
        if label_queue is not None:
            released = label_queue.release(now=t)
            if released:
                _, (ax, ay) = released[0]
                event = _resolve_audit(t, ax, ay)

        if _trigger(policy, signals, t) and controller.can_alarm():
            if policy == "abstain_route":
                routed += 1
                cost += C_ROUTE
                event = "route"
            else:
                try:
                    controller.raise_alarm(policy)
                    audits += 1
                    ax, ay = stream.recent(t, audit_size)
                    cost += C_LABEL * len(ax)
                    risk_base_at_alarm = risk_base
                    if risk_base_at_alarm <= source_risk + HARM_TOL:
                        false_retrain = True
                    if label_queue is not None:
                        label_queue.submit(f"audit-{t}", (ax, ay), available_at=t + label_delay, submitted_at=t)
                        event = "audit_pending"
                    else:
                        event = _resolve_audit(t, ax, ay)
                except IllegalTransition:
                    event = "blocked"

        cost += C_RISK * max(0.0, risk_cur - source_risk)
        timeline.append(
            {
                "policy": policy,
                "time": t,
                "state": controller.state,
                "event": event,
                "latent_risk": round(risk_cur, 6),
                "champion_risk": round(risk_base, 6),
                "candidate_risk": round(risk_cand, 6) if np.isfinite(risk_cand) else np.nan,
                "cost": round(cost, 6),
                "seed": stream.seed,
                "run_id": stream.run_id,
            }
        )

    tail_risk = float(np.mean(recent_risk[-20:]))
    recovered = bool(tail_risk <= source_risk + RECOVERY_TOL)
    harmful = scenario.regime in HARMFUL_REGIMES
    missed = bool(harmful and first_post_onset is None)
    delay = float(first_post_onset - stream.onset_step) if first_post_onset is not None else float("nan")
    return PolicyOutcome(
        policy=policy,
        seed=stream.seed,
        run_id=stream.run_id,
        false_retrain=bool(false_retrain),
        missed=missed,
        detection_delay=delay,
        recovered=recovered,
        cost=float(cost),
        retrains=retrains,
        audits=audits,
        routed=routed,
        events=len(controller.events),
        final_risk=tail_risk,
        timeline=timeline,
    )


def aggregate(policy: str, outcomes: list[PolicyOutcome]) -> dict[str, Any]:
    """Per-policy metric row with every decision quantity kept separate.

    ``miss_rate`` is the fraction of streams where a harmful regime deteriorated
    without any confirmed intervention (benign streams cannot be missed).
    """
    n = len(outcomes)
    delays = [o.detection_delay for o in outcomes if np.isfinite(o.detection_delay)]
    return {
        "policy": policy,
        "false_retrain_rate": float(np.mean([o.false_retrain for o in outcomes])),
        "miss_rate": float(np.mean([o.missed for o in outcomes])),
        "median_detection_delay": float(np.median(delays)) if delays else float("nan"),
        "recovery_rate": float(np.mean([o.recovered for o in outcomes])),
        "total_cost": float(np.mean([o.cost for o in outcomes])),
        "retrain_count": float(np.mean([o.retrains for o in outcomes])),
        "audit_count": float(np.mean([o.audits for o in outcomes])),
        "route_count": float(np.mean([o.routed for o in outcomes])),
        "mean_final_risk": float(np.mean([o.final_risk for o in outcomes])),
        "audit_budget": float(AUDIT_BUDGET),
        "streams": n,
    }


def compare_policies(
    data: ShiftScenario,
    policies: tuple[str, ...] | list[str] = POLICY_NAMES,
    seeds: range | list[int] = range(8),
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Compare policies on paired seeds: one stream per seed, shared by all policies.

    ``data`` supplies the regime and sample sizes; each seed in ``seeds``
    rebuilds a scenario and a deployment stream, and every policy sees exactly
    the same stream.  Latent truth is used only for evaluation.
    """
    from .synthetic import make_shift_scenario

    scenarios: dict[int, tuple[ShiftScenario, DeploymentStream, StreamSignals]] = {}
    for s in seeds:
        scenario = make_shift_scenario(
            seed=int(s), regime=data.regime, n_source=data.n_source, n_target=data.n_target
        )
        stream = scenario.sample_deployment(seed=int(s))
        signals = compute_signals(scenario, stream, seed=int(s))
        scenarios[int(s)] = (scenario, stream, signals)

    report: dict[str, dict[str, Any]] = {}
    for name in policies:
        outcomes = []
        for s in seeds:
            scenario, stream, signals = scenarios[int(s)]
            outcomes.append(run_stream(scenario, stream, name, signals, seed=int(s), **kwargs))
        row = aggregate(name, outcomes)
        row["outcomes"] = outcomes
        report[name] = row
    return report


def policy_timeline_rows(report: dict[str, dict[str, Any]], policies: list[str]) -> list[dict[str, Any]]:
    """Flatten selected policies' timelines for Fig A4."""
    rows: list[dict[str, Any]] = []
    for name in policies:
        for outcome in report[name]["outcomes"]:
            rows.extend(outcome.timeline)
    return rows


def policy_metric_rows(report: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-policy-per-seed metric rows for Fig A5."""
    rows: list[dict[str, Any]] = []
    for name, row in report.items():
        for outcome in row["outcomes"]:
            rows.append(
                {
                    "policy": name,
                    "false_retrain_rate": float(outcome.false_retrain),
                    "miss_rate": float(outcome.missed),
                    "median_detection_delay": outcome.detection_delay,
                    "recovery_rate": float(outcome.recovered),
                    "total_cost": outcome.cost,
                    "seed": outcome.seed,
                    "run_id": outcome.run_id,
                }
            )
    return rows
