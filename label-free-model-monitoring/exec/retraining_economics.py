"""Retraining economics: finite-horizon optimizer, conditional count bound, queue.

Everything here is `derived here`: no selected paper defines a cost objective, a
finite intervention capacity, or a training/deployment latency.  The optimizer
is exact (O(T^2) dynamic program) and reproduces the transparent exhaustive
oracle's schedule *and* its lexicographic tie-breaking.  The Regol-style count
ceiling ``r* <= T - sqrt(alpha / L)`` is only meaningful when the uniform
adjacent-model gap ``L`` has been checked on the loss table; callers must pass
``observed_L <= L`` or drop the claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# exact optimizer
# --------------------------------------------------------------------------- #
def _cost_of(losses: np.ndarray, times: Sequence[int], retrain_cost: float) -> float:
    """Cumulative loss plus retraining charges for one schedule."""
    model = 0
    total = 0.0
    chosen = set(times)
    for t in range(losses.shape[1]):
        if t in chosen:
            model = t
        total += float(losses[model, t])
    return total + retrain_cost * len(times)


def optimal_retraining_schedule(
    loss_by_model_and_time: Any, retrain_cost: float
) -> tuple[tuple[int, ...], float]:
    """Exact cost-optimal retraining schedule over a finite horizon.

    A retrain at time ``t`` serves model-epoch row ``t`` from ``t`` onward and
    costs ``retrain_cost``; a retrain at time 0 is not a decision.  Ties are
    broken toward the lexicographically smallest schedule, matching the
    exhaustive oracle used to validate this implementation.
    """
    losses = np.asarray(loss_by_model_and_time, dtype=float)
    if losses.ndim != 2:
        raise ValueError("loss_by_model_and_time must be a 2-D array")
    horizon = losses.shape[1]
    if losses.shape[0] < horizon:
        raise ValueError("loss matrix needs at least `horizon` model epochs (rows)")
    if horizon == 0:
        return (), 0.0

    tol = 1e-12
    best: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}

    def better(cand: tuple[float, tuple[int, ...]], incumbent: tuple[float, tuple[int, ...]] | None) -> bool:
        if incumbent is None:
            return True
        if cand[0] < incumbent[0] - tol:
            return True
        if abs(cand[0] - incumbent[0]) <= tol and cand[1] < incumbent[1]:
            return True
        return False

    for t in range(horizon, -1, -1):
        for model in range(horizon):
            if t == horizon:
                best[(t, model)] = (0.0, ())
                continue
            keep_cost, keep_times = best[(t + 1, model)]
            candidate = (float(losses[model, t]) + keep_cost, keep_times)
            if t >= 1 and model != t:
                next_cost, next_times = best[(t + 1, t)]
                retrain = (retrain_cost + float(losses[t, t]) + next_cost, (t,) + next_times)
                if better(retrain, candidate):
                    candidate = retrain
            best[(t, model)] = candidate

    cost, times = best[(0, 0)]
    return tuple(times), float(cost)


def brute_force_schedule(loss_by_model_and_time: Any, retrain_cost: float) -> tuple[tuple[int, ...], float]:
    """Transparent exhaustive oracle used to validate the optimizer."""
    losses = np.asarray(loss_by_model_and_time, dtype=float)
    horizon = losses.shape[1]
    choices = []
    for count in range(horizon):
        for times in combinations(range(1, horizon), count):
            choices.append((_cost_of(losses, times, retrain_cost), times))
    return min(choices)[1], min(choices)[0]


def observed_adjacent_gap_max(loss_by_model_and_time: Any) -> float:
    """Largest absolute adjacent-model loss gap over all evaluation times."""
    losses = np.asarray(loss_by_model_and_time, dtype=float)
    if losses.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(losses, axis=0)).max(initial=0.0))


def regol_count_bound(horizon: int, alpha: float, L: float) -> float:
    """Regol-style ceiling ``T - sqrt(alpha / L)`` for a checked uniform ``L``."""
    if L <= 0:
        raise ValueError("L must be positive for the bound to be defined")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return float(horizon - np.sqrt(alpha / L))


# --------------------------------------------------------------------------- #
# finite-capacity retraining queue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueueEvent:
    kind: str
    time: int
    epoch: int
    detail: str = ""


@dataclass
class QueueRun:
    """Observable outcome of a finite-budget retraining queue."""

    events: list[QueueEvent]
    horizon: int
    retrain_budget: int
    completed_retrains: int
    requested_retrains: int
    rejected_requests: int
    _epoch_timeline: list[int] = field(default_factory=list)

    def model_epoch_at(self, time: int) -> int:
        if not self._epoch_timeline:
            return 0
        t = min(max(int(time), 0), len(self._epoch_timeline) - 1)
        return int(self._epoch_timeline[t])

    def completed_by_time(self, time: int) -> int:
        return sum(1 for e in self.events if e.kind == "deployed" and e.time <= time)

    @property
    def stranded_budget(self) -> int:
        return int(self.retrain_budget - self.completed_retrains)

    @property
    def in_flight_at_horizon(self) -> int:
        """Accepted requests still training/deploying when the horizon ends.

        A slot spent on work that never finished is not "unrequested" and it
        is not "completed": it is capacity that was genuinely used but is
        stranded by the clock, not by nobody asking for it.
        """
        deployed_epochs = {e.epoch for e in self.events if e.kind == "deployed"}
        return sum(
            1
            for e in self.events
            if e.kind == "request_accepted" and (e.epoch + 1) not in deployed_epochs
        )

    @property
    def unrequested_capacity(self) -> int:
        """Budget that was never consumed by any accepted request at all."""
        return max(0, self.retrain_budget - self.completed_retrains - self.in_flight_at_horizon)

    @property
    def busy_rejected_requests(self) -> int:
        return sum(1 for e in self.events if e.kind == "request_rejected" and e.detail == "busy")

    @property
    def budget_rejected_requests(self) -> int:
        return sum(1 for e in self.events if e.kind == "request_rejected" and e.detail == "budget")

    def events_at(self, time: int) -> list[QueueEvent]:
        return [e for e in self.events if e.time == time]

    @property
    def event_kinds(self) -> tuple[str, ...]:
        return tuple(e.kind for e in self.events)


def simulate_retraining_queue(
    request_times: Iterable[int],
    horizon: int,
    retrain_budget: int,
    train_latency: int,
    deploy_latency: int,
) -> QueueRun:
    """Queue retraining requests under training/deployment latency and budget.

    The serving model stays in place until training *and* deployment finish.  A
    request while a job is in flight is rejected without changing service or
    consuming a slot.  Work that cannot finish inside the horizon strands its
    nominal budget.  Negative latencies raise ``ValueError``.
    """
    if train_latency < 0 or deploy_latency < 0:
        raise ValueError("latency must be non-negative")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if retrain_budget < 0:
        raise ValueError("retrain_budget must be non-negative")

    timeline = [0] * horizon
    events: list[QueueEvent] = []
    epoch = 0
    completed = 0
    requested = 0
    rejected = 0
    busy_until: int | None = None

    for t in sorted({int(x) for x in request_times}):
        if t < 0 or t >= horizon:
            continue
        requested += 1
        if busy_until is not None and t >= busy_until:
            busy_until = None
        if busy_until is not None:
            rejected += 1
            events.append(QueueEvent("request_rejected", t, epoch, "busy"))
            continue
        if completed >= retrain_budget:
            rejected += 1
            events.append(QueueEvent("request_rejected", t, epoch, "budget"))
            continue
        events.append(QueueEvent("request_accepted", t, epoch))
        train_done = t + train_latency
        deploy_at = train_done + deploy_latency
        busy_until = deploy_at
        if train_done < horizon:
            events.append(QueueEvent("training_complete", train_done, epoch))
        if deploy_at < horizon:
            events.append(QueueEvent("deployed", deploy_at, epoch + 1))
            for step in range(deploy_at, horizon):
                timeline[step] = epoch + 1
            epoch += 1
            completed += 1

    _EVENT_ORDER = {"request_accepted": 0, "training_complete": 1, "deployed": 2, "request_rejected": 3}
    events.sort(key=lambda e: (e.time, _EVENT_ORDER.get(e.kind, 99)))

    return QueueRun(
        events=events,
        horizon=horizon,
        retrain_budget=retrain_budget,
        completed_retrains=completed,
        requested_retrains=requested,
        rejected_requests=rejected,
        _epoch_timeline=timeline,
    )


# --------------------------------------------------------------------------- #
# loss tables and the policy sweep behind Fig A7
# --------------------------------------------------------------------------- #
def uniform_L_loss_table(horizon: int, alpha: float, L: float, base_high: float = 0.20) -> np.ndarray:
    """A loss table whose adjacent-model gap is bounded by ``L`` everywhere.

    Later epochs are *worse* here (``alpha`` and ``L`` set the scale), which is
    the configuration in which the conditional ceiling is informative.
    """
    base = np.linspace(base_high, max(base_high - 0.06, 0.0), horizon)
    return np.vstack([base + epoch * (L / 5.0) for epoch in range(horizon)])


def improving_loss_table(horizon: int, stale_loss: float, fresh_loss: float) -> np.ndarray:
    """A loss table where a model retrained after the shift beats the stale one.

    Row ``e`` is the performance of model epoch ``e`` at each evaluation time.
    Epochs trained before the onset (``e < T/3``) never learn the shift, so the
    initial model is stale everywhere; an epoch trained at or after the onset
    delivers the fresh loss from its own training time onward.
    """
    onset = max(1, horizon // 3)
    rows = []
    for epoch in range(horizon):
        row = np.full(horizon, stale_loss)
        if epoch >= onset:
            row[epoch:] = fresh_loss
        rows.append(row)
    return np.vstack(rows)


def stepwise_improvement_table(
    horizon: int,
    stale_loss: float,
    fresh_loss: float,
    epoch_gain: float,
    onset: int | None = None,
) -> np.ndarray:
    """An improving table where every later epoch is slightly better still.

    Provides a non-trivial multi-retrain optimum for the count-versus-ceiling
    panel while keeping adjacent-model gaps bounded by the stale/fresh jump.
    """
    onset = max(1, horizon // 3) if onset is None else onset
    rows = []
    for epoch in range(horizon):
        row = np.full(horizon, stale_loss)
        if epoch >= onset:
            gain = epoch_gain * min(epoch - onset, horizon - 1)
            row[epoch:] = max(fresh_loss - gain, 0.0)
        rows.append(row)
    return np.vstack(rows)


@dataclass
class SweepRow:
    policy: str
    drift_regime: str
    update_mode: str
    train_latency: int
    deploy_latency: int
    retrain_budget: int
    requested_retrains: int
    completed_retrains: int
    stranded_budget: int
    in_flight_at_horizon: int
    unrequested_capacity: int
    busy_rejected_requests: int
    budget_rejected_requests: int
    post_drift_risk: float
    seed: int
    run_id: str
    provenance: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _drift_risk_profile(regime: str, horizon: int, seed: int, base: float, peak: float) -> np.ndarray:
    """Latent risk of the *original* model at each step under a drift regime."""
    rng = np.random.default_rng(1000 + seed + len(regime))
    t = np.arange(horizon)
    if regime == "abrupt":
        profile = np.where(t >= horizon // 3, peak, base)
    elif regime == "gradual":
        start = horizon // 4
        ramp = np.clip((t - start) / max(1, horizon - start - 1), 0.0, 1.0)
        profile = base + (peak - base) * ramp
    elif regime == "recurring":
        period = max(3, horizon // 5)
        active = ((t // period) % 2 == 1)
        profile = np.where(active & (t >= horizon // 4), peak, base)
    else:
        raise ValueError(f"unknown drift regime {regime!r}")
    return profile + rng.normal(0.0, 0.004, size=horizon)


def _policy_request_times(
    policy: str,
    drift: np.ndarray,
    horizon: int,
    retrain_charge: float,
    update_mode: str,
    seed: int,
) -> list[int]:
    """Observable policy decision rule (no latent labels involved)."""
    if policy == "no_action":
        return []
    if policy == "threshold":
        baseline = float(np.median(drift[: horizon // 4]))
        return [t for t in range(1, horizon) if drift[t] > baseline + 0.05]
    if policy == "periodic":
        period = max(2, horizon // 6)
        return list(range(period, horizon, period))
    if policy == "cost_aware":
        onset = max(1, horizon // 3)
        fresh = 0.10
        rows = []
        for epoch in range(horizon):
            row = drift.copy()
            if epoch >= onset:
                row[epoch:] = fresh
            rows.append(row)
        times, _ = optimal_retraining_schedule(np.vstack(rows), retrain_cost=retrain_charge)
        return list(times)
    raise ValueError(f"unknown policy {policy!r}")


def sweep_policy_latency_budget(
    policies: Sequence[str],
    drift_regimes: Sequence[str],
    update_modes: Sequence[str],
    latency_settings: Sequence[tuple[int, int]],
    budgets: Sequence[int],
    seeds: Sequence[int] = (0, 1),
    horizon: int = 24,
    retrain_charge: float = 0.05,
    run_id: str = "t5-r2",
) -> list[SweepRow]:
    """Paired sweep of policy x drift x update-mode x latency x budget.

    Each configuration is run for every seed with the same latent risk profile,
    so differences inside a seed are attributable to the decision rule or the
    queue, not to different streams.  Risk after the drift onset is the mean
    latent risk served by the model epoch in service at each step.
    """
    rows: list[SweepRow] = []
    fresh_risk = 0.10
    for regime in drift_regimes:
        for mode in update_modes:
            for seed in seeds:
                drift = _drift_risk_profile(regime, horizon, int(seed), base=0.10, peak=0.34)
                onset = horizon // 3
                active = np.zeros(horizon, dtype=bool)
                if regime == "recurring":
                    period = max(3, horizon // 5)
                    active = ((np.arange(horizon) // period) % 2 == 1) & (np.arange(horizon) >= horizon // 4)

                def active_block_start(t: int) -> int:
                    if not active[t]:
                        return -1
                    start = t
                    while start > 0 and active[start - 1]:
                        start -= 1
                    return start

                for policy in policies:
                    requests = _policy_request_times(policy, drift, horizon, retrain_charge, mode, int(seed))
                    for train_latency, deploy_latency in latency_settings:
                        for budget in budgets:
                            run = simulate_retraining_queue(
                                request_times=requests,
                                horizon=horizon,
                                retrain_budget=budget,
                                train_latency=train_latency,
                                deploy_latency=deploy_latency,
                            )
                            deploy_by_epoch = {
                                event.epoch: event.time
                                for event in run.events
                                if event.kind == "deployed"
                            }
                            served = np.empty(horizon)
                            for t in range(horizon):
                                epoch = run.model_epoch_at(t)
                                known_at = deploy_by_epoch.get(epoch)
                                if known_at is None or known_at < onset:
                                    served[t] = drift[t]
                                elif regime == "recurring" and mode == "static" and active_block_start(t) > known_at + 1:
                                    served[t] = drift[t]
                                else:
                                    served[t] = min(fresh_risk, drift[t])
                            post = served[onset:]
                            rows.append(
                                SweepRow(
                                    policy=policy,
                                    drift_regime=regime,
                                    update_mode=mode,
                                    train_latency=int(train_latency),
                                    deploy_latency=int(deploy_latency),
                                    retrain_budget=int(budget),
                                    requested_retrains=len(requests),
                                    completed_retrains=run.completed_retrains,
                                    stranded_budget=run.stranded_budget,
                                    in_flight_at_horizon=run.in_flight_at_horizon,
                                    unrequested_capacity=run.unrequested_capacity,
                                    busy_rejected_requests=run.busy_rejected_requests,
                                    budget_rejected_requests=run.budget_rejected_requests,
                                    post_drift_risk=float(post.mean()),
                                    seed=int(seed),
                                    run_id=run_id,
                                    provenance=(
                                        f"seed={seed}|run_id={run_id}|policy={policy}|"
                                        f"latency={train_latency}+{deploy_latency}|budget={budget}"
                                    ),
                                )
                            )
    return rows
