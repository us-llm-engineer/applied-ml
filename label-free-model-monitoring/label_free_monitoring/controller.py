"""Retraining lifecycle controller.

The three source papers end at the alarm: P2 raises a time-uniform alarm and
explicitly leaves the intervention unspecified, P3 exposes the failure regime,
and P1 produces a batch gap estimate with no control loop.  The state machine
below is therefore `derived here`: stable -> audit -> candidate -> promote or
rollback -> cooldown, with an audit-label budget, a promotion gate, and a
rollback rule.  Every legal transition appends exactly one event; every illegal
event raises :class:`IllegalTransition` and leaves the full state untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STABLE = "stable"
AUDIT = "audit"
CANDIDATE = "candidate"
COOLDOWN = "cooldown"

_LEGAL_FROM = {
    "raise_alarm": {STABLE},
    "complete_audit": {AUDIT},
    "train_candidate": {CANDIDATE},
    "promote_candidate": {CANDIDATE},
    "rollback": {CANDIDATE, COOLDOWN},
}


class IllegalTransition(RuntimeError):
    """Raised when a controller event is not legal in the current state."""


@dataclass(frozen=True)
class Event:
    """One accepted lifecycle transition."""

    event: str
    time: int
    detail: dict[str, Any] = field(default_factory=dict)


class Controller:
    """Retraining controller with cooldown, audit budget and promotion gate.

    Parameters
    ----------
    cooldown : int
        Number of time steps the controller stays in ``cooldown`` after a
        promotion before a new alarm is accepted.
    audit_budget : int
        Total audit-label units available to the deployment campaign.
    min_validation_gain : float
        Minimum held-out improvement required to promote a candidate.
    """

    def __init__(self, cooldown: int = 3, audit_budget: int = 20, min_validation_gain: float = 0.0):
        if cooldown < 0:
            raise ValueError("cooldown must be non-negative")
        if audit_budget < 0:
            raise ValueError("audit_budget must be non-negative")
        self.cooldown = int(cooldown)
        self.audit_budget = int(audit_budget)
        self.min_validation_gain = float(min_validation_gain)
        self.state = STABLE
        self.time = 0
        self.audit_labels_used = 0
        self.retrains = 0
        self.candidates_trained = 0
        self.rollbacks = 0
        self._cooldown_until: int | None = None
        self._alarm_reason: str | None = None
        self._candidate_window: str | None = None
        self.events: list[Event] = []

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict[str, Any]:
        """Immutable view used to assert that rejected events change nothing."""
        return {
            "state": self.state,
            "time": self.time,
            "audit_labels_used": self.audit_labels_used,
            "audit_budget_remaining": self.audit_budget - self.audit_labels_used,
            "retrains": self.retrains,
            "candidates_trained": self.candidates_trained,
            "rollbacks": self.rollbacks,
            "cooldown_until": self._cooldown_until,
            "alarm_reason": self._alarm_reason,
            "candidate_window": self._candidate_window,
            "events": tuple((e.event, e.time) for e in self.events),
        }

    def _require(self, action: str) -> None:
        if self.state not in _LEGAL_FROM[action]:
            raise IllegalTransition(
                f"{action!r} is not legal in state {self.state!r} (legal from {sorted(_LEGAL_FROM[action])})"
            )

    def _log(self, event: str, **detail: Any) -> None:
        self.events.append(Event(event=event, time=self.time, detail=dict(detail)))

    # -------------------------------------------------------------- lifecycle
    def raise_alarm(self, reason: str) -> None:
        """Enter the audit state; rejected during cooldown."""
        self._require("raise_alarm")
        self.state = AUDIT
        self._alarm_reason = reason
        self._log("alarm", reason=reason)

    def complete_audit(self, labels: int, harm_confirmed: bool = True) -> None:
        """Close the audit and either move to candidate training or back to stable.

        ``labels`` is charged against the audit budget.  With
        ``harm_confirmed=False`` (the alarm was a false alarm on audited
        ground truth) the controller returns to ``stable`` and logs one event.
        """
        self._require("complete_audit")
        if labels < 0:
            raise ValueError("labels must be non-negative")
        if self.audit_labels_used + labels > self.audit_budget:
            raise IllegalTransition(
                f"audit budget exhausted: {self.audit_labels_used}+{labels} > {self.audit_budget}"
            )
        self.audit_labels_used += labels
        if harm_confirmed:
            self.state = CANDIDATE
            self._log("audit_complete", labels=labels, harm_confirmed=True)
        else:
            self.state = STABLE
            self._alarm_reason = None
            self._log("audit_complete", labels=labels, harm_confirmed=False)

    def train_candidate(self, window: str) -> None:
        """Record the candidate training window; stays in ``candidate``."""
        self._require("train_candidate")
        self._candidate_window = window
        self.candidates_trained += 1
        self._log("candidate_trained", window=window)

    def promote_candidate(self, validation_gain: float) -> None:
        """Promotion gate; a non-positive (or below-threshold) gain is rejected."""
        self._require("promote_candidate")
        if validation_gain < self.min_validation_gain:
            raise IllegalTransition(
                f"promotion gate failed: validation_gain={validation_gain:.4f} < min={self.min_validation_gain:.4f}"
            )
        self.state = COOLDOWN
        self.retrains += 1
        self._cooldown_until = self.time + self.cooldown
        self._alarm_reason = None
        self._log("promote", validation_gain=validation_gain, cooldown_until=self._cooldown_until)

    def rollback(self, reason: str = "unspecified") -> None:
        """Discard the candidate and return to stable (from candidate or cooldown)."""
        self._require("rollback")
        self.state = STABLE
        self.rollbacks += 1
        self._alarm_reason = None
        self._cooldown_until = None
        self._log("rollback", reason=reason)

    def advance_time(self, steps: int = 1) -> None:
        """Advance the clock; cooldown expiry is the only implicit transition."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.time += int(steps)
        if (
            self.state == COOLDOWN
            and self._cooldown_until is not None
            and self.time >= self._cooldown_until
        ):
            self.state = STABLE
            self._cooldown_until = None
            self._log("cooldown_complete")

    # ----------------------------------------------------------------- helpers
    @property
    def budget_remaining(self) -> int:
        return self.audit_budget - self.audit_labels_used

    def can_alarm(self) -> bool:
        return self.state == STABLE and self.budget_remaining > 0

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Controller(state={self.state!r}, time={self.time}, "
            f"retrains={self.retrains}, labels_used={self.audit_labels_used}/{self.audit_budget})"
        )
