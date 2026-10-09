"""Event-time delayed audit labels (`derived here`).

None of the three source papers specifies *when* an audited label becomes
usable: P2 raises an alarm and stops, P3 leaves the intervention open, and P1
produces a batch estimate.  The queue below makes the information boundary
executable: a label submitted at ``submitted_at`` with ``available_at`` stays
hidden until ``now >= available_at``, is released exactly once, and a duplicate
sample id is rejected atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class AuditRecord:
    """One audited label and its event times."""

    sample_id: str
    label: Any
    submitted_at: int
    available_at: int
    released_at: int | None = None

    @property
    def pending(self) -> bool:
        return self.released_at is None


class DelayedLabelQueue:
    """Hold audited labels until their declared availability time.

    ``submit`` rejects a repeated sample id (including one already released)
    with ``ValueError`` and changes nothing.  ``release(now)`` returns the
    ``(sample_id, label)`` pairs whose ``available_at <= now`` exactly once,
    in submission order.
    """

    def __init__(self) -> None:
        self._pending: dict[str, AuditRecord] = {}
        self._released: list[AuditRecord] = []
        self._seen: set[str] = set()

    # ------------------------------------------------------------------ writes
    def submit(self, sample_id: str, label: Any, available_at: int, submitted_at: int = 0) -> AuditRecord:
        if sample_id in self._seen:
            raise ValueError(f"duplicate sample id {sample_id!r} rejected")
        available_at = int(available_at)
        if available_at < 0:
            raise ValueError("available_at must be non-negative")
        record = AuditRecord(
            sample_id=str(sample_id),
            label=label,
            submitted_at=int(submitted_at),
            available_at=available_at,
        )
        self._pending[record.sample_id] = record
        self._seen.add(record.sample_id)
        return record

    def release(self, now: int) -> list[tuple[str, Any]]:
        """Return labels due at ``now`` and remove them from the queue."""
        now = int(now)
        due = sorted(
            (r for r in self._pending.values() if r.available_at <= now),
            key=lambda r: (r.submitted_at, r.sample_id),
        )
        released: list[tuple[str, Any]] = []
        for record in due:
            del self._pending[record.sample_id]
            record.released_at = now
            self._released.append(record)
            released.append((record.sample_id, record.label))
        return released

    # ------------------------------------------------------------------ reads
    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def released_count(self) -> int:
        return len(self._released)

    def pending_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._released)

    def released_labels(self) -> tuple[Any, ...]:
        return tuple(record.label for record in self._released)


def simulate_delayed_audit(
    arrivals: Iterable[tuple[int, str, Any]],
    horizon: int,
    delay: int,
    start_time: int = 0,
) -> list[dict[str, Any]]:
    """Release-audit timeline for a stream of audited arrivals.

    ``arrivals`` is ``(arrival_step, sample_id, label)``; each label is submitted
    at its arrival step and becomes available ``delay`` steps later.  Returns one
    row per step with queue occupancy and the running audited-label mean.
    """
    queue = DelayedLabelQueue()
    arrivals = sorted(arrivals, key=lambda item: item[0])
    pos = 0
    rows: list[dict[str, Any]] = []
    seen_labels: list[float] = []
    for step in range(horizon):
        while pos < len(arrivals) and arrivals[pos][0] == step:
            _, sample_id, label = arrivals[pos]
            queue.submit(sample_id, label, available_at=step + delay, submitted_at=step)
            pos += 1
        released = queue.release(now=step)
        seen_labels.extend(float(value) for _, value in released)
        rows.append(
            {
                "step": step,
                "time": start_time + step,
                "submitted_this_step": sum(1 for a in arrivals if a[0] == step),
                "released_this_step": len(released),
                "released_sample_ids": "|".join(sample_id for sample_id, _ in released),
                "pending_labels": queue.pending_count,
                "total_released": queue.released_count,
                "audited_label_mean": float(sum(seen_labels) / len(seen_labels)) if seen_labels else float("nan"),
            }
        )
    return rows
