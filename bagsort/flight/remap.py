"""Atomic re-mapping of many bags in a single commit.

A schedule change moves a whole group of bags to another flight. The batch is
validated as a whole while the store's write lock is held, then staged on one
session, so either every mapping lands in a single commit or none does. A
concurrent batch can neither observe the staged commands nor slip its own
commit into the middle of this one, and every rejection names the item that
caused it so the operator can fix the batch and resubmit it whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from bagsort.audit.chain import AuditChain
from bagsort.clock import Clock
from bagsort.errors import BatchRemapError, ValidationError
from bagsort.model.errors import unknown_remap_batch
from bagsort.model.state import State
from bagsort.model.validate import is_open_state
from bagsort.store.store import Store


@dataclass(frozen=True)
class RemapItem:
    """One bag and the flight it should belong to."""

    bag_id: str
    flight_id: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RemapItem":
        return cls(
            bag_id=str(raw.get("bag_id", "")),
            flight_id=str(raw.get("flight_id", "")),
        )


@dataclass(frozen=True)
class BatchResult:
    """What a committed batch changed, line by line."""

    batch_id: str
    watermark: int
    items: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "count": len(self.items),
            "watermark": self.watermark,
            "items": [dict(item) for item in self.items],
        }


class BatchRemap:
    """Validates a whole batch, then commits it as one journal transaction."""

    def __init__(self, store: Store, audit: AuditChain, clock: Clock) -> None:
        self._store = store
        self._audit = audit
        self._clock = clock

    def apply(self, items: Iterable[RemapItem]) -> BatchResult:
        """Commit every item in one shot, or reject the batch and commit nothing."""

        items = list(items)
        self._check_shape(items)
        with self._store.session() as session:
            planned = self._plan(session.state, items)
            at = self._clock.now_iso()
            # The first journal sequence this batch will occupy; unique because
            # the write lock is held and the watermark only moves forward.
            batch_id = f"remap-{self._store.watermark + 1:06d}"
            moved: list[dict[str, Any]] = []
            for item, previous in planned:
                bag = session.state.bags[item.bag_id]
                updated = bag.evolved(flight_id=item.flight_id)
                session.state.bags[item.bag_id] = updated
                session.state.mappings[item.bag_id] = item.flight_id
                session.record(
                    "mapping.put",
                    {
                        "bag_id": item.bag_id,
                        "flight_id": item.flight_id,
                        "batch_id": batch_id,
                    },
                    at,
                )
                session.record("bag.put", {"bag": updated.as_dict()}, at)
                moved.append(
                    {
                        "bag_id": item.bag_id,
                        "from_flight_id": previous,
                        "to_flight_id": item.flight_id,
                    }
                )
            for line in moved:
                self._audit.stage(
                    session,
                    "remap",
                    str(line["bag_id"]),
                    "checkin",
                    {
                        "batch_id": batch_id,
                        "from_flight_id": line["from_flight_id"],
                        "to_flight_id": line["to_flight_id"],
                    },
                    at=at,
                )
        return BatchResult(batch_id, self._store.watermark, tuple(moved))

    def history(self, batch_id: str) -> dict[str, Any]:
        """Every journal and audit line one batch left behind, for review."""

        entries = [
            entry
            for entry in self._store.entries()
            if entry.kind == "mapping.put" and entry.payload.get("batch_id") == batch_id
        ]
        if not entries:
            raise unknown_remap_batch(batch_id)
        audits = {
            entry.subject: entry
            for entry in self._audit.entries()
            if entry.kind == "remap" and entry.detail.get("batch_id") == batch_id
        }
        mappings = self._store.state().mappings
        lines = []
        for entry in entries:
            bag_id = str(entry.payload.get("bag_id", ""))
            audit = audits.get(bag_id)
            lines.append(
                {
                    "sequence": entry.sequence,
                    "bag_id": bag_id,
                    "flight_id": str(entry.payload.get("flight_id", "")),
                    "from_flight_id": str(audit.detail.get("from_flight_id", ""))
                    if audit
                    else "",
                    "audit_sequence": audit.sequence if audit else 0,
                    "audit_digest": audit.digest if audit else "",
                    "current_flight_id": mappings.get(bag_id, ""),
                    "at": entry.at,
                }
            )
        return {"batch_id": batch_id, "count": len(lines), "items": lines}

    def _check_shape(self, items: list[RemapItem]) -> None:
        if not items:
            raise ValidationError("a remap batch must name at least one bag")
        seen: dict[str, int] = {}
        for index, item in enumerate(items):
            if not item.bag_id.strip() or not item.flight_id.strip():
                raise ValidationError(f"batch item {index} must name both a bag and a flight")
            first = seen.get(item.bag_id)
            if first is not None:
                raise ValidationError(
                    f"bag {item.bag_id!r} appears twice in the batch (items {first} and {index})"
                )
            seen[item.bag_id] = index

    def _plan(
        self, state: State, items: list[RemapItem]
    ) -> list[tuple[RemapItem, str]]:
        failures = []
        planned = []
        for index, item in enumerate(items):
            reason = self._problem(state, item)
            if reason:
                failures.append({"index": index, "bag_id": item.bag_id, "reason": reason})
            else:
                planned.append((item, state.mappings.get(item.bag_id, "")))
        if failures:
            raise BatchRemapError(failures)
        return planned

    @staticmethod
    def _problem(state: State, item: RemapItem) -> str:
        if item.bag_id not in state.bags:
            return f"bag {item.bag_id!r} is not registered"
        flight = state.flights.get(item.flight_id)
        if flight is None:
            return f"flight {item.flight_id!r} is not registered"
        if not is_open_state(flight.state):
            return f"flight {item.flight_id!r} is not accepting baggage"
        if state.mappings.get(item.bag_id) == item.flight_id:
            return f"bag {item.bag_id!r} already belongs to {item.flight_id!r}"
        return ""
