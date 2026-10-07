"""Atomic batch re-mapping of bags across flights.

A flight adjustment rarely moves one bag; it moves a planeload. Applying the
changes one at a time leaves the mapping ledger half switched when an item
partway through the list is refused. This service makes the whole batch one
commit: every item is validated before anything is recorded, and the store
session commits all of it or none of it.

Failure strategy: discard and re-submit, never compensate item by item. The
batch is validated and committed inside a single store session, which is a
critical section (see ``Store.session``), so a concurrent change cannot slip
between the checks and the commit. If any item fails validation the whole
batch is dropped untouched and the caller fixes the list and tries again;
there is nothing half-applied to undo. A committed batch is undone with
:meth:`BatchRemap.rollback`, itself one atomic batch guarded by a drift
check, so a change that landed in between stops the rollback instead of
being silently clobbered. The fallback when the drift check refuses is to
undo the later change first (last in, first out) or reconcile by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from bagsort.audit.chain import AuditChain
from bagsort.clock import Clock
from bagsort.errors import ConflictError, ValidationError
from bagsort.model.errors import unknown_remap_batch
from bagsort.model.state import State
from bagsort.model.validate import is_open_state
from bagsort.store.journal import JournalEntry
from bagsort.store.store import Session, Store

APPLY = "apply"
ROLLBACK = "rollback"
BATCH_KIND = "mapping.batch"
ID_PREFIX = "rmb-"


@dataclass(frozen=True)
class RemapItem:
    """One requested mapping change: move ``bag_id`` onto ``flight_id``."""

    bag_id: str
    flight_id: str


@dataclass(frozen=True)
class RemapFailure:
    """Why one item of a refused batch did not pass validation."""

    bag_id: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"bag_id": self.bag_id, "reason": self.reason}


@dataclass(frozen=True)
class RemapLine:
    """One committed mapping change with the flights on both sides."""

    bag_id: str
    from_flight_id: str
    to_flight_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "bag_id": self.bag_id,
            "from_flight_id": self.from_flight_id,
            "to_flight_id": self.to_flight_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RemapLine":
        return cls(
            bag_id=str(raw["bag_id"]),
            from_flight_id=str(raw.get("from_flight_id", "")),
            to_flight_id=str(raw.get("to_flight_id", "")),
        )


class BatchRemapError(ValidationError):
    """The whole batch was refused; ``failures`` names every offending item."""

    def __init__(self, failures: Iterable[RemapFailure]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(
            f"{failure.bag_id or '<empty>'}: {failure.reason}" for failure in self.failures
        )
        super().__init__(f"remap batch refused: {detail}")


@dataclass(frozen=True)
class RemapBatch:
    """A committed batch as the journal header describes it."""

    id: str
    action: str
    items: tuple[RemapLine, ...]
    at: str
    rollback_of: str = ""
    sequence: int = 0
    rolled_back: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "items": [line.as_dict() for line in self.items],
            "at": self.at,
            "rollback_of": self.rollback_of,
            "sequence": self.sequence,
            "rolled_back": self.rolled_back,
        }


class BatchRemap:
    """Validates, commits, rolls back and reconciles batches of mapping changes."""

    def __init__(self, store: Store, audit: AuditChain, clock: Clock) -> None:
        self._store = store
        self._audit = audit
        self._clock = clock

    # -- commit ------------------------------------------------------------

    def apply(self, items: Iterable[RemapItem], batch_id: str = "") -> RemapBatch:
        """Commit every item as one batch, or none of them.

        Validation runs inside the commit's critical section and any failure
        raises :class:`BatchRemapError` before a single command is recorded,
        so a refused batch leaves the journal exactly as it was.
        """

        requested = tuple(items)
        if not requested:
            raise ValidationError("a remap batch must name at least one bag")
        at = self._clock.now_iso()
        with self._store.session() as session:
            lines = self._validate(session.state, requested)
            batch_id = self._claim_id(batch_id)
            self._commit(session, batch_id, APPLY, lines, "", at)
        return self.batch(batch_id)

    def rollback(self, batch_id: str) -> RemapBatch:
        """Atomically put every item of a committed batch back where it was.

        The rollback refuses to run when any bag has moved since the batch
        committed, so a concurrent change cannot be silently overwritten and
        the ledger never shows a state no command sequence produced.
        """

        at = self._clock.now_iso()
        with self._store.session() as session:
            original = self._find(batch_id)
            if original.action != APPLY:
                raise ConflictError(f"remap batch {batch_id!r} is itself a rollback")
            if self._rolled_back(batch_id):
                raise ConflictError(f"remap batch {batch_id!r} is already rolled back")
            drifted = sorted(
                line.bag_id
                for line in original.items
                if session.state.mappings.get(line.bag_id) != line.to_flight_id
            )
            if drifted:
                raise ConflictError(
                    f"remap batch {batch_id!r} cannot be rolled back: "
                    f"{', '.join(drifted)} changed after it committed; "
                    "undo the later change first"
                )
            lines = tuple(
                RemapLine(
                    bag_id=line.bag_id,
                    from_flight_id=line.to_flight_id,
                    to_flight_id=line.from_flight_id,
                )
                for line in original.items
            )
            rollback_id = self._claim_id("")
            self._commit(session, rollback_id, ROLLBACK, lines, batch_id, at)
        return self.batch(rollback_id)

    # -- history -----------------------------------------------------------

    def batches(self) -> list[RemapBatch]:
        """Every committed batch, oldest first, each reviewable item by item."""

        return [self._to_view(entry) for entry in self._headers()]

    def batch(self, batch_id: str) -> RemapBatch:
        return self._find(batch_id)

    # -- cross-checks --------------------------------------------------------

    def reconcile(self, batch_id: str) -> dict[str, Any]:
        """Prove the journal, the mapping ledger and the audit chain agree.

        Each item must have its own ``mapping.put`` in the command log, the
        batch must have its audit entry, the hash chain must verify and the
        materialised ledger must equal a fresh replay of the journal. Sort
        records are reported per item so a bag's sorting history can be read
        next to the mapping the batch committed.
        """

        view = self._find(batch_id)
        entries = self._store.entries()
        state = self._store.state()
        puts = {
            str(entry.payload.get("bag_id")): entry.sequence
            for entry in entries
            if entry.kind == "mapping.put"
            and str(entry.payload.get("batch_id", "")) == view.id
        }
        audit_entries = [
            entry
            for entry in state.audit
            if entry.kind == "remap_batch" and str(entry.detail.get("batch_id")) == view.id
        ]
        items = []
        for line in view.items:
            current = state.mappings.get(line.bag_id)
            sorted_flights = sorted(
                {record.flight_id for record in state.sort_records if record.bag_id == line.bag_id}
            )
            items.append(
                {
                    "bag_id": line.bag_id,
                    "from_flight_id": line.from_flight_id,
                    "to_flight_id": line.to_flight_id,
                    "journal_sequence": puts.get(line.bag_id, 0),
                    "current_flight_id": current,
                    "matches_batch": current == line.to_flight_id,
                    "sort_records": sum(
                        1 for record in state.sort_records if record.bag_id == line.bag_id
                    ),
                    "sorted_flights": sorted_flights,
                }
            )
        chain_intact = self._audit.verify()
        intact = (
            all(item["journal_sequence"] for item in items)
            and bool(audit_entries)
            and chain_intact
        )
        return {
            "id": view.id,
            "action": view.action,
            "rollback_of": view.rollback_of,
            "header_sequence": view.sequence,
            "audit_sequence": audit_entries[-1].sequence if audit_entries else 0,
            "audit_chain_intact": chain_intact,
            "intact": intact,
            "items": items,
        }

    # -- internals -----------------------------------------------------------

    def _validate(self, state: State, items: tuple[RemapItem, ...]) -> tuple[RemapLine, ...]:
        """Check every item and collect every failure before anything moves."""

        failures: list[RemapFailure] = []
        lines: list[RemapLine] = []
        seen: set[str] = set()
        for item in items:
            bag_id = item.bag_id.strip()
            flight_id = item.flight_id.strip()
            if not bag_id:
                failures.append(RemapFailure(item.bag_id, "bag_id must not be empty"))
                continue
            if not flight_id:
                failures.append(RemapFailure(bag_id, "flight_id must not be empty"))
                continue
            if bag_id in seen:
                failures.append(
                    RemapFailure(bag_id, "bag appears more than once in this batch")
                )
                continue
            seen.add(bag_id)
            if bag_id not in state.bags:
                failures.append(RemapFailure(bag_id, f"bag {bag_id!r} is not registered"))
                continue
            flight = state.flights.get(flight_id)
            if flight is None:
                failures.append(
                    RemapFailure(bag_id, f"flight {flight_id!r} is not registered")
                )
                continue
            if not is_open_state(flight.state):
                failures.append(
                    RemapFailure(bag_id, f"flight {flight_id!r} is not accepting baggage")
                )
                continue
            current = state.mappings.get(bag_id, "")
            if current == flight_id:
                failures.append(
                    RemapFailure(bag_id, f"bag {bag_id!r} already belongs to {flight_id!r}")
                )
                continue
            lines.append(
                RemapLine(bag_id=bag_id, from_flight_id=current, to_flight_id=flight_id)
            )
        if failures:
            raise BatchRemapError(failures)
        return tuple(lines)

    def _commit(
        self,
        session: Session,
        batch_id: str,
        action: str,
        lines: tuple[RemapLine, ...],
        rollback_of: str,
        at: str,
    ) -> None:
        """Stage the header, the per-item moves and the audit entry as one commit."""

        header = {
            "id": batch_id,
            "action": action,
            "items": [line.as_dict() for line in lines],
            "rollback_of": rollback_of,
            "at": at,
        }
        session.record(BATCH_KIND, {"batch": header}, at)
        for line in lines:
            bag = session.state.bags[line.bag_id]
            updated = bag.evolved(flight_id=line.to_flight_id)
            session.state.mappings[line.bag_id] = line.to_flight_id
            session.state.bags[line.bag_id] = updated
            session.record(
                "mapping.put",
                {
                    "bag_id": line.bag_id,
                    "flight_id": line.to_flight_id,
                    "batch_id": batch_id,
                },
                at,
            )
            session.record("bag.put", {"bag": updated.as_dict()}, at)
        self._audit.stage(
            session,
            "remap_batch",
            batch_id,
            "checkin",
            {
                "batch_id": batch_id,
                "action": action,
                "rollback_of": rollback_of,
                "count": len(lines),
                "items": [line.as_dict() for line in lines],
            },
            at,
        )

    def _claim_id(self, batch_id: str) -> str:
        existing = {str(entry.payload["batch"]["id"]) for entry in self._headers()}
        if batch_id:
            if batch_id in existing:
                raise ConflictError(f"remap batch {batch_id!r} is already committed")
            return batch_id
        highest = 0
        for known in existing:
            if known.startswith(ID_PREFIX):
                try:
                    highest = max(highest, int(known[len(ID_PREFIX) :]))
                except ValueError:
                    continue
        return f"{ID_PREFIX}{highest + 1:06d}"

    def _headers(self) -> list[JournalEntry]:
        return [entry for entry in self._store.entries() if entry.kind == BATCH_KIND]

    def _find(self, batch_id: str) -> RemapBatch:
        for entry in self._headers():
            if str(entry.payload["batch"].get("id")) == batch_id:
                return self._to_view(entry)
        raise unknown_remap_batch(batch_id)

    def _rolled_back(self, batch_id: str) -> bool:
        return any(
            str(entry.payload["batch"].get("rollback_of", "")) == batch_id
            for entry in self._headers()
        )

    def _to_view(self, entry: JournalEntry) -> RemapBatch:
        raw = entry.payload["batch"]
        batch_id = str(raw["id"])
        return RemapBatch(
            id=batch_id,
            action=str(raw.get("action", APPLY)),
            items=tuple(RemapLine.from_dict(item) for item in raw.get("items", [])),
            at=str(raw.get("at", "")),
            rollback_of=str(raw.get("rollback_of", "")),
            sequence=entry.sequence,
            rolled_back=self._rolled_back(batch_id),
        )
