"""Atomic batch re-mapping: all-or-nothing commits, rollback and cross-checks."""

from __future__ import annotations

import json
import threading

import pytest

from bagsort.app import Plant
from bagsort.clock import Clock
from bagsort.errors import ConflictError, NotFoundError, ValidationError
from bagsort.flight.remap import BatchRemapError, RemapItem
from bagsort.store.store import Store
from tests.support import check_in, make_plant, make_server, register_flight


def _two_flights(plant):
    first = register_flight(plant, code="CA1234", carousel="CR1")
    second = register_flight(plant, code="MU4321", carousel="CR2")
    return first, second


def _items(bags, flight_id):
    return [RemapItem(bag_id=bag.id, flight_id=flight_id) for bag in bags]


def _call(server, method, path, payload=None):
    response = server.call(method, path, payload)
    return response.status, json.loads(response.body)


# ---------------------------------------------------------------------------
# all or nothing
# ---------------------------------------------------------------------------


def test_batch_apply_moves_every_bag_in_one_commit(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 3)
    watermark_before = plant.store.watermark

    batch = plant.remap_batch(_items(bags, second.id))

    assert batch.action == "apply"
    assert [line.bag_id for line in batch.items] == [bag.id for bag in bags]
    assert all(line.from_flight_id == first.id for line in batch.items)
    assert all(line.to_flight_id == second.id for line in batch.items)
    state = plant.store.state()
    assert all(state.mappings[bag.id] == second.id for bag in bags)
    assert all(state.bags[bag.id].flight_id == second.id for bag in bags)
    # One commit: header, two commands per bag and the audit entry.
    entries = plant.store.entries_since(watermark_before)
    assert entries[0].kind == "mapping.batch"
    assert entries[0].payload["batch"]["id"] == batch.id
    puts = [entry for entry in entries if entry.kind == "mapping.put"]
    assert [entry.payload["batch_id"] for entry in puts] == [batch.id] * 3
    assert entries[-1].kind == "audit.append"
    assert entries[-1].payload["entry"]["subject"] == batch.id
    # The audit entry is part of the same commit and the chain still verifies.
    audit = plant.audit.last()
    assert audit.kind == "remap_batch"
    assert audit.detail["batch_id"] == batch.id
    assert audit.detail["count"] == 3
    assert plant.audit.verify() is True


def test_one_bad_item_fails_the_whole_batch(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 3)
    watermark_before = plant.store.watermark
    items = _items(bags, second.id)
    items[1] = RemapItem(bag_id=bags[1].id, flight_id="T1-NOPE")

    with pytest.raises(BatchRemapError) as caught:
        plant.remap_batch(items)

    assert caught.value.failures[0].bag_id == bags[1].id
    assert "T1-NOPE" in caught.value.failures[0].reason
    assert plant.store.watermark == watermark_before
    assert all(
        plant.store.state().mappings[bag.id] == first.id for bag in bags
    )
    assert plant.remapper.batches() == []


def test_every_offending_item_is_named(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 2)
    closed = register_flight(plant, code="CZ9876")
    plant.book.cutoff(closed.id)
    items = [
        RemapItem(bag_id="T1-BAG-GHOST", flight_id=second.id),
        RemapItem(bag_id=bags[0].id, flight_id=closed.id),
        RemapItem(bag_id=bags[1].id, flight_id=second.id),
    ]

    with pytest.raises(BatchRemapError) as caught:
        plant.remap_batch(items)

    named = {failure.bag_id: failure.reason for failure in caught.value.failures}
    assert "not registered" in named["T1-BAG-GHOST"]
    assert "not accepting baggage" in named[bags[0].id]
    # The valid item is not a failure, but the batch still did not commit.
    assert bags[1].id not in named
    assert plant.store.state().mappings[bags[1].id] == first.id


def test_a_bag_may_appear_only_once_per_batch(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 2)
    items = _items(bags, second.id)
    items.append(RemapItem(bag_id=bags[0].id, flight_id=second.id))

    with pytest.raises(BatchRemapError) as caught:
        plant.remap_batch(items)

    assert "more than once" in caught.value.failures[0].reason
    assert plant.store.state().mappings[bags[0].id] == first.id


def test_empty_and_noop_batches_are_refused(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 1)
    with pytest.raises(ValidationError):
        plant.remap_batch([])
    with pytest.raises(BatchRemapError):
        plant.remap_batch(_items(bags, first.id))
    assert plant.remapper.batches() == []


def test_a_supplied_batch_id_is_single_use(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 2)
    plant.remap_batch(_items(bags, second.id), batch_id="IRROPS-7")

    with pytest.raises(ConflictError):
        plant.remap_batch(_items(bags, first.id), batch_id="IRROPS-7")

    assert [batch.id for batch in plant.remap_batches()] == ["IRROPS-7"]


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def test_rollback_restores_every_mapping_atomically(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 3)
    batch = plant.remap_batch(_items(bags, second.id))

    rolled_back = plant.rollback_remap_batch(batch.id)

    assert rolled_back.action == "rollback"
    assert rolled_back.rollback_of == batch.id
    state = plant.store.state()
    assert all(state.mappings[bag.id] == first.id for bag in bags)
    assert all(state.bags[bag.id].flight_id == first.id for bag in bags)
    assert plant.remap_batch_detail(batch.id).rolled_back is True
    kinds = [entry.kind for entry in plant.audit.entries() if entry.kind == "remap_batch"]
    assert kinds == ["remap_batch", "remap_batch"]
    assert plant.audit.verify() is True


def test_rollback_refuses_a_batch_that_drifted(plant):
    first, second = _two_flights(plant)
    third = register_flight(plant, code="CZ9876")
    bags = check_in(plant, first.id, 3)
    batch = plant.remap_batch(_items(bags, second.id))
    # A concurrent change lands on one of the batch's bags.
    plant.remap_bag(bags[1].id, third.id)
    watermark_before = plant.store.watermark

    with pytest.raises(ConflictError) as caught:
        plant.rollback_remap_batch(batch.id)

    assert bags[1].id in str(caught.value)
    # Nothing moved: the other two bags keep the batch's mapping.
    assert plant.store.watermark == watermark_before
    state = plant.store.state()
    assert state.mappings[bags[0].id] == second.id
    assert state.mappings[bags[1].id] == third.id
    assert plant.remap_batch_detail(batch.id).rolled_back is False


def test_rollback_guards(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 1)
    batch = plant.remap_batch(_items(bags, second.id))

    with pytest.raises(NotFoundError):
        plant.rollback_remap_batch("rmb-999999")
    rolled_back = plant.rollback_remap_batch(batch.id)
    with pytest.raises(ConflictError):
        plant.rollback_remap_batch(batch.id)
    with pytest.raises(ConflictError):
        plant.rollback_remap_batch(rolled_back.id)


# ---------------------------------------------------------------------------
# history and cross-checks
# ---------------------------------------------------------------------------


def test_history_is_reviewable_batch_by_batch(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 4)
    first_batch = plant.remap_batch(_items(bags[:2], second.id))
    second_batch = plant.remap_batch(_items(bags[2:], second.id))

    batches = plant.remap_batches()
    assert [batch.id for batch in batches] == [first_batch.id, second_batch.id]
    detail = plant.remap_batch_detail(first_batch.id)
    assert [line.bag_id for line in detail.items] == [bags[0].id, bags[1].id]
    assert detail.sequence < second_batch.sequence
    with pytest.raises(NotFoundError):
        plant.remap_batch_detail("rmb-004242")


def test_reconcile_ties_journal_ledger_and_audit_together(plant):
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 2)
    batch = plant.remap_batch(_items(bags, second.id))

    report = plant.remap_reconcile(batch.id)

    assert report["intact"] is True
    assert report["audit_chain_intact"] is True
    assert report["audit_sequence"] > 0
    assert report["header_sequence"] > 0
    assert all(item["journal_sequence"] > report["header_sequence"] for item in report["items"])
    assert all(item["matches_batch"] for item in report["items"])
    # A later change is reported as superseding the batch, not as tampering.
    plant.remap_bag(bags[0].id, first.id)
    again = plant.remap_reconcile(batch.id)
    assert again["intact"] is True
    assert again["items"][0]["matches_batch"] is False
    assert again["items"][0]["current_flight_id"] == first.id


def test_reconcile_lists_sort_records_next_to_the_mapping(plant, flight):
    target = register_flight(plant, code="MU4321")
    bags = check_in(plant, flight.id, 1)
    plant.push_bag(bags[0].id, "T1-CH1")
    batch = plant.remap_batch(_items(bags, target.id))

    report = plant.remap_reconcile(batch.id)

    assert report["items"][0]["sort_records"] == 1
    assert report["items"][0]["sorted_flights"] == [flight.id]
    assert report["items"][0]["current_flight_id"] == target.id


def test_a_committed_batch_survives_persist_and_replay(tmp_path):
    plant = make_plant(tmp_path)
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 3)
    batch = plant.remap_batch(_items(bags, second.id))
    digest = plant.store.persist()

    reopened = Plant.build(Store.open(plant.store.path), clock=Clock())

    assert reopened.store.envelope().digest == digest
    assert [item.id for item in reopened.remap_batches()] == [batch.id]
    state = reopened.store.state()
    assert all(state.mappings[bag.id] == second.id for bag in bags)
    assert reopened.remap_reconcile(batch.id)["intact"] is True
    # Rolling back on the reopened store restores the original mapping.
    reopened.rollback_remap_batch(batch.id)
    assert all(
        reopened.store.state().mappings[bag.id] == first.id for bag in bags
    )


def test_concurrent_batches_never_interleave(tmp_path):
    plant = make_plant(tmp_path)
    first, second = _two_flights(plant)
    bags = check_in(plant, first.id, 8)
    errors: list[Exception] = []

    def move(chunk):
        try:
            plant.remap_batch(_items(chunk, second.id))
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=move, args=(bags[index : index + 2],))
        for index in range(0, 8, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert all(plant.store.state().mappings[bag.id] == second.id for bag in bags)
    entries = plant.store.entries()
    headers = [entry for entry in entries if entry.kind == "mapping.batch"]
    assert len(headers) == 4
    for header in headers:
        batch_id = header.payload["batch"]["id"]
        count = len(header.payload["batch"]["items"])
        # The batch's whole footprint - header, per-item commands, audit
        # entry - is one contiguous run of journal sequences.
        span = 2 + count * 2
        run = entries[header.sequence - 1 : header.sequence - 1 + span]
        assert run[0] is header
        assert run[-1].kind == "audit.append"
        puts = [entry for entry in run if entry.kind == "mapping.put"]
        assert len(puts) == count
        assert all(entry.payload["batch_id"] == batch_id for entry in puts)
        assert all(entry.kind != "mapping.batch" for entry in run[1:])
    assert plant.remap_reconcile(headers[0].payload["batch"]["id"])["intact"] is True


# ---------------------------------------------------------------------------
# console
# ---------------------------------------------------------------------------


def test_console_batch_lifecycle(server):
    assert _call(server, "POST", "/api/flights", {"code": "CA1234"})[0] == 201
    assert _call(server, "POST", "/api/flights", {"code": "MU4321"})[0] == 201
    bags = [
        _call(server, "POST", "/api/bags/checkin", {"barcode": f"BAG{n:04d}", "flight_id": "T1-CA1234"})[1]
        for n in range(2)
    ]

    status, batch = _call(
        server,
        "POST",
        "/api/remap/batch",
        {
            "items": [
                {"bag_id": bags[0]["id"], "flight_id": "T1-MU4321"},
                {"bag_id": bags[1]["id"], "flight_id": "T1-MU4321"},
            ]
        },
    )
    assert status == 201
    assert batch["digest"]
    assert [item["bag_id"] for item in batch["items"]] == [bags[0]["id"], bags[1]["id"]]

    status, batches = _call(server, "GET", "/api/remap/batches")
    assert status == 200
    assert [item["id"] for item in batches] == [batch["id"]]

    status, detail = _call(server, "GET", f"/api/remap/{batch['id']}")
    assert status == 200
    assert detail["rolled_back"] is False

    status, report = _call(server, "GET", f"/api/remap/{batch['id']}/reconcile")
    assert status == 200
    assert report["intact"] is True

    status, rolled_back = _call(server, "POST", f"/api/remap/{batch['id']}/rollback")
    assert status == 200
    assert rolled_back["rollback_of"] == batch["id"]
    assert rolled_back["digest"]
    mapping = _call(server, "GET", f"/api/bags/{bags[0]['id']}/flight")[1]
    assert mapping["flight_id"] == "T1-CA1234"


def test_console_batch_failure_names_each_item(server):
    assert _call(server, "POST", "/api/flights", {"code": "CA1234"})[0] == 201
    assert _call(server, "POST", "/api/flights", {"code": "MU4321"})[0] == 201
    bag = _call(
        server, "POST", "/api/bags/checkin", {"barcode": "BAG0000", "flight_id": "T1-CA1234"}
    )[1]

    status, body = _call(
        server,
        "POST",
        "/api/remap/batch",
        {
            "items": [
                {"bag_id": bag["id"], "flight_id": "T1-MU4321"},
                {"bag_id": "T1-BAG-GHOST", "flight_id": "T1-MU4321"},
                {"bag_id": bag["id"], "flight_id": "T1-MU4321"},
            ]
        },
    )

    assert status == 400
    named = {failure["bag_id"]: failure["reason"] for failure in body["failures"]}
    assert "not registered" in named["T1-BAG-GHOST"]
    assert "more than once" in named[bag["id"]]
    # Nothing was applied: the valid item kept its original mapping.
    mapping = _call(server, "GET", f"/api/bags/{bag['id']}/flight")[1]
    assert mapping["flight_id"] == "T1-CA1234"
    assert _call(server, "GET", "/api/remap/batches")[1] == []


def test_console_batch_rollback_conflict_is_a_409(server):
    assert _call(server, "POST", "/api/flights", {"code": "CA1234"})[0] == 201
    assert _call(server, "POST", "/api/flights", {"code": "MU4321"})[0] == 201
    assert _call(server, "POST", "/api/flights", {"code": "CZ9876"})[0] == 201
    bag = _call(
        server, "POST", "/api/bags/checkin", {"barcode": "BAG0000", "flight_id": "T1-CA1234"}
    )[1]
    batch = _call(
        server,
        "POST",
        "/api/remap/batch",
        {"items": [{"bag_id": bag["id"], "flight_id": "T1-MU4321"}]},
    )[1]
    _call(server, "POST", f"/api/bags/{bag['id']}/remap", {"flight_id": "T1-CZ9876"})

    status, body = _call(server, "POST", f"/api/remap/{batch['id']}/rollback")

    assert status == 409
    assert bag["id"] in body["error"]
    assert _call(server, "GET", "/api/remap/rmb-404040")[0] == 404
