"""Batch re-mapping: atomic commit, per-item failures, audit and concurrency."""

from __future__ import annotations

import json
import threading

import pytest

from bagsort.app import Plant
from bagsort.clock import Clock
from bagsort.errors import BatchRemapError, NotFoundError, ValidationError
from bagsort.flight.remap import RemapItem
from bagsort.store.store import Store
from tests.support import check_in, make_plant, make_server, register_flight


@pytest.fixture()
def two_flights(plant):
    first = register_flight(plant, code="CA1234", carousel="CR1")
    second = register_flight(plant, code="CA9876", carousel="CR2")
    return first, second


def _items(bags, flight_id):
    return [RemapItem(bag_id=bag.id, flight_id=flight_id) for bag in bags]


def test_batch_remap_commits_everything_in_one_shot(plant, two_flights):
    first, second = two_flights
    bags = check_in(plant, first.id, 3)
    watermark_before = plant.store.watermark
    audit_before = plant.audit.count()

    result = plant.remap_batch(_items(bags, second.id))

    assert result.batch_id == f"remap-{watermark_before + 1:06d}"
    assert [item["to_flight_id"] for item in result.items] == [second.id] * 3
    assert [item["from_flight_id"] for item in result.items] == [first.id] * 3
    # One commit: 3 x (mapping.put + bag.put + audit.append), nothing else.
    assert plant.store.watermark == watermark_before + 9
    assert result.watermark == plant.store.watermark
    assert plant.audit.count() == audit_before + 3

    state = plant.store.state()
    for bag in bags:
        assert state.mappings[bag.id] == second.id
        assert state.bags[bag.id].flight_id == second.id
        assert plant.mapping.flight_of(bag.id) == second.id

    # Journal, mapping ledger and audit chain tell the same story.
    entries = [e for e in plant.store.entries_since(watermark_before)]
    assert [e.kind for e in entries] == (
        ["mapping.put", "bag.put"] * 3 + ["audit.append"] * 3
    )
    assert all(e.payload.get("batch_id") == result.batch_id for e in entries[::2][:3])
    for bag in bags:
        trail = plant.audit.for_subject(bag.id)
        remap_entries = [e for e in trail if e.kind == "remap"]
        assert len(remap_entries) == 1
        assert remap_entries[0].detail["batch_id"] == result.batch_id
        assert remap_entries[0].detail["to_flight_id"] == second.id
    assert plant.audit.verify() is True


def test_one_bad_item_rejects_the_whole_batch(plant, two_flights):
    first, second = two_flights
    bags = check_in(plant, first.id, 3)
    watermark_before = plant.store.watermark
    audit_before = plant.audit.count()
    items = _items(bags, second.id)
    items[1] = RemapItem(bag_id="T1-GHOST", flight_id=second.id)

    with pytest.raises(BatchRemapError) as caught:
        plant.remap_batch(items)

    failures = caught.value.failures
    assert len(failures) == 1
    assert failures[0]["index"] == 1
    assert failures[0]["bag_id"] == "T1-GHOST"
    assert "not registered" in failures[0]["reason"]

    # Nothing landed: no partial mappings, no journal entries, no audit lines.
    assert plant.store.watermark == watermark_before
    assert plant.audit.count() == audit_before
    state = plant.store.state()
    for bag in bags:
        assert state.mappings[bag.id] == first.id
        assert state.bags[bag.id].flight_id == first.id


def test_every_failing_item_is_named(plant, two_flights):
    first, second = two_flights
    bags = check_in(plant, first.id, 2)
    plant.book.cutoff(second.id)
    items = _items(bags, second.id) + [RemapItem(bag_id="T1-GHOST", flight_id=second.id)]

    with pytest.raises(BatchRemapError) as caught:
        plant.remap_batch(items)

    failures = caught.value.failures
    assert [f["bag_id"] for f in failures] == [bags[0].id, bags[1].id, "T1-GHOST"]
    assert "not accepting baggage" in failures[0]["reason"]
    assert "not registered" in failures[2]["reason"]
    # The batch was rejected as a whole, even for the bags that only failed
    # because the target flight is closed.
    assert all(
        plant.store.state().mappings[bag.id] == first.id for bag in bags
    )


def test_unknown_flight_and_already_belonging_are_item_failures(plant, two_flights):
    first, second = two_flights
    bags = check_in(plant, first.id, 2)
    items = [
        RemapItem(bag_id=bags[0].id, flight_id="T1-NOSUCH"),
        RemapItem(bag_id=bags[1].id, flight_id=first.id),
    ]
    with pytest.raises(BatchRemapError) as caught:
        plant.remap_batch(items)
    reasons = [f["reason"] for f in caught.value.failures]
    assert "flight 'T1-NOSUCH' is not registered" in reasons[0]
    assert f"already belongs to {first.id!r}" in reasons[1]


def test_a_bag_may_appear_only_once_per_batch(plant, two_flights):
    first, second = two_flights
    bags = check_in(plant, first.id, 2)
    items = _items(bags, second.id) + [RemapItem(bag_id=bags[0].id, flight_id=second.id)]
    with pytest.raises(ValidationError, match="appears twice"):
        plant.remap_batch(items)
    assert plant.store.state().mappings[bags[0].id] == first.id


def test_empty_and_shapeless_batches_are_rejected(plant, two_flights):
    _, second = two_flights
    with pytest.raises(ValidationError, match="at least one bag"):
        plant.remap_batch([])
    with pytest.raises(ValidationError, match="item 0"):
        plant.remap_batch([RemapItem(bag_id="", flight_id=second.id)])


def test_batch_history_replays_line_by_line(plant, two_flights):
    first, second = two_flights
    bags = check_in(plant, first.id, 2)
    result = plant.remap_batch(_items(bags, second.id))

    history = plant.batch_remap.history(result.batch_id)
    assert history["count"] == 2
    for line, bag in zip(history["items"], bags):
        assert line["bag_id"] == bag.id
        assert line["flight_id"] == second.id
        assert line["from_flight_id"] == first.id
        assert line["current_flight_id"] == second.id
        assert line["audit_sequence"] > 0
        assert line["audit_digest"]
        # The audit digest on the line is the one in the committed chain.
        committed = plant.audit.for_subject(bag.id)[-1]
        assert committed.digest == line["audit_digest"]

    with pytest.raises(NotFoundError):
        plant.batch_remap.history("remap-999999")


def test_committed_batch_survives_a_restart(plant, two_flights, tmp_path):
    first, second = two_flights
    bags = check_in(plant, first.id, 2)
    result = plant.remap_batch(_items(bags, second.id))
    plant.store.persist()

    reopened = Plant.build(Store.open(plant.store.path), clock=Clock())
    assert reopened.store.state().mappings == plant.store.state().mappings
    assert reopened.audit.verify() is True
    history = reopened.batch_remap.history(result.batch_id)
    assert history["count"] == 2
    assert {line["bag_id"] for line in history["items"]} == {bag.id for bag in bags}


def test_concurrent_batches_are_serialized_whole(plant, two_flights):
    first, second = two_flights
    bags_a = check_in(plant, first.id, 6, prefix="AAAA")
    bags_b = check_in(plant, first.id, 6, prefix="BBBB")
    watermark_before = plant.store.watermark
    errors: list[Exception] = []
    results = []

    def move(bags):
        try:
            results.append(plant.remap_batch(_items(bags, second.id)))
        except Exception as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=move, args=(bags,)) for bags in (bags_a, bags_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    # Both batches committed whole: exactly 2 x 6 x 3 entries, and each
    # batch's mapping.put entries form one contiguous run in the journal.
    assert plant.store.watermark == watermark_before + 36
    runs: list[str] = []
    for entry in plant.store.entries_since(watermark_before):
        if entry.kind != "mapping.put":
            continue
        batch_id = entry.payload["batch_id"]
        if not runs or runs[-1] != batch_id:
            runs.append(batch_id)
    assert len(runs) == 2
    assert plant.audit.verify() is True
    state = plant.store.state()
    for bag in bags_a + bags_b:
        assert state.mappings[bag.id] == second.id


def test_a_failing_batch_never_disturbs_a_concurrent_one(plant, two_flights):
    first, second = two_flights
    good = check_in(plant, first.id, 4, prefix="GOOD")
    bad = check_in(plant, first.id, 4, prefix="BAAD")
    errors: list[Exception] = []

    def move_good():
        plant.remap_batch(_items(good, second.id))

    def move_bad():
        try:
            items = _items(bad, second.id)
            items[2] = RemapItem(bag_id="T1-GHOST", flight_id=second.id)
            plant.remap_batch(items)
        except BatchRemapError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=move_good), threading.Thread(target=move_bad)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    state = plant.store.state()
    # The good batch landed whole; the rejected batch left no trace at all.
    assert all(state.mappings[bag.id] == second.id for bag in good)
    assert all(state.mappings[bag.id] == first.id for bag in bad)
    assert plant.audit.verify() is True
    replayed = plant.store.state().as_dict()
    assert replayed["mappings"] == {
        key: value for key, value in sorted(state.mappings.items())
    }


# ---------------------------------------------------------------------------
# console
# ---------------------------------------------------------------------------


def _call(server, method, path, payload=None):
    response = server.call(method, path, payload)
    return response.status, json.loads(response.body)


def test_console_remap_batch_round_trip(server):
    plant = server.plant
    first = register_flight(plant, code="CA1234", carousel="CR1")
    second = register_flight(plant, code="CA9876", carousel="CR2")
    bags = check_in(plant, first.id, 2)

    status, body = _call(
        server,
        "POST",
        "/api/bags/remap-batch",
        [{"bag_id": bag.id, "flight_id": second.id} for bag in bags],
    )
    assert status == 200
    assert body["count"] == 2
    assert body["batch_id"].startswith("remap-")
    assert body["items"][0]["to_flight_id"] == second.id

    status, history = _call(server, "GET", f"/api/remaps/{body['batch_id']}")
    assert status == 200
    assert history["count"] == 2
    assert {line["bag_id"] for line in history["items"]} == {bag.id for bag in bags}

    status, _ = _call(server, "GET", "/api/remaps/remap-424242")
    assert status == 404


def test_console_rejects_a_bad_batch_with_per_item_detail(server):
    plant = server.plant
    first = register_flight(plant, code="CA1234", carousel="CR1")
    second = register_flight(plant, code="CA9876", carousel="CR2")
    bags = check_in(plant, first.id, 2)
    watermark_before = plant.store.watermark

    status, body = _call(
        server,
        "POST",
        "/api/bags/remap-batch",
        [
            {"bag_id": bags[0].id, "flight_id": second.id},
            {"bag_id": "T1-GHOST", "flight_id": second.id},
        ],
    )
    assert status == 409
    assert body["code"] == "batch_remap_rejected"
    assert body["failures"] == [
        {"index": 1, "bag_id": "T1-GHOST", "reason": "bag 'T1-GHOST' is not registered"}
    ]
    assert plant.store.watermark == watermark_before
    assert plant.store.state().mappings[bags[0].id] == first.id

    status, _ = _call(server, "POST", "/api/bags/remap-batch", [])
    assert status == 400
    status, _ = _call(
        server,
        "POST",
        "/api/bags/remap-batch",
        [
            {"bag_id": bags[0].id, "flight_id": second.id},
            {"bag_id": bags[0].id, "flight_id": second.id},
        ],
    )
    assert status == 400
