"""Capacity protocol migration and primary-probe authority fail before backend dispatch."""

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

import httpx
import pytest
from test_warm_routes import action, backend

from finserve.contracts.inference import InferenceRequest
from finserve.gateway.app import create_app
from finserve.gateway.warm_route_app import WarmRouteEngine, WarmRouteMiddleware
from finserve.reliability import capacity_store
from finserve.reliability.capacity_store import probe_token, revoke_probe, select_admission
from finserve.reliability.rollback import ControlConflict, DeploymentStore
from finserve.reliability.warm_drain import AdmissionLease
from finserve.reliability.warm_routes import RouteSnapshot, WarmRouteStore


def seeded_capacity(path: Path) -> WarmRouteStore:
    """Only a fresh, explicitly enabled store accepts the capacity protocol."""
    store = WarmRouteStore(path, capacity_enabled=True)
    store.register(backend("baseline"))
    store.bootstrap("service", "baseline")
    return store


def test_capacity_refuses_legacy_enrollment_and_old_snapshot_writes(tmp_path: Path) -> None:
    """An existing cooperating drain gateway is still too old for capacity selection."""
    legacy = WarmRouteStore(tmp_path / "legacy.sqlite")
    with pytest.raises(ControlConflict, match="new store"):
        WarmRouteStore(legacy.path, capacity_enabled=True)
    store = seeded_capacity(tmp_path / "capacity.sqlite")
    old = store.snapshot("service").model_dump()
    old.pop("capacity_protocol")
    with pytest.raises(sqlite3.IntegrityError, match="capacity protocol"):
        with store.transaction() as connection:
            connection.execute(
                "UPDATE warm_routes SET payload=? WHERE id='service'", (json.dumps(old),)
            )
    with pytest.raises(ControlConflict, match="authenticated dispatch"):
        store.admit("service")


def test_probe_payload_substitution_replay_and_revocation(tmp_path: Path) -> None:
    """The trusted token selects primary only for its one exact post-auth request."""
    store = seeded_capacity(tmp_path / "capacity.sqlite")
    route = store.snapshot("service")
    request = InferenceRequest(prompt="control smoke", max_tokens=1)
    token = probe_token(store, route, request)
    with pytest.raises(ControlConflict, match="substituted"):
        select_admission(store, route, request.model_copy(update={"prompt": "different"}), token)
    lease = select_admission(store, route, request, token)
    assert lease.kind == "probe" and lease.backend.revision.revision_id == "baseline"
    assert lease.phase == "reserved"
    with pytest.raises(ControlConflict, match="substituted"):
        select_admission(store, route, request, token)
    store.finish_admission(lease)
    other = probe_token(store, route, request)
    revoke_probe(store, other)
    with pytest.raises(ControlConflict, match="substituted"):
        select_admission(store, route, request, other)


def test_probe_route_change_cannot_pin_an_old_primary(tmp_path: Path) -> None:
    """Promotion invalidates old probe authority without waiting for controller observation."""
    store = seeded_capacity(tmp_path / "capacity.sqlite")
    route = store.snapshot("service")
    request = InferenceRequest(prompt="control smoke")
    token = probe_token(store, route, request)
    target = backend("candidate", 9001)
    store.register(target)
    store.apply(action(target))
    with pytest.raises(ControlConflict, match="route changed"):
        select_admission(store, route, request, token)
    with pytest.raises(ControlConflict, match="substituted"):
        select_admission(store, store.snapshot("service"), request, token)
    revoke_probe(store, token)


def test_probe_authority_is_bounded_and_revocation_does_not_release_work(tmp_path: Path) -> None:
    """Unused tokens have a cap; revoking an already consumed token cannot clear a stream."""
    store = seeded_capacity(tmp_path / "capacity.sqlite")
    route, request = store.snapshot("service"), InferenceRequest(prompt="control smoke")
    tokens = [probe_token(store, route, request) for _ in range(64)]
    with pytest.raises(ControlConflict, match="token limit"):
        probe_token(store, route, request)
    lease = select_admission(store, route, request, tokens[0])
    revoke_probe(store, tokens[0])
    with store.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM warm_admissions").fetchone()[0] == 1
    store.finish_admission(lease)


def test_pool_membership_blocks_both_generic_retirement_paths(tmp_path: Path) -> None:
    """Physical pool history is served history despite not being the logical route revision."""
    store = seeded_capacity(tmp_path / "capacity.sqlite")
    control = DeploymentStore(tmp_path / "control.sqlite")
    extra = backend("extra", 9001)
    store.register(extra)
    with store.transaction() as connection:
        connection.execute(
            "INSERT INTO warm_capacity_members VALUES(?,?,?,?,?,?,?)",
            ("extra", "service", "synthetic-plan", "baseline", 0, 1, "fixture"),
        )
    assert not store.retire_unserved(control, extra)
    assert not store.retire_drained(control, extra)
    with store.transaction() as connection:
        connection.execute("UPDATE warm_capacity_members SET ready=0")
    assert not store.retire_unserved(control, extra)
    assert store.retire_drained(control, extra)


def test_route_cutover_advances_durable_pool_identity(tmp_path: Path) -> None:
    """Pool generation changes even before a capacity controller reconciles a new anchor."""
    store = seeded_capacity(tmp_path / "capacity.sqlite")
    lease = select_admission(store, store.snapshot("service"), InferenceRequest(prompt="one"), None)
    first = lease.pool_generation
    store.finish_admission(lease)
    target = backend("candidate", 9001)
    store.register(target)
    store.apply(action(target))
    lease = select_admission(store, store.snapshot("service"), InferenceRequest(prompt="two"), None)
    assert first is not None and lease.pool_generation is not None
    assert lease.pool_generation > first
    store.finish_admission(lease)


@pytest.mark.asyncio
async def test_cancelled_reservation_exposes_owned_pin_before_finalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation after SQL commit releases proven never-entered work rather than losing it."""
    store = seeded_capacity(tmp_path / "capacity.sqlite")
    engine = WarmRouteEngine(store, "service")
    app = create_app(engine)
    app.add_middleware(WarmRouteMiddleware, store=store, deployment_id="service")
    entered, release = threading.Event(), threading.Event()
    original = capacity_store.select_admission

    def delayed(
        routes: WarmRouteStore,
        route: RouteSnapshot,
        request: InferenceRequest,
        token: str | None,
    ) -> AdmissionLease:
        """Pause after durable insert while the async owner is still waiting for its result."""
        lease = original(routes, route, request, token)
        entered.set()
        assert release.wait(5)
        return lease

    monkeypatch.setattr(capacity_store, "select_admission", delayed)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://fixture"
    ) as client:
        task = asyncio.create_task(client.post("/v1/completions", json={"prompt": "hello"}))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.01)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    with store.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM warm_admissions").fetchone()[0] == 0
    assert engine.clients == {}
