"""Cross-process stream obligations and immutable gateway protocol fence runtime retirement."""

import asyncio
import json
import multiprocessing
import os
import sqlite3
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_warm_rollback import bound_backend

from finserve.contracts.deployment import HealthObservation, ImmutableModel
from finserve.engines.openai_adapter import OpenAICompletionEngine
from finserve.gateway.app import create_app
from finserve.gateway.warm_route_app import WarmRouteEngine, WarmRouteMiddleware
from finserve.registry.model_assets import owned_disk
from finserve.reliability.rollback import ApplyRequest, ControlConflict, DeploymentStore
from finserve.reliability.warm_drain import ADMISSION_PROTOCOL, AdmissionTransport
from finserve.reliability.warm_routes import WarmBackend, WarmRouteStore


def stores(
    root: Path, *, legacy: bool = False
) -> tuple[WarmRouteStore, DeploymentStore, WarmBackend, WarmBackend]:
    """Use two immutable fixture endpoints; an existing blank database takes the legacy path."""
    path = root / "routes.sqlite"
    if legacy:
        path.touch()
    routes = WarmRouteStore(path)
    control = DeploymentStore(root / "control.sqlite")
    old = bound_backend("old", "http://127.0.0.1:9000")
    new = bound_backend("new", "http://127.0.0.1:9001")
    routes.register(old)
    routes.register(new)
    routes.bootstrap("service", "old")
    return routes, control, old, new


def cutover(routes: WarmRouteStore, new: WarmBackend) -> None:
    """Only the new endpoint receives admissions after this committed generation change."""
    routes.apply(
        ApplyRequest(
            deployment_id="service",
            expected_revision="old",
            expected_generation=0,
            target=new.revision,
            idempotency_key="switch",
        )
    )


def obligations(routes: WarmRouteStore) -> int:
    """Inspect durable row count without interpreting process age as ownership evidence."""
    with routes.transaction() as connection:
        return connection.execute("SELECT COUNT(*) FROM warm_admissions").fetchone()[0]


def test_retire_requires_drained_rows_and_never_reopens_revision(tmp_path: Path) -> None:
    """A pending pin survives cutover; the irreversible tombstone bars future rollback."""
    routes, control, old, new = stores(tmp_path)
    lease = routes.admit("service")
    cutover(routes, new)
    assert not routes.retire_drained(control, old)
    with pytest.raises(ControlConflict, match="retired"):
        routes.backend("old")
    assert lease.backend == old
    with pytest.raises(ControlConflict, match="retired"):
        routes.apply(
            ApplyRequest(
                deployment_id="service",
                expected_revision="new",
                expected_generation=1,
                target=old.revision,
                idempotency_key="rollback",
            )
        )
    assert routes.admit("service").snapshot.revision_id == "new"
    routes.finish_admission(lease)
    assert routes.retire_drained(control, old)
    with routes.transaction() as connection:
        first = connection.execute(
            "SELECT payload FROM warm_drain_receipts WHERE revision_id='old'"
        ).fetchone()[0]
    assert routes.retire_drained(control, old)
    with routes.transaction() as connection:
        assert (
            connection.execute(
                "SELECT payload FROM warm_drain_receipts WHERE revision_id='old'"
            ).fetchone()[0]
            == first
        )
    assert json.loads(first)["revision_digest"] == old.revision.digest()


@pytest.mark.parametrize("legacy", [False, True])
def test_legacy_stores_and_controller_targets_remain_protected(
    tmp_path: Path, legacy: bool
) -> None:
    """No protocol is retrofitted, and active/known-good targets override historical drain."""
    routes, control, old, new = stores(tmp_path, legacy=legacy)
    assert (routes.admission_protocol is None) == legacy
    assert not routes.retire_drained(control, old)
    cutover(routes, new)
    control.register_revision(old.revision)
    control.bootstrap(
        "other",
        "old",
        HealthObservation(
            revision_id="old", revision_digest=old.revision.digest(), ready=True, smoke_passed=True
        ),
    )
    assert not routes.retire_drained(control, old)
    with routes.transaction() as connection:
        assert not connection.execute("SELECT 1 FROM warm_retirements").fetchone()
    if legacy:
        assert WarmRouteStore(routes.path).admission_protocol is None
        with pytest.raises(ControlConflict, match="legacy"):
            routes.admit("service")


def test_new_store_rejects_legacy_route_readers_and_writers(tmp_path: Path) -> None:
    """Extra-forbid old readers cannot pin a route and SQLite rejects their original write shape."""
    routes, _, _, _ = stores(tmp_path)
    snapshot = routes.snapshot("service")
    assert snapshot.admission_protocol == ADMISSION_PROTOCOL

    class LegacySnapshot(ImmutableModel):
        """The actual historical route schema has no admission protocol field."""

        deployment_id: str
        revision_id: str
        revision_digest: str
        generation: int
        changed_at: float

    with pytest.raises(ValidationError):
        LegacySnapshot.model_validate_json(snapshot.model_dump_json())
    old_payload = snapshot.model_dump()
    old_payload.pop("admission_protocol")
    for table in ("warm_routes", "warm_events", "warm_actions"):
        with pytest.raises(sqlite3.IntegrityError, match="legacy admission"):
            with routes.transaction() as connection:
                if table == "warm_routes":
                    connection.execute(
                        "UPDATE warm_routes SET payload=?", (json.dumps(old_payload),)
                    )
                elif table == "warm_events":
                    connection.execute(
                        "INSERT INTO warm_events(deployment_id,payload) VALUES(?,?)",
                        ("service", json.dumps(old_payload)),
                    )
                else:
                    connection.execute(
                        "INSERT INTO warm_actions VALUES(?,?,?)",
                        ("old", "old", json.dumps(old_payload)),
                    )


async def test_selection_and_admission_are_atomic_with_cutover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause after snapshot selection cannot let retirement miss its not-yet-inserted pin."""
    routes, control, old, new = stores(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = routes._snapshot  # pyright: ignore[reportPrivateUsage]

    def paused(connection: sqlite3.Connection, deployment_id: str):
        """Hold the route transaction at the historical read/registration race boundary."""
        snapshot = original(connection, deployment_id)
        entered.set()
        assert release.wait(5)
        return snapshot

    monkeypatch.setattr(routes, "_snapshot", paused)
    pin = asyncio.create_task(owned_disk(lambda: routes.admit("service")))
    assert await asyncio.to_thread(entered.wait, 5)
    switch = asyncio.create_task(owned_disk(lambda: cutover(routes, new)))
    await asyncio.sleep(0.05)
    assert not switch.done()
    release.set()
    lease = await pin
    await switch
    assert not routes.retire_drained(control, old)
    routes.finish_admission(lease)
    assert routes.retire_drained(control, old)


class HeldResponse(httpx.AsyncByteStream):
    """The final backend-close operation can remain active after valid protocol EOF."""

    def __init__(self, entered: asyncio.Event, release: asyncio.Event, *, fail: bool) -> None:
        """Inject failure only in actual transport closure, not parser output."""
        self.entered, self.release, self.fail = entered, release, fail

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Return complete valid SSE so the test cannot confuse DONE with closed ownership."""
        yield b'data: {"choices":[{"index":0,"text":"yes","finish_reason":"length"}]}\n\n'
        yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'

    async def aclose(self) -> None:
        """Report close completion independently of receiving every response byte."""
        self.entered.set()
        await self.release.wait()
        if self.fail:
            raise OSError("transport close failed")


@pytest.mark.parametrize("fault", ["none", "cancel", "close", "send", "publication"])
async def test_direct_gateway_assembly_releases_only_verified_closed_streams(
    tmp_path: Path, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct create_app+middleware composition participates in durable admission automatically."""
    routes, control, old, new = stores(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    def response(request: httpx.Request) -> httpx.Response:
        """A failed send has no response handle and cannot claim it never entered the backend."""
        if fault == "send":
            entered.set()
            raise httpx.ReadError("request accepted, response lost")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=HeldResponse(entered, release, fail=fault == "close"),
        )

    if fault == "publication":

        def failed_publication(*args: object) -> None:
            """A closed socket does not excuse losing the durable acknowledgement write."""
            raise OSError("admission acknowledgement failed")

        monkeypatch.setattr(routes, "finish_admission", failed_publication)
    engine = WarmRouteEngine(routes, "service")
    engine.clients["old"] = OpenAICompletionEngine(
        old.configuration.base_url, transport=AdmissionTransport(httpx.MockTransport(response))
    )
    app = create_app(engine, model="reference")
    app.add_middleware(WarmRouteMiddleware, store=routes, deployment_id="service")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            task = asyncio.create_task(
                client.post(
                    "/v1/completions",
                    json={
                        "model": "reference",
                        "prompt": "hi",
                        "max_tokens": 1,
                        "stream": False,
                    },
                )
            )
            await asyncio.wait_for(entered.wait(), 5)
            cutover(routes, new)
            assert obligations(routes) == 1
            assert not routes.retire_drained(control, old)
            if fault == "cancel":
                task.cancel()
                task.cancel()
                await asyncio.sleep(0.05)
                assert not task.done() and obligations(routes) == 1
            release.set()
            if fault == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif fault == "publication":
                with pytest.raises(OSError, match="acknowledgement failed"):
                    await task
            else:
                await task
            uncertain = fault in {"close", "send", "publication"}
            assert obligations(routes) == int(uncertain)
            assert routes.retire_drained(control, old) == (not uncertain)
    finally:
        release.set()
        await engine.close()


def pin_in_process(root: str) -> None:
    """A separate gateway executor owns a durable pin until explicit successful completion."""
    path = Path(root)
    routes = WarmRouteStore(path / "routes.sqlite")
    lease = routes.admit("service")
    (path / "pinned").write_text(lease.admission_id)
    import time

    while not (path / "finish").exists():
        time.sleep(0.02)
    routes.finish_admission(lease)


@pytest.mark.skipif(os.name != "posix", reason="separate-process failure test runs on Linux")
@pytest.mark.parametrize("crash", [False, True])
async def test_process_exit_does_not_expire_unacknowledged_pin(tmp_path: Path, crash: bool) -> None:
    """Process death cannot turn a possibly still-active backend request into drain proof."""
    routes, control, old, new = stores(tmp_path)
    process = multiprocessing.get_context("spawn").Process(
        target=pin_in_process, args=(str(tmp_path),)
    )
    try:
        process.start()
        async with asyncio.timeout(10):
            while not (tmp_path / "pinned").exists():  # noqa: ASYNC110
                await asyncio.sleep(0.02)
        cutover(routes, new)
        assert not routes.retire_drained(control, old)
        if crash:
            process.terminate()
        else:
            (tmp_path / "finish").touch()
        await asyncio.to_thread(process.join, 5)
        assert not process.is_alive()
        assert routes.retire_drained(control, old) is (not crash)
        assert obligations(routes) == int(crash)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


async def test_new_gateway_rejects_untracked_engine_composition(tmp_path: Path) -> None:
    """A new middleware cannot publish drain proof for an engine outside its tracking protocol."""
    routes, _, old, _ = stores(tmp_path)
    calls = 0

    def unexpected(request: httpx.Request) -> httpx.Response:
        """No backend request is allowed before the bound-engine capability is verified."""
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    engine = OpenAICompletionEngine(
        old.configuration.base_url, transport=httpx.MockTransport(unexpected)
    )
    app = create_app(engine, model="reference")
    app.add_middleware(WarmRouteMiddleware, store=routes, deployment_id="service")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            response = await client.post(
                "/v1/completions", json={"model": "reference", "prompt": "hello"}
            )
        assert response.status_code == 503 and calls == obligations(routes) == 0
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_retired_pool_eviction_preserves_local_stream_owner(tmp_path: Path) -> None:
    """Retirement alone cannot close an adapter used by a locally admitted stream."""
    routes, control, old, new = stores(tmp_path)
    cutover(routes, new)
    assert routes.retire_drained(control, old)
    engine = WarmRouteEngine(routes, "service")
    engine.client_limit = 1
    old_client = OpenAICompletionEngine(old.configuration.base_url)
    engine.clients["old"] = old_client
    engine.client_users["old"] = 1
    with pytest.raises(RuntimeError, match="client limit"):
        await engine._client(routes.snapshot("service"))  # pyright: ignore[reportPrivateUsage]
    assert engine.clients == {"old": old_client}
    assert not old_client._client.is_closed  # pyright: ignore[reportPrivateUsage]
    engine.client_users.pop("old")
    replacement = await engine._client(  # pyright: ignore[reportPrivateUsage]
        routes.snapshot("service")
    )
    assert old_client._client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert engine.clients == {"new": replacement}
    await engine.close()
