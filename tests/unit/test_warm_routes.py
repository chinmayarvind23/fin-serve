"""Traffic cutover correctness does not depend on the orchestration store being in sync."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from finserve.contracts.deployment import Revision
from finserve.contracts.inference import InferenceRequest
from finserve.gateway.body_limit import BodyLimit
from finserve.gateway.warm_route_app import (
    WarmRouteEngine,
    WarmRouteMiddleware,
    request_admission,
    request_route,
)
from finserve.reliability.rollback import ApplyRequest, ControlConflict
from finserve.reliability.warm_routes import (
    BackendConfiguration,
    WarmBackend,
    WarmRouteAdapter,
    WarmRouteStore,
)


def backend(name: str, port: int = 9000) -> WarmBackend:
    """Synthetic image/model labels describe only local transport test fixtures."""
    configuration = BackendConfiguration(base_url=f"http://127.0.0.1:{port}/v1", model="reference")
    return WarmBackend(
        configuration=configuration,
        revision=Revision(
            revision_id=name,
            model_revision="fixture-weights-v1",
            tokenizer_revision="fixture-v1",
            source_revision="fixture-source-v1",
            image_digest="sha256:" + "a" * 64,
            config_digest=configuration.digest(),
            engine="http-fixture",
            engine_config="warm-fixture",
        ),
    )


def action(
    target: WarmBackend, expected: str = "baseline", generation: int = 0, key: str = "apply"
) -> ApplyRequest:
    """Every test action names the exact generation and revision it expects to replace."""
    return ApplyRequest(
        deployment_id="service",
        expected_revision=expected,
        expected_generation=generation,
        target=target.revision,
        idempotency_key=key,
    )


def seeded(tmp_path: Path) -> tuple[WarmRouteStore, WarmBackend, WarmBackend]:
    """Use an external real SQLite database and two independently identified endpoints."""
    store = WarmRouteStore(tmp_path / "routes.db")
    baseline, candidate = backend("baseline"), backend("candidate", 9001)
    store.register(baseline)
    store.register(candidate)
    store.bootstrap("service", "baseline")
    return store, baseline, candidate


def test_endpoint_and_configuration_identity_are_immutable(tmp_path: Path) -> None:
    """Embedded credentials and substituted endpoints cannot retain a revision identity."""
    store, baseline, _ = seeded(tmp_path)
    store.register(baseline)
    for address in (
        "http://user:secret@host/v1",
        "http://host/v1?key=secret",
        "file:///v1",
        "http://host/v1/",
    ):
        with pytest.raises(ValueError):
            BackendConfiguration(base_url=address, model="reference")
    with pytest.raises(ValueError):
        WarmBackend(revision=baseline.revision, configuration=backend("other", 9999).configuration)
    with pytest.raises(ControlConflict):
        store.register(backend("baseline", 9999))
    with pytest.raises(KeyError):
        store.backend("absent")
    assert store.bootstrap("service", "baseline").generation == 0


def test_fenced_cutover_replay_and_stale_action(tmp_path: Path) -> None:
    """Same-key replay acknowledges one effect; old generations cannot switch newer traffic."""
    store, baseline, candidate = seeded(tmp_path)
    first = store.apply(action(candidate))
    assert first.generation == 1
    assert store.apply(action(candidate)) == first
    with pytest.raises(ControlConflict):
        store.apply(action(baseline))
    with pytest.raises(ControlConflict):
        store.apply(action(baseline, key="stale"))
    restored = store.apply(action(baseline, "candidate", 1, "rollback"))
    assert restored.generation == 2
    assert store.apply(action(candidate)) == first
    assert store.snapshot("service") == restored
    assert [row.generation for row in store.history("service")] == [0, 1, 2]
    with pytest.raises(ControlConflict):
        store.bootstrap("service", "baseline")


def test_two_controllers_cannot_both_win_same_generation(tmp_path: Path) -> None:
    """Independent SQLite connections arbitrate a real simultaneous promotion/rollback race."""
    store, _, candidate = seeded(tmp_path)

    def attempt(key: str) -> bool:
        """Each worker opens its own transaction; only one generation-zero action can commit."""
        try:
            store.apply(action(candidate, key=key))
            return True
        except ControlConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, ["one", "two"])) == [False, True]
    assert store.snapshot("service").generation == 1


async def test_health_rejects_generation_race_and_times_out(tmp_path: Path) -> None:
    """A successful smoke on an old snapshot cannot verify a concurrently changed traffic route."""
    store, _, candidate = seeded(tmp_path)
    before = store.snapshot("service")

    async def reply(_: httpx.Request) -> httpx.Response:
        """Switch traffic while the old generation's response is in flight."""
        store.apply(action(candidate))
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "x-finserve-revision": before.revision_id,
                "x-finserve-revision-digest": before.revision_digest,
                "x-finserve-route-generation": "0",
            },
            text=(
                'data: {"choices":[{"index":0,"text":"b","finish_reason":"stop"}],'
                '"usage":{"completion_tokens":1}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        assert not (await WarmRouteAdapter(store, "http://traffic", client).health("service")).ready

    async def stalled(_: httpx.Request) -> httpx.Response:
        """A transport that never replies still gets a bounded total health deadline."""
        await asyncio.sleep(1)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(stalled)) as client:
        with pytest.raises(TimeoutError):
            await WarmRouteAdapter(store, "http://traffic", client, 0.01).health("service")


@pytest.mark.parametrize(
    "variant",
    ["valid", "missing_finish", "missing_usage", "missing_done", "wrong_media", "compressed"],
)
async def test_health_requires_complete_successful_sse(tmp_path: Path, variant: str) -> None:
    """Matching route headers cannot bless malformed or incomplete inference evidence."""
    store, _, _ = seeded(tmp_path)
    before = store.snapshot("service")
    event: dict[str, object] = {
        "choices": [{"index": 0, "text": "b", "finish_reason": "stop"}],
        "usage": {"completion_tokens": 1},
    }
    if variant == "missing_finish":
        event["choices"] = [{"index": 0, "text": "b"}]
    if variant == "missing_usage":
        event.pop("usage")
    body = "data: " + json.dumps(event) + "\n\n"
    if variant != "missing_done":
        body += "data: [DONE]\n\n"

    def reply(_: httpx.Request) -> httpx.Response:
        """Vary protocol evidence while preserving the exact trusted route identity."""
        response = httpx.Response(
            200,
            text=body,
            headers={
                "content-type": "text/plain" if variant == "wrong_media" else "text/event-stream",
                "x-finserve-revision": before.revision_id,
                "x-finserve-revision-digest": before.revision_digest,
                "x-finserve-route-generation": "0",
            },
        )
        if variant == "compressed":
            response.headers["content-encoding"] = "gzip"
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        observation = await WarmRouteAdapter(store, "http://traffic", client).health("service")
    assert observation.ready == observation.smoke_passed == (variant == "valid")


def test_registry_capacity_duplicate_endpoint_and_repository_guard(tmp_path: Path) -> None:
    """Bound the lifetime pool and reject alternate identities for the same physical endpoint."""
    with pytest.raises(ValueError, match="outside"):
        WarmRouteStore(Path(__file__).resolve().parents[2] / "forbidden-routes.db")
    store = WarmRouteStore(tmp_path / "routes.db")
    store.register(backend("first"))
    with pytest.raises(ControlConflict, match="distinct endpoint"):
        store.register(backend("alias"))
    with pytest.raises(KeyError):
        store.backend("alias")
    for index in range(1, 32):
        store.register(backend(f"revision-{index}", 9000 + index))
    with pytest.raises(ControlConflict, match="limit"):
        store.register(backend("overflow", 9999))
    store.register(backend("first"))


def test_rejected_cutovers_leave_route_and_receipts_unchanged(tmp_path: Path) -> None:
    """Reject substituted targets, empty action IDs and model-alias changes before mutation."""
    store, baseline, candidate = seeded(tmp_path)
    altered = action(candidate).model_copy(
        update={"target": candidate.revision.model_copy(update={"model_revision": "other-weights"})}
    )
    for request in (altered, action(candidate, key="")):
        with pytest.raises(ControlConflict, match="target identity|empty action"):
            store.apply(request)
    configuration = BackendConfiguration(base_url="http://127.0.0.1:9999/v1", model="other-model")
    other = WarmBackend(
        configuration=configuration,
        revision=candidate.revision.model_copy(
            update={"revision_id": "other", "config_digest": configuration.digest()}
        ),
    )
    store.register(other)
    with pytest.raises(ControlConflict, match="public model alias"):
        store.apply(action(other))
    assert store.snapshot("service").revision_id == baseline.revision.revision_id
    assert len(store.history("service")) == 1
    assert store.apply(action(candidate)).generation == 1


async def test_invalid_health_configuration_cannot_issue_requests(tmp_path: Path) -> None:
    """Reject nonfinite deadlines and destination credentials before a health request can run."""
    store, _, _ = seeded(tmp_path)
    async with httpx.AsyncClient(trust_env=False) as client:
        for timeout in (0, float("nan"), float("inf"), 31):
            with pytest.raises(ValueError, match="health timeout"):
                WarmRouteAdapter(store, "http://traffic", client, timeout)
        for url in ("ftp://host", "http://user:secret@host", "http://host/?key=x"):
            with pytest.raises(ValueError, match="traffic URL"):
                WarmRouteAdapter(store, url, client)


async def test_missing_credential_and_unpinned_stream_fail_before_client_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither a missing secret nor missing middleware ownership can create an upstream request."""
    store = WarmRouteStore(tmp_path / "routes.db")
    configuration = BackendConfiguration(
        base_url="http://127.0.0.1:9000/v1",
        model="reference",
        credential_env="FINSERVE_TEST_ABSENT_WARM_KEY",
    )
    registered = WarmBackend(
        configuration=configuration,
        revision=backend("secured").revision.model_copy(
            update={"config_digest": configuration.digest()}
        ),
    )
    store.register(registered)
    snapshot = store.bootstrap("service", "secured")
    monkeypatch.delenv("FINSERVE_TEST_ABSENT_WARM_KEY", raising=False)
    engine = WarmRouteEngine(store, "service")
    try:
        with pytest.raises(RuntimeError, match="pinned"):
            _ = [item async for item in engine.stream(InferenceRequest(prompt="test"))]
        token = request_route.set(snapshot)
        lease = store.admit("service")
        admission_token = request_admission.set(lease)
        try:
            with pytest.raises(ValueError, match="credential"):
                _ = [item async for item in engine.stream(InferenceRequest(prompt="test"))]
        finally:
            request_route.reset(token)
            request_admission.reset(admission_token)
            store.finish_admission(lease)
        assert not engine.clients
    finally:
        await engine.close()


async def test_route_unavailable_rejects_without_downstream_work(tmp_path: Path) -> None:
    """A missing route returns a sanitized error without invoking inference or leaking identity."""
    store = WarmRouteStore(tmp_path / "routes.db")

    async def forbidden(scope: Scope, receive: Receive, send: Send) -> None:
        """Any downstream invocation would violate the unavailable-route admission boundary."""
        pytest.fail("unavailable route must not invoke the downstream application")

    transport = httpx.ASGITransport(app=WarmRouteMiddleware(forbidden, store, "missing"))
    async with httpx.AsyncClient(transport=transport, base_url="http://traffic") as client:
        response = await client.post("/v1/completions", json={"prompt": "test"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ROUTE_UNAVAILABLE"
    assert response.json()["error"]["retryable"] is True
    assert str(store.path) not in response.text
    assert "x-finserve-revision" not in response.headers
    assert request_route.get() is None


@pytest.mark.parametrize("preexisting", [False, True])
async def test_receipt_precedes_route_lookup_and_survives_body_middleware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preexisting: bool
) -> None:
    """Legacy route timing still counts toward request receipt through body/admission handling."""
    await asyncio.to_thread((tmp_path / "routes.db").touch)
    store, _, _ = seeded(tmp_path)
    expected = 5.0 if preexisting else 10.0
    observed: list[float] = []
    state: dict[str, float] = {}
    original = store.snapshot
    monkeypatch.setattr("finserve.gateway.warm_route_app.time.perf_counter", lambda: 10.0)

    def checked_snapshot(deployment_id: str):
        """Observe receipt inside the actual offloaded store lookup, before body processing."""
        observed.append(state["finserve_received"])
        return original(deployment_id)

    monkeypatch.setattr(store, "snapshot", checked_snapshot)

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        """Capture the pinned snapshot and original clock after the real body middleware ran."""
        observed.append(scope["state"]["finserve_received"])
        pinned = request_route.get()
        assert pinned is not None and pinned.revision_id == "baseline"
        await JSONResponse({"ok": True})(scope, receive, send)

    middleware = WarmRouteMiddleware(BodyLimit(downstream), store, "service")

    async def ingress(scope: Scope, receive: Receive, send: Send) -> None:
        """Model a trusted earlier ASGI receipt without allowing request headers to set it."""
        scope["state"] = state
        if preexisting:
            state["finserve_received"] = expected
        await middleware(scope, receive, send)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=ingress), base_url="http://traffic"
    ) as client:
        response = await client.post(
            "/v1/completions", json={"prompt": "test"}, headers={"x-finserve-revision": "forged"}
        )
    assert response.status_code == 200
    assert observed == [expected, expected]
    assert response.headers["x-finserve-revision"] == "baseline"
    assert request_route.get() is None
