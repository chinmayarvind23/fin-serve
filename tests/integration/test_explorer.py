"""Real SQL/CAS evidence reads preserve failed requests and reject unsafe GraphQL operations."""

import asyncio
import json
import threading
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import insert, update
from starlette.types import Message, Scope

from finserve.benchmark.runner import RunConfig, run_benchmark
from finserve.benchmark.workload import default_workload
from finserve.contracts.deployment import Revision
from finserve.registry import explorer
from finserve.registry.annotations import AnnotationStore
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.explorer import (
    EvidenceReader,
    ExplorerBoundary,
    QueryInput,
    create_explorer_app,
    execute_query,
)
from finserve.registry.metadata import Registry, RunBundle, decisions, runs
from finserve.reliability.promotion import PromotionDecision

KEY = "explorer-test-service-credential"


@pytest.fixture
async def catalog(tmp_path: Path) -> AsyncGenerator[tuple[Registry, LocalArtifactStore, RunBundle]]:
    """Register actual recorder output, including one failure, rather than fabricating a summary."""
    calls = 0

    def reply(_: httpx.Request) -> httpx.Response:
        """The controlled third HTTP response fails after one separately retained warmup."""
        nonlocal calls
        calls += 1
        if calls == 3:
            return httpx.Response(503)
        return httpx.Response(
            200,
            text=(
                'data: {"choices":[{"text":"<script>literal output</script>"}]}\n\n'
                'data: {"choices":[],"usage":{"completion_tokens":2}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
        await run_benchmark(
            client,
            "http://fixture/v1/completions",
            default_workload(),
            RunConfig(
                requests=3,
                warmup=1,
                concurrency=1,
                hardware="CPU-transport-fixture",
                revision="fixture-source",
                model_revision="fixture-model",
                tokenizer_revision="fixture-tokenizer",
                engine="http-fixture",
                engine_config="one-declared-failure",
            ),
            tmp_path / "run",
        )
    registry = Registry(f"sqlite:///{tmp_path / 'registry.db'}")
    store = LocalArtifactStore(tmp_path / "artifacts")
    bundle = registry.register_run(tmp_path / "run", store)
    try:
        yield registry, store, bundle
    finally:
        registry.close()


async def test_read_registered_metrics_and_failures(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """Nested reads expose source identity, measured failures, slices and artifact checksums."""
    registry, store, bundle = catalog
    response = execute_query(
        registry,
        store,
        QueryInput(
            query="""
      query Run($id: ID!) {
        run(id: $id) {
          id configuration { imageDigest workloadHash concurrency }
          metrics { offered succeeded failed generatedTokens }
          slices { name metrics { offered } }
          requests(first: 4) { phase success error output generatedTokens }
          artifacts { kind sha256 sizeBytes }
        }
        lifecycle { id } decisions { id } events { id }
      }
    """,
            variables={"id": bundle.run_id},
        ),
    )
    assert "errors" not in response
    run = response["data"]["run"]
    assert run["metrics"] == {"offered": 3, "succeeded": 2, "failed": 1, "generatedTokens": 4}
    assert run["configuration"]["imageDigest"] == "undeclared"
    assert len(run["requests"]) == 4
    assert sum(not record["success"] for record in run["requests"]) == 1
    assert run["requests"][0]["phase"] == "warmup"
    assert len(run["slices"]) >= 3
    assert len(run["artifacts"]) == 3
    assert store.namespace not in json.dumps(response)


async def test_cursor_missing_and_invalid_page(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """Stable bounded pages cannot become unbounded scans or caller-selected paths."""
    registry, store, bundle = catalog
    page = execute_query(registry, store, QueryInput(query="{ runs(first: 1) { id } }"))
    assert page["data"]["runs"] == [{"id": bundle.run_id}]
    page = execute_query(
        registry,
        store,
        QueryInput(
            query="query($after: ID) { runs(after: $after) { id } }",
            variables={"after": bundle.run_id},
        ),
    )
    assert page["data"]["runs"] == []
    assert (
        execute_query(registry, store, QueryInput(query='{ run(id: "missing") { id } }'))["data"][
            "run"
        ]
        is None
    )
    with pytest.raises(ValueError, match="bounds"):
        execute_query(registry, store, QueryInput(query="{ runs(first: 1000) { id } }"))


@pytest.mark.parametrize(
    "query",
    [
        "mutation { deleteRun }",
        "{ __schema { types { name } } }",
        "query A { runs { id } } query B { runs { id } }",
        "{ runs { ...Fields } } fragment Fields on Run { id }",
        "{ " + " ".join(f"q{i}: runs {{ id }}" for i in range(121)) + " }",
        "{ a { b { c { d { e { f { g } } } } } } }",
    ],
)
async def test_query_bounds(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle], query: str
) -> None:
    """Reject recursive and amplified operation shapes before executing any storage resolver."""
    registry, store, _ = catalog
    with pytest.raises(ValueError):
        execute_query(registry, store, QueryInput(query=query))


async def test_corrupt_artifact_fails_without_path_leak(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """Registered metadata cannot make changed bytes look like a valid measured result."""
    registry, store, bundle = catalog
    reference = bundle.summary
    (store.root / "sha256" / reference.sha256[:2] / reference.sha256).write_bytes(b"corrupt")
    result = execute_query(registry, store, QueryInput(query="{ runs { id metrics { offered } } }"))
    assert result == {"data": None, "errors": [{"message": "EVIDENCE_QUERY_FAILED"}]}


async def test_read_budget_and_namespace_cannot_be_bypassed_by_cache(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """Cached content must not validate another namespace or disable the operation byte budget."""
    registry, store, bundle = catalog
    reader = EvidenceReader(registry, store)
    assert reader.read(bundle.summary)
    with pytest.raises(ValueError):
        reader.read(bundle.summary.model_copy(update={"namespace": "file:///untrusted"}))
    reader.remaining = 0
    with pytest.raises(ValueError):
        reader.read(bundle.requests)
    reader.deadline = 0
    with pytest.raises(TimeoutError):
        reader.read(bundle.summary)


async def test_http_auth_before_query_and_bounded_body(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """The evidence endpoint has an independent credential and never returns internal exceptions."""
    registry, store, _ = catalog
    app = create_explorer_app(registry, store, KEY)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            assert (await client.post("/graphql", content=b"malformed")).status_code == 401
            headers = {"Authorization": f"Bearer {KEY}"}
            ok = await client.post("/graphql", headers=headers, json={"query": "{ runs { id } }"})
            assert ok.status_code == 200 and len(ok.json()["data"]["runs"]) == 1
            assert ok.headers["cache-control"] == "private, no-store"
            bad = await client.post("/graphql", headers=headers, json={"query": "mutation { x }"})
            assert bad.status_code == 400 and "INVALID_EVIDENCE_QUERY" in bad.text
            assert (
                await client.post("/graphql", headers=headers, content=b"x" * 16385)
            ).status_code == 413
    with pytest.raises(ValueError):
        create_explorer_app(registry, store, "short")


def scope(headers: list[tuple[bytes, bytes]]) -> Scope:
    """Construct direct ASGI transport input so authentication can be checked before receive."""
    return {
        "type": "http",
        "http_version": "1.1",
        "asgi": {"version": "3.0"},
        "method": "POST",
        "scheme": "http",
        "path": "/graphql",
        "raw_path": b"/graphql",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "server": ("test", 80),
        "client": ("client", 1234),
    }


@pytest.mark.parametrize("duplicate", [False, True])
async def test_boundary_rejects_auth_without_receiving(duplicate: bool) -> None:
    """Missing or duplicate credentials fail before a malicious client must send any bytes."""

    async def unused(*_args: object) -> None:
        """No unauthorized request reaches the routed application."""
        raise AssertionError("application invoked")

    async def receive() -> Message:
        """Waiting for a body here would itself violate authentication ordering."""
        raise AssertionError("body consumed")

    sent: list[Message] = []

    async def send(message: Message) -> None:
        """Retain the actual ASGI status without involving an HTTP client buffer."""
        sent.append(message)

    headers = [(b"authorization", f"Bearer {KEY}".encode())] * 2 if duplicate else []
    await ExplorerBoundary(unused, KEY)(scope(headers), receive, send)
    assert sent[0]["status"] == 401


async def test_body_timeout_releases_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """One total deadline stops a client that never finishes its bounded-size request body."""
    monkeypatch.setattr(explorer, "BODY_SECONDS", 0.01)

    async def unused(*_args: object) -> None:
        """An incomplete body never reaches JSON parsing or storage."""
        raise AssertionError("application invoked")

    async def receive() -> Message:
        """Model a connected client that sends no further bytes."""
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    sent: list[Message] = []

    async def send(message: Message) -> None:
        """Collect timeout status and body independently of transport behavior."""
        sent.append(message)

    boundary = ExplorerBoundary(unused, KEY)
    await boundary(scope([(b"authorization", f"Bearer {KEY}".encode())]), receive, send)
    assert sent[0]["status"] == 408
    assert boundary.admission.active == 0


@pytest.mark.parametrize("mode", ["cancel", "native_error", "timeout"])
async def test_native_work_retains_capacity_during_cleanup(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Retain capacity while native threads drain after repeated cancellation or timeout."""
    registry, store, _ = catalog
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    count = 0

    def blocked(*_args: object) -> dict[str, Any]:
        """Use an explicit native barrier; no timing assumption controls worker ownership."""
        nonlocal count
        with lock:
            count += 1
            if count == 4:
                entered.set()
        if not release.wait(10):
            raise AssertionError("test failed to release worker")
        if mode == "native_error":
            raise ValueError("private native failure")
        return {"data": {"runs": []}}

    monkeypatch.setattr(explorer, "execute_query", blocked)
    monkeypatch.setattr(explorer, "QUERY_SECONDS", 0.02 if mode == "timeout" else 60.0)
    app = create_explorer_app(registry, store, KEY)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {KEY}"},
    ) as client:
        tasks = [
            asyncio.create_task(client.post("/graphql", json={"query": "{ runs { id } }"}))
            for _ in range(4)
        ]
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            if mode == "timeout":
                await asyncio.sleep(0.04)
            else:
                tasks[0].cancel()
                await asyncio.sleep(0)
                tasks[0].cancel()
                await asyncio.sleep(0)
            assert not tasks[0].done()
            assert (
                await client.post("/graphql", json={"query": "{ runs { id } }"})
            ).status_code == 429
        finally:
            release.set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        if mode == "timeout":
            assert all(
                isinstance(result, httpx.Response) and result.status_code == 504
                for result in outcomes
            )
        else:
            assert isinstance(outcomes[0], asyncio.CancelledError)
        assert (await client.post("/graphql", json={"query": "{ runs { id } }"})).status_code != 429


async def test_alias_amplification_and_registry_payload_caps(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound expanded fields before SQL and scalar aliases before materializing a large response."""
    registry, store, bundle = catalog
    with pytest.raises(ValueError, match="expanded"):
        execute_query(
            registry,
            store,
            QueryInput(query="{ runs(first:50) { requests(first:50) { output } } }"),
        )
    with pytest.raises(ValueError, match="expanded"):
        execute_query(
            registry,
            store,
            QueryInput(
                query="query($n:Int=50) { runs(first:$n) { requests(first:$n) { output } } }"
            ),
        )
    original = EvidenceReader.run_view

    def enlarged(self: EvidenceReader, value: RunBundle) -> dict[str, Any]:
        """A legitimate long configuration value must be charged once per selected alias."""
        result = original(self, value)
        result["configuration"]["engineConfig"] = "x" * 32768
        return result

    monkeypatch.setattr(EvidenceReader, "run_view", enlarged)
    query = (
        '{ run(id:"'
        + bundle.run_id
        + '") { configuration {'
        + " ".join(f"alias{index}: engineConfig" for index in range(90))
        + "} } }"
    )
    assert execute_query(registry, store, QueryInput(query=query)) == {
        "data": None,
        "errors": [{"message": "EVIDENCE_QUERY_FAILED"}],
    }
    with registry.engine.begin() as connection:
        connection.execute(
            update(runs).where(runs.c.id == bundle.run_id).values(payload="x" * 65537)
        )
    assert "errors" in execute_query(registry, store, QueryInput(query="{ runs { id } }"))


async def test_nonempty_decision_history_uses_typed_property(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """Stored decisions do not serialize their approved property; the explorer computes it."""
    registry, store, bundle = catalog
    revision = Revision(
        revision_id="fixture",
        model_revision="model",
        tokenizer_revision="tokenizer",
        source_revision="source",
        image_digest="sha256:" + "a" * 64,
        config_digest="b" * 64,
        engine="fixture",
        engine_config="test",
    )
    registry.register_revision(revision)
    decision = PromotionDecision(
        candidate_revision=revision.revision_id,
        candidate_digest=revision.digest(),
        policy_version="fixture",
        evidence_digest=None,
        rejection_reasons=("fixture_missing_evidence",),
    )
    # This is a SQL rendering fixture, not a gate-authorized run or real deployment decision.
    with registry.engine.begin() as connection:
        connection.execute(
            insert(decisions).values(
                id="fixture",
                digest="c" * 64,
                revision_id=revision.revision_id,
                candidate_run_id=bundle.run_id,
                payload=decision.model_dump_json(),
            )
        )
    result = execute_query(registry, store, QueryInput(query="{ decisions { approved reasons } }"))
    assert result["data"]["decisions"] == [
        {"approved": False, "reasons": ["fixture_missing_evidence"]}
    ]


async def test_nullable_annotations_are_verified_and_whitelisted(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """Rendering fixtures remain nullable and expose only verified report values, never paths."""
    registry, store, bundle = catalog
    query = QueryInput(
        query='{ run(id:"' + bundle.run_id + '") { gpu { coverage '
        "averageUtilization reportSha256 } quality { passed accuracy "
        "referenceScope hardFailures } } }"
    )
    assert execute_query(registry, store, query)["data"]["run"] == {"gpu": None, "quality": None}
    annotation_store = AnnotationStore(registry, store)
    gpu = annotation_store.save(
        bundle.run_id,
        "gpu",
        {
            "average_gpu_utilization_percent": None,
            "coverage": 0.0,
            "device_ids": [],
            "method": "synthetic rendering fixture, not GPU evidence",
        },
        [b"fixture-input"],
    )
    annotation_store.save(
        bundle.run_id,
        "quality",
        {
            "candidate_accuracy": 0.25,
            "parity": 0.5,
            "case_count": 4,
            "passed": False,
            "scope": "synthetic rendering fixture",
            "reference_scope": "self",
            "hard_failures": ["fixture_failure"],
            "private_path": "must not appear",
        },
        [b"fixture-input"],
    )
    result = execute_query(registry, store, query)
    assert result["data"]["run"]["gpu"] == {
        "coverage": 0.0,
        "averageUtilization": None,
        "reportSha256": gpu.report.sha256,
    }
    assert result["data"]["run"]["quality"]["passed"] is False
    assert "must not appear" not in json.dumps(result)
    (store.root / "sha256" / gpu.report.sha256[:2] / gpu.report.sha256).write_bytes(b"corrupt")
    assert "errors" in execute_query(registry, store, query)


async def test_lifecycle_events_and_invalid_identifiers(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
) -> None:
    """Nonempty state history stays readable while oversized IDs cannot reach SQL predicates."""
    registry, store, _ = catalog
    registry.create_job("history-fixture", "{}")
    state = registry.claim("history-fixture", "fixture-deployment", "fixture-owner", 30)
    registry.advance(state, "fixture-owner", last_error=None)
    registry.release(state.job_id, "fixture-owner")
    result = execute_query(
        registry,
        store,
        QueryInput(
            query="""{
      lifecycle { id status version applyAttempts decisionId }
      events { jobId status version observedAt }
    }"""
        ),
    )
    assert result["data"]["lifecycle"][0]["status"] == "registered"
    assert result["data"]["events"][0]["jobId"] == "history-fixture"
    assert result["data"]["events"][0]["observedAt"] > 0
    reader = EvidenceReader(registry, store)
    with pytest.raises(ValueError, match="cursor"):
        reader.list_runs(after="x" * 129)
    with pytest.raises(ValueError, match="identity"):
        reader.one_run("x" * 129)
    with pytest.raises(ValueError, match="invalid page"):
        explorer.bounded_query("query($n:Int) { runs(first:$n) { id } }", {"n": "injected"})
    with pytest.raises(ValueError, match="invalid page"):
        explorer.bounded_query("{ runs(first:null) { id } }")


async def test_runtime_rejects_network_storage_and_closes_failed_startup(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The initial service cannot silently accept a network store with unbounded transport waits."""
    registry, store, _ = catalog
    with monkeypatch.context() as patch:
        patch.setattr(registry.engine.dialect, "name", "postgresql")
        with pytest.raises(ValueError, match="local SQLite"):
            create_explorer_app(registry, store, KEY)
    closed = False
    original_close = Registry.close

    def observed_close(self: Registry) -> None:
        """Retain a lifecycle observation while actually disposing the failed startup pool."""
        nonlocal closed
        closed = True
        original_close(self)

    monkeypatch.setattr(Registry, "close", observed_close)
    monkeypatch.setenv("FINSERVE_REGISTRY_URL", f"sqlite:///{tmp_path / 'startup.db'}")
    monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(tmp_path / "startup-artifacts"))
    monkeypatch.setenv("FINSERVE_API_KEY", "short")
    with pytest.raises(ValueError, match="credential"):
        explorer.from_env()
    assert closed


@pytest.mark.parametrize("blocked_type", ["http.response.start", "http.response.body"])
async def test_stalled_response_send_releases_capacity(
    catalog: tuple[Registry, LocalArtifactStore, RunBundle],
    monkeypatch: pytest.MonkeyPatch,
    blocked_type: str,
) -> None:
    """Header or body backpressure cannot retain a slot after native query work has finished."""
    registry, store, _ = catalog
    monkeypatch.setattr(explorer, "SEND_SECONDS", 0.01)
    app = create_explorer_app(registry, store, KEY)
    sent: list[Message] = []

    async def receive() -> Message:
        """Supply a complete authenticated operation through the actual body-limit middleware."""
        return {"type": "http.request", "body": b'{"query":"{ runs { id } }"}', "more_body": False}

    async def send(message: Message) -> None:
        """Retain one response attempt and block the chosen transport stage."""
        sent.append(message)
        if message["type"] == blocked_type:
            await asyncio.Event().wait()

    with pytest.raises(explorer.ExplorerSendTimeout):
        await app(scope([(b"authorization", f"Bearer {KEY}".encode())]), receive, send)
    assert len([message for message in sent if message["type"] == "http.response.start"]) == 1
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/graphql",
            headers={"Authorization": f"Bearer {KEY}"},
            json={"query": "{ runs { id } }"},
        )
    assert response.status_code == 200
