"""Real loopback HTTP rollback drill; synthetic engine failures never touch existing services."""

import asyncio
import json
import os
import socket
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from finserve.benchmark.metrics import RequestRecord
from finserve.benchmark.runner import RunConfig, request_one
from finserve.benchmark.workload import WorkItem
from finserve.contracts.deployment import HealthObservation, RegressionSignal, Revision
from finserve.gateway.warm_route_app import create_warm_app
from finserve.http_ownership import HTTPClosureError
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.producer_stages import ProducerStages
from finserve.reliability.monitor import MonitorPolicy, ProbeMonitor
from finserve.reliability.promotion import PromotionDecision
from finserve.reliability.rollback import ApplyRequest, DeploymentStore, RollbackController
from finserve.reliability.warm_routes import (
    BackendConfiguration,
    WarmBackend,
    WarmRouteAdapter,
    WarmRouteStore,
)


@asynccontextmanager
async def live_server(app: FastAPI) -> AsyncGenerator[str]:
    """Own a fresh ephemeral loopback listener and drain only this test's Uvicorn instance."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.setblocking(False)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("test HTTP server failed startup")
                    await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 10)


class BackendFixture:
    """A controlled transport fixture emits valid one-token SSE or a declared HTTP failure."""

    def __init__(self, name: str, journal: list[dict[str, Any]], path: Path) -> None:
        """Journal every actual backend request, including health smoke and failed requests."""
        self.name, self.journal, self.failed = name, journal, False
        self.path = path
        self.app = FastAPI()
        self.app.post("/v1/completions", response_model=None)(self.complete)

    async def complete(self, request: Request) -> StreamingResponse | JSONResponse:
        """The fault flag affects this owned HTTP fixture only, not another service or container."""
        payload = await request.json()
        self.journal.append(
            {
                "revision": self.name,
                "received_at": time.time(),
                "failed": self.failed,
                "model": payload["model"],
                "max_tokens": payload["max_tokens"],
            }
        )
        append_evidence(self.path, self.journal[-1])
        if self.failed:
            return JSONResponse({"error": {"code": "synthetic_regression"}}, status_code=503)

        async def events() -> AsyncGenerator[str]:
            """The fixture supplies authoritative token usage independently of event count."""
            yield (
                "data: "
                + json.dumps(
                    {"choices": [{"index": 0, "text": self.name[0], "finish_reason": None}]}
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                        "usage": {"completion_tokens": 1},
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")


def bound_backend(name: str, url: str) -> WarmBackend:
    """These fixture identities do not claim a built image, language model or measured quality."""
    configuration = BackendConfiguration(base_url=url + "/v1", model="reference")
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


def append_evidence(path: Path, record: dict[str, Any]) -> None:
    """Append observed requests immediately so a later failed assertion cannot discard them."""
    with path.open("a") as journal:
        journal.write(json.dumps(record) + "\n")


def remember(records: list[dict[str, Any]], record: RequestRecord, directory: Path) -> None:
    """Preserve each completed or failed offered request before inspecting its result."""
    records.append(record.model_dump())
    append_evidence(directory / "requests.jsonl", record.model_dump())


async def test_actual_http_detect_switch_and_verified_recovery(tmp_path: Path) -> None:
    """Measure real HTTP detector-to-health and retain failures and generation changes."""
    journal: list[dict[str, Any]] = []
    baseline_server, candidate_server = (
        BackendFixture("baseline", journal, tmp_path / "backend-requests.jsonl"),
        BackendFixture("candidate", journal, tmp_path / "backend-requests.jsonl"),
    )
    route_store = WarmRouteStore(tmp_path / "routes.db")
    control = DeploymentStore(tmp_path / "control.db")
    records: list[dict[str, Any]] = []
    async with (
        live_server(baseline_server.app) as baseline_url,
        live_server(candidate_server.app) as candidate_url,
    ):
        baseline, candidate = (
            bound_backend("baseline", baseline_url),
            bound_backend("candidate", candidate_url),
        )
        for backend in (baseline, candidate):
            route_store.register(backend)
            control.register_revision(backend.revision)
        route_store.bootstrap("service", "baseline")
        async with live_server(create_warm_app(route_store, "service")) as traffic_url:
            async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                adapter = WarmRouteAdapter(route_store, traffic_url, client)
                control.bootstrap("service", "baseline", await adapter.health("service"))
                await adapter.apply(
                    ApplyRequest(
                        deployment_id="service",
                        expected_revision="baseline",
                        expected_generation=0,
                        target=candidate.revision,
                        idempotency_key="fixture-promotion",
                    )
                )
                decision = PromotionDecision(
                    candidate_revision="candidate",
                    candidate_digest=candidate.revision.digest(),
                    policy_version="fixture-control-only",
                    evidence_digest="f" * 64,
                    rejection_reasons=(),
                )
                control.activate_candidate("service", 0, decision, await adapter.health("service"))
                item = WorkItem(case_id="health-fixture", prompt="health", max_tokens=1)
                config = RunConfig(requests=6, concurrency=1, warmup=0, timeout_s=5)
                for index in range(3):
                    record = await request_one(
                        client,
                        traffic_url + "/v1/completions",
                        item,
                        index,
                        time.perf_counter(),
                        config,
                        "before_fault",
                    )
                    remember(records, record, tmp_path)
                    assert record.success
                candidate_server.failed = True
                failed = await request_one(
                    client,
                    traffic_url + "/v1/completions",
                    item,
                    3,
                    time.perf_counter(),
                    config,
                    "fault",
                )
                remember(records, failed, tmp_path)
                assert not failed.success
                observed = route_store.snapshot("service")
                detected = time.time()
                operation = control.detect(
                    RegressionSignal(
                        signal_id="owned-http-fault",
                        deployment_id="service",
                        observed_revision=observed.revision_id,
                        observed_generation=observed.generation,
                        detected_at=detected,
                        detector="local-http-error",
                        reason="synthetic owned backend 503",
                    )
                )
                restored = await RollbackController(control, timeout_seconds=5).resume(
                    operation.operation_id, adapter
                )
                assert restored.status == "restored" and restored.duration_seconds() is not None
                assert (
                    route_store.snapshot("service").generation
                    == control.deployment("service").generation
                    == 2
                )
                for index in (4, 5):
                    record = await request_one(
                        client,
                        traffic_url + "/v1/completions",
                        item,
                        index,
                        time.perf_counter(),
                        config,
                        "restored",
                    )
                    remember(records, record, tmp_path)
                    assert record.success and record.output == "b"
                (tmp_path / "rollback.json").write_text(
                    json.dumps(
                        {
                            "scope": (
                                "real warm local HTTP route; synthetic engine and promotion "
                                "evidence; no GPU or cloud rollout"
                            ),
                            "rollback": restored.model_dump(),
                            "detection_to_health_seconds": restored.duration_seconds(),
                            "route_history": [
                                state.model_dump() for state in route_store.history("service")
                            ],
                        },
                        indent=2,
                    )
                )
    assert len(records) == 6 and sum(record["success"] for record in records) == 5
    assert len(journal) == 9


@asynccontextmanager
async def monitored_candidate(
    tmp_path: Path,
    *,
    baseline_failure: bool = False,
) -> AsyncGenerator[tuple[ProbeMonitor, BackendFixture]]:
    """Activate a real fixture route with explicitly synthetic gate identities."""
    backend_requests: list[dict[str, Any]] = []
    baseline = BackendFixture("baseline", backend_requests, tmp_path / "backends.jsonl")
    candidate = BackendFixture("candidate", backend_requests, tmp_path / "backends.jsonl")
    routes = WarmRouteStore(tmp_path / "routes.db")
    control = DeploymentStore(tmp_path / "control.db")
    registry = Registry("sqlite:///" + str(tmp_path / "journal.db"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        async with live_server(baseline.app) as good_url, live_server(candidate.app) as bad_url:
            good, bad = bound_backend("baseline", good_url), bound_backend("candidate", bad_url)
            for backend in (good, bad):
                routes.register(backend)
                control.register_revision(backend.revision)
            routes.bootstrap("service", "baseline")
            async with live_server(create_warm_app(routes, "service")) as traffic_url:
                async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
                    adapter = WarmRouteAdapter(routes, traffic_url, client)
                    control.bootstrap("service", "baseline", await adapter.health("service"))
                    await adapter.apply(
                        ApplyRequest(
                            deployment_id="service",
                            expected_revision="baseline",
                            expected_generation=0,
                            target=bad.revision,
                            idempotency_key="synthetic-activation",
                        )
                    )
                    decision = PromotionDecision(
                        candidate_revision="candidate",
                        candidate_digest=bad.revision.digest(),
                        policy_version="synthetic-control-only",
                        evidence_digest="f" * 64,
                        rejection_reasons=(),
                    )
                    control.activate_candidate(
                        "service", 0, decision, await adapter.health("service")
                    )
                    policy = MonitorPolicy(
                        monitor_id="probe-test",
                        deployment_id="service",
                        revision_id="candidate",
                        revision_digest=bad.revision.digest(),
                        generation=1,
                        maximum_probes=3,
                        consecutive_regressions=2,
                        interval_seconds=0.1,
                        slow_probe_seconds=4,
                    )
                    baseline.failed = baseline_failure
                    yield ProbeMonitor(policy, journal, control, adapter), candidate
    finally:
        registry.close()


async def test_monitor_detects_actual_http_failure_and_restores_without_manual_signal(
    tmp_path: Path,
) -> None:
    """Two actual failed streams trigger retained automatic rollback; replay issues no more work."""
    async with monitored_candidate(tmp_path) as (monitor, candidate):
        first = await monitor.observe("service")
        assert first.signal is None and first.observation.health is not None
        candidate.failed = True
        failed = await monitor.observe("service")
        assert failed.signal is None and failed.observation.health is not None
        assert not failed.observation.health.smoke_passed
        result = await monitor.run()
        assert result.signal is not None and result.rollback is not None
        assert result.rollback.status == "restored"
        assert monitor.control.deployment("service").active_revision == "baseline"
        assert monitor.adapter.store.snapshot("service").revision_id == "baseline"
        assert result.rollback.duration_seconds() is not None
        count = len(candidate.journal)
        assert await monitor.run() == result
        assert len(candidate.journal) == count
        (tmp_path / "automatic-monitor-result.json").write_text(result.model_dump_json())


async def test_healthy_probe_window_does_not_approve_or_replace_known_good(tmp_path: Path) -> None:
    """Healthy probes leave the previous known-good and quality decision intact."""
    async with monitored_candidate(tmp_path) as (monitor, _candidate):
        result = await monitor.run()
        assert result.finished and result.signal is None and result.rollback is None
        state = monitor.control.deployment("service")
        assert state.active_revision == "candidate" and state.known_good_revision == "baseline"
        assert result.observation.sequence == 2
        assert await monitor.run() == result


async def test_monitor_refuses_changed_policy_or_deployment(tmp_path: Path) -> None:
    """Restarting with changed thresholds or another deployment cannot reuse a monitor identity."""
    async with monitored_candidate(tmp_path) as (monitor, candidate):
        await monitor.observe("service")
        count = len(candidate.journal)
        with pytest.raises(ValueError, match="deployment differs"):
            await monitor.observe("different")
        changed = ProbeMonitor(
            monitor.policy.model_copy(update={"slow_probe_seconds": 3}),
            monitor.journal,
            monitor.control,
            monitor.adapter,
        )
        with pytest.raises(RegistryConflict, match="input identity changed"):
            await changed.observe("service")
        assert len(candidate.journal) == count


async def test_monitor_stops_on_stale_generation_without_probe_or_rollback(tmp_path: Path) -> None:
    """A monitor cannot label a newer candidate's traffic with an older deployment generation."""
    async with monitored_candidate(tmp_path) as (monitor, candidate):
        monitor.policy = monitor.policy.model_copy(update={"generation": 99})
        count = len(candidate.journal)
        result = await monitor.run()
        assert result.finished and result.signal is None
        assert not result.observation.route_matches and result.observation.health is None
        assert len(candidate.journal) == count
        assert monitor.control.deployment("service").rollback_id is None


@pytest.mark.parametrize("fault", ["cancel", "closure"])
async def test_uncertain_probe_stays_owned_and_cannot_blindly_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Cancellation or failed close preserves the unresolved attempt without overlapping work."""
    async with monitored_candidate(tmp_path) as (monitor, _candidate):
        entered = asyncio.Event()

        async def uncertain(_deployment_id: str) -> HealthObservation:
            """Expose the exact point where an offered health request lacks a terminal receipt."""
            entered.set()
            if fault == "closure":
                raise HTTPClosureError("synthetic close failure")
            await asyncio.Future[None]()
            raise AssertionError("unreachable")

        monkeypatch.setattr(monitor.adapter, "health", uncertain)
        task = asyncio.create_task(monitor.observe("service"))
        await entered.wait()
        if fault == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else HTTPClosureError):
            await task
        assert monitor.journal.state("probe-test:probe-000").status == "running"
        with pytest.raises(RegistryConflict, match="reconciliation"):
            await monitor.observe("service")


@pytest.mark.parametrize("baseline_failure", [False, True])
async def test_monitor_cli_runs_automatic_recovery(tmp_path: Path, baseline_failure: bool) -> None:
    """The CLI derives its signal from real HTTP failures and returns recovery state."""
    async with monitored_candidate(tmp_path, baseline_failure=baseline_failure) as (
        monitor,
        candidate,
    ):
        candidate.failed = True
        policy = tmp_path / "policy.json"
        policy.write_text(monitor.policy.model_dump_json())
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "finserve.reliability.monitor_cli",
            "--policy",
            str(policy),
            "--control",
            str(monitor.control.path),
            "--routes",
            str(monitor.adapter.store.path),
            "--journal",
            str(tmp_path / "journal.db"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--traffic-url",
            monitor.adapter.traffic_url,
            env=dict(os.environ, FINSERVE_API_KEY="synthetic-monitor-key-only"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(30):
                output, errors = await process.communicate()
            assert process.returncode == (2 if baseline_failure else 0), errors.decode()
            result = json.loads(output)
            assert result["rollback"]["status"] == ("verifying" if baseline_failure else "restored")
            assert "synthetic-monitor-key-only" not in output.decode()
            (tmp_path / "cli-result.json").write_bytes(output)
        finally:
            if process.returncode is None:
                process.kill()
            await process.wait()


async def test_monitor_healthy_probe_resets_failure_streak(tmp_path: Path) -> None:
    """A failed/healthy/failed population must not satisfy two consecutive regressions."""
    async with monitored_candidate(tmp_path) as (monitor, candidate):
        candidate.failed = True
        await monitor.observe("service")
        candidate.failed = False
        await monitor.observe("service")
        candidate.failed = True
        result = await monitor.run()
        assert result.finished and result.signal is None
        assert monitor.control.deployment("service").rollback_id is None


async def test_monitor_mixed_slow_and_failed_probes_trigger_policy(tmp_path: Path) -> None:
    """Synthetic clock injection isolates the latency rule without making a real server stall."""
    async with monitored_candidate(tmp_path) as (monitor, candidate):
        ticks = iter([0.0, 4.5, 5.0, 5.1])
        monitor.monotonic = lambda: next(ticks)
        first = await monitor.observe("service")
        assert first.observation.health is not None and first.observation.health.smoke_passed
        assert first.signal is None
        candidate.failed = True
        result = await monitor.run()
        assert result.signal is not None and result.rollback is not None
        assert result.rollback.status == "restored"


async def test_monitor_retries_unverified_recovery_with_same_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delayed healthy traffic gets another bounded verification without another route cutover."""
    async with monitored_candidate(tmp_path) as (monitor, candidate):
        candidate.failed = True
        original = monitor.adapter.health
        recovery_probes = 0

        async def delayed(deployment_id: str) -> HealthObservation:
            """Retain the first failed restoration observation, then use actual HTTP health."""
            nonlocal recovery_probes
            health = await original(deployment_id)
            if health.revision_id == "baseline":
                recovery_probes += 1
                if recovery_probes == 1:
                    return health.model_copy(update={"ready": False, "smoke_passed": False})
            return health

        monkeypatch.setattr(monitor.adapter, "health", delayed)
        result = await monitor.run()
        assert result.rollback is not None and result.rollback.status == "restored"
        assert result.rollback.apply_attempts == 1 and recovery_probes == 2
        assert len(monitor.adapter.store.history("service")) == 3
