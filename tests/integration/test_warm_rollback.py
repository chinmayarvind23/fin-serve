"""Real loopback HTTP rollback drill; synthetic engine failures never touch existing services."""

import asyncio
import json
import socket
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from finserve.benchmark.metrics import RequestRecord
from finserve.benchmark.runner import RunConfig, request_one
from finserve.benchmark.workload import WorkItem
from finserve.contracts.deployment import RegressionSignal, Revision
from finserve.gateway.warm_route_app import create_warm_app
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
