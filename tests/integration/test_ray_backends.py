"""Opt-in real Ray routing over real local HTTP streams; all backends are CPU fixtures."""

import asyncio
import importlib
import json
import os
import socket
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest

from finserve.contracts.inference import InferenceRequest
from finserve.engines.ray_backends import (
    RoutedBackends,
    ServeOpenAIEngineReplica,
    build_application,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("FINSERVE_RUN_RAY_INTEGRATION") != "1",
    reason="Set FINSERVE_RUN_RAY_INTEGRATION=1 in the isolated CPU Ray environment",
)
MODEL = "cpu-http-exact-model"


class BackendServer(ThreadingHTTPServer):
    """Keep independently observable HTTP connections without sharing any actor internals."""

    daemon_threads = True

    def __init__(self, marker: str) -> None:
        """One localhost listener represents one separately managed upstream engine endpoint."""
        super().__init__(("127.0.0.1", 0), BackendHandler)
        self.marker = marker
        self.model = MODEL
        self.health_delay = 0.0
        self.metrics_valid = True
        self.lock = threading.Lock()
        self.active = 0
        self.completed = 0
        self.disconnected = 0
        self.requests = 0

    @property
    def endpoint(self) -> str:
        """Publish only the unique loopback endpoint selected by the operating system."""
        return f"http://127.0.0.1:{self.server_port}/v1"


class BackendHandler(BaseHTTPRequestHandler):
    """Emit standard bounded OpenAI SSE over sockets, with independent disconnect accounting."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Keep the fixture from logging request bodies or noisy per-token connection messages."""

    def do_GET(self) -> None:
        """Advertise a mutable model ID so exact-model health rejection can be tested live."""
        backend = cast(BackendServer, self.server)
        time.sleep(backend.health_delay)
        if self.path == "/metrics":
            body = (
                "".join(
                    f'# TYPE vllm:{name} gauge\n'
                    f'vllm:{name}{{model_name="{backend.model}"}} {value}\n'
                    for name, value in (
                        ("num_requests_running", backend.active),
                        ("num_requests_waiting", 0),
                        ("kv_cache_usage_perc", 0.125),
                    )
                ).encode()
                if backend.metrics_valid
                else b"missing gauges"
            )
        else:
            body = json.dumps({"data": [{"id": backend.model}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _event(self, document: object) -> None:
        """Flush each SSE frame so the downstream path must handle incremental network data."""
        self.wfile.write(f"data: {json.dumps(document)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self) -> None:
        """Record real socket abandonment separately from proxy routing-slot release."""
        backend = cast(BackendServer, self.server)
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/completions" and payload["model"] == MODEL
        count = min(payload["max_tokens"], 64)
        with backend.lock:
            backend.active += 1
            backend.requests += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        complete = False
        try:
            for _ in range(count):
                self._event(
                    {"choices": [{"index": 0, "text": backend.marker, "finish_reason": None}]}
                )
                time.sleep(0.05)
            self._event({"choices": [{"index": 0, "text": "", "finish_reason": "length"}]})
            self._event({"choices": [], "usage": {"completion_tokens": count}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            complete = True
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with backend.lock:
                backend.active -= 1
                backend.completed += int(complete)
                backend.disconnected += int(not complete)
            self.close_connection = True


@dataclass
class Runtime:
    """Keep control handles and upstream observations scoped to this disposable CPU cluster."""

    serve: Any
    backends: tuple[BackendServer, BackendServer]
    handle: Any
    url: str


@pytest.fixture(scope="module")
def runtime() -> Generator[Runtime, None, None]:
    """Start two local HTTP fixtures and one isolated Ray cluster; never attach to another task."""
    ray = importlib.import_module("ray")
    serve = importlib.import_module("ray.serve")
    backends = (BackendServer("a"), BackendServer("b"))
    threads = [threading.Thread(target=backend.serve_forever, daemon=True) for backend in backends]
    for thread in threads:
        thread.start()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    ray.init(
        address="local",
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=128 * 1024 * 1024,
        namespace=f"finserve-http-bridge-{uuid4().hex}",
    )
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": port})
        handle = serve.run(
            build_application(
                {
                    "model": MODEL,
                    "capacity_per_worker": 1,
                    "backends": {"a": backends[0].endpoint, "b": backends[1].endpoint},
                }
            ),
            name="backend-pair",
            route_prefix="/backends",
        )
        yield Runtime(serve, backends, handle, f"http://127.0.0.1:{port}/backends")
    finally:
        serve.shutdown()
        ray.shutdown()
        for backend in backends:
            backend.shutdown()
            backend.server_close()
        for thread in threads:
            thread.join(2)


async def wait_idle(runtime: Runtime) -> dict[str, Any]:
    """Observe HTTP and router ownership separately; socket drain is not GPU drain evidence."""
    for _ in range(200):
        state = cast(dict[str, Any], await runtime.handle.status.remote())
        if (
            state["active_reservations"] == 0
            and all(worker["ongoing_requests"] == 0 for worker in state["workers"])
            and all(backend.active == 0 for backend in runtime.backends)
        ):
            return state
        await asyncio.sleep(0.05)
    raise AssertionError("HTTP/backend/router ownership did not drain")


async def test_two_real_http_workers_route_streams_and_authoritative_usage(
    runtime: Runtime,
) -> None:
    """Two concurrent streams must visit distinct real HTTP servers and preserve usage."""
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:

        async def collect() -> list[dict[str, Any]]:
            """Read real JSON-line envelopes incrementally through the public Ray HTTP route."""
            request = InferenceRequest(model=MODEL, prompt="public fixture", max_tokens=8)
            async with client.stream("POST", runtime.url, json=request.model_dump()) as response:
                response.raise_for_status()
                return [json.loads(line) async for line in response.aiter_lines() if line]

        first, second = await asyncio.gather(collect(), collect())
    assert {first[0]["replica_id"], second[0]["replica_id"]} == {"a", "b"}
    for output in (first, second):
        assert sum(frame["token"]["generated_tokens"] for frame in output) == 8
        assert len("".join(frame["token"]["text"] for frame in output)) == 8
        assert output[-1]["token"]["finish_reason"] == "length"
    assert (await wait_idle(runtime))["active_reservations"] == 0


async def test_disconnect_after_partial_output_closes_http_and_proxy_slots(
    runtime: Runtime,
) -> None:
    """Client abandonment after visible output must drain both real HTTP and proxy reservations."""
    before = sum(backend.disconnected for backend in runtime.backends)
    request = InferenceRequest(model=MODEL, prompt="cancel", max_tokens=64)
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        async with client.stream("POST", runtime.url, json=request.model_dump()) as response:
            async for line in response.aiter_lines():
                if line:
                    assert json.loads(line)["token"]["text"] in {"a", "b"}
                    break
    state = await wait_idle(runtime)
    assert sum(backend.disconnected for backend in runtime.backends) > before
    assert sum(worker["cancelled"] for worker in state["workers"]) >= 1


async def test_wrong_model_and_unreachable_endpoint_fail_closed(runtime: Runtime) -> None:
    """A live wrong model and an unreachable peer must both fail before sending generation."""
    bad = runtime.backends[0]
    original, bad.model = bad.model, "different-model"
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        unreachable = unused.getsockname()[1]
    before = bad.requests
    handle = await asyncio.to_thread(
        runtime.serve.run,
        build_application(
            {
                "model": MODEL,
                "backends": {"wrong": bad.endpoint, "down": f"http://127.0.0.1:{unreachable}/v1"},
            }
        ),
        name="unhealthy-backends",
        route_prefix="/unhealthy",
    )
    try:
        stream = handle.options(stream=True).generate.remote(
            InferenceRequest(model=MODEL, prompt="unhealthy").model_dump()
        )
        with pytest.raises(Exception, match="NoReplicaAvailable"):
            _ = [item async for item in stream]
        state = await handle.status.remote()
        assert state["active_reservations"] == 0
        assert all(worker["healthy"] is False for worker in state["workers"])
        assert bad.requests == before
    finally:
        bad.model = original
        await asyncio.to_thread(runtime.serve.delete, "unhealthy-backends")


async def test_single_http_worker_and_slow_health_deadline(runtime: Runtime) -> None:
    """A single endpoint is useful, and discovery time consumes the same request deadline."""
    backend = runtime.backends[0]
    handle = await asyncio.to_thread(
        runtime.serve.run,
        build_application({"model": MODEL, "backends": {"solo": backend.endpoint}}),
        name="single-backend",
        route_prefix="/single",
    )
    try:
        request = InferenceRequest(model=MODEL, prompt="single", max_tokens=2)
        stream = handle.options(stream=True).generate.remote(request.model_dump())
        output = [item async for item in stream]
        assert sum(item["token"]["generated_tokens"] for item in output) == 2
        await asyncio.sleep(1.05)
        backend.health_delay = 0.5
        before = backend.requests
        stream = handle.options(stream=True).generate.remote(
            request.model_copy(update={"timeout_seconds": 0.1}).model_dump()
        )
        with pytest.raises(Exception, match="TimeoutError"):
            _ = [item async for item in stream]
        assert backend.requests == before
    finally:
        backend.health_delay = 0
        await asyncio.to_thread(runtime.serve.delete, "single-backend")


async def test_slow_admission_is_revoked_before_generation(runtime: Runtime) -> None:
    """An admission RPC finishing after budget expiry must be retired without starting HTTP work."""

    class SlowReplica(ServeOpenAIEngineReplica):
        """Add controlled Ray admission latency while preserving the real HTTP adapter."""

        async def admit(self, lease_id: str) -> None:  # type: ignore[override]
            """Represent queued actor work instead of an instantaneous reservation."""
            await asyncio.sleep(0.3)
            super().admit(lease_id)

    serve = runtime.serve
    worker = serve.deployment(name="slow", ray_actor_options={"num_cpus": 0.1})(SlowReplica).bind(
        "slow", 1, MODEL, runtime.backends[0].endpoint
    )
    app = serve.deployment(name="slow-router", ray_actor_options={"num_cpus": 0.1})(
        RoutedBackends
    ).bind({"slow": worker}, MODEL, "least_load")
    handle = await asyncio.to_thread(serve.run, app, name="slow-admission", route_prefix="/slow")
    before = runtime.backends[0].requests
    try:
        stream = handle.options(stream=True).generate.remote(
            InferenceRequest(model=MODEL, prompt="budget", timeout_seconds=0.1).model_dump()
        )
        with pytest.raises(Exception, match="TimeoutError"):
            _ = [item async for item in stream]
        state = await handle.status.remote()
        assert state["active_reservations"] == 0
        assert state["workers"][0]["ongoing_requests"] == 0
        assert runtime.backends[0].requests == before
    finally:
        await asyncio.to_thread(serve.delete, "slow-admission")


async def test_serve_delete_awaits_actual_destructor(runtime: Runtime, tmp_path: Path) -> None:
    """Observe Serve awaiting async pool close before deleting the actual actor."""
    marker = tmp_path / "actor-closed.txt"

    class ObservedReplica(ServeOpenAIEngineReplica):
        """Use a test-only artifact to distinguish awaited close from abrupt process socket loss."""

        async def close(self) -> None:
            """Mark completion only after both real HTTPX pools have closed."""
            await super().close()
            marker.write_text(str(self.probe.is_closed))

    worker = runtime.serve.deployment(name="observed", ray_actor_options={"num_cpus": 0.1})(
        ObservedReplica
    ).bind("observed", 1, MODEL, runtime.backends[0].endpoint)
    handle = await asyncio.to_thread(
        runtime.serve.run, worker, name="destructor-test", route_prefix=None
    )
    assert (await handle.state.remote())["healthy"] is True
    await asyncio.to_thread(runtime.serve.delete, "destructor-test")
    assert marker.read_text() == "True"


async def test_live_http_auth_and_body_validation_before_routing(runtime: Runtime) -> None:
    """Internal HTTP credentials and byte/schema bounds apply before upstream work is admitted."""
    key = "ray-http-integration-only-key"
    serve = runtime.serve
    worker = serve.get_deployment_handle("a", app_name="backend-pair")
    app = serve.deployment(
        name="authenticated-router",
        ray_actor_options={
            "num_cpus": 0.1,
            "runtime_env": {"env_vars": {"FINSERVE_RAY_API_KEY": key}},
        },
    )(RoutedBackends).bind({"a": worker}, MODEL, "least_load")
    await asyncio.to_thread(serve.run, app, name="authenticated-router", route_prefix="/secured")
    url = runtime.url.removesuffix("/backends") + "/secured"
    before = sum(backend.requests for backend in runtime.backends)
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            assert (await client.get(url)).status_code == 401
            assert (await client.post(url, json={})).status_code == 401
            headers = {"Authorization": f"Bearer {key}"}
            assert (await client.get(url, headers=headers)).status_code == 200
            assert (await client.put(url, headers=headers)).status_code == 405
            assert (await client.post(url, headers=headers, content=b"bad-json")).status_code == 422
            assert (
                await client.post(url, headers=headers, content=b"x" * 131073)
            ).status_code == 413
            assert sum(backend.requests for backend in runtime.backends) == before
            request = InferenceRequest(model=MODEL, prompt="authorized", max_tokens=2)
            response = await client.post(url, headers=headers, json=request.model_dump())
            response.raise_for_status()
            frames = [json.loads(line) for line in response.text.splitlines()]
            assert sum(frame["token"]["generated_tokens"] for frame in frames) == 2
    finally:
        await asyncio.to_thread(serve.delete, "authenticated-router")


async def test_real_http_engine_metrics_and_missing_observation_quarantine(
    runtime: Runtime,
) -> None:
    """Real socket gauges drive readiness; model discovery alone cannot hide failed telemetry."""
    handle = await asyncio.to_thread(
        runtime.serve.run,
        build_application(
            {
                "model": MODEL,
                "backends": {
                    "observed-a": runtime.backends[0].endpoint,
                    "observed-b": runtime.backends[1].endpoint,
                },
                "observe_engine_metrics": True,
                "capacity_per_worker": 4,
            }
        ),
        name="metrics-pair",
        route_prefix=None,
    )
    try:
        status = await handle.status.remote()
        assert status["physical_gpu_observation"] is None
        assert all(
            row["healthy"] and row["kv_cache_utilization"] == 0.125 for row in status["workers"]
        )
        runtime.backends[0].metrics_valid = False
        await asyncio.sleep(1.05)
        request = InferenceRequest(model=MODEL, prompt="gauges", max_tokens=2)
        stream = handle.options(stream=True).generate.remote(request.model_dump())
        frames = [frame async for frame in stream]
        assert frames and all(frame["replica_id"] == "observed-b" for frame in frames)
        status = await handle.status.remote()
        assert status["active_reservations"] == 0
        assert not next(row for row in status["workers"] if row["replica_id"] == "observed-a")[
            "healthy"
        ]
    finally:
        runtime.backends[0].metrics_valid = True
        await asyncio.to_thread(runtime.serve.delete, "metrics-pair")
