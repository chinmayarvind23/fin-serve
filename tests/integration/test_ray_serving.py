"""Opt-in real Ray Serve process integration; these tests make no performance claims."""

import asyncio
import importlib
import json
import os
import socket
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
import pytest

from finserve.contracts.inference import InferenceRequest
from finserve.engines.ray_serve import RayServeEngine, build_fixture_application

pytestmark = pytest.mark.skipif(
    os.environ.get("FINSERVE_RUN_RAY_INTEGRATION") != "1",
    reason="Set FINSERVE_RUN_RAY_INTEGRATION=1 in the isolated Ray environment",
)


@dataclass
class RayFixtureRuntime:
    """Confine optional Ray dynamic handles to the real-process integration boundary."""

    handle: Any
    url: str


@pytest.fixture(scope="module")
def runtime() -> Generator[RayFixtureRuntime, None, None]:
    """Start an isolated localhost CPU cluster and always tear down its actors and HTTP proxy."""
    ray = importlib.import_module("ray")
    serve = importlib.import_module("ray.serve")
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
        namespace=f"finserve-integration-{uuid4().hex}",
    )
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": port})
        handle = serve.run(
            build_fixture_application(token_delay_seconds=0.02),
            name="finserve-routing-fixture",
            route_prefix="/fixture",
        )
        yield RayFixtureRuntime(handle=handle, url=f"http://127.0.0.1:{port}/fixture")
    finally:
        serve.shutdown()
        ray.shutdown()


@pytest.mark.asyncio
async def test_real_ray_routes_concurrent_streams_to_two_workers(
    runtime: RayFixtureRuntime,
) -> None:
    """Verify actual actor routing and streamed payloads across HTTP and Ray handle boundaries."""
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:

        async def collect(prompt: str) -> list[dict[str, Any]]:
            """Read incrementally so buffering cannot replace the streaming integration path."""
            request = InferenceRequest(model="fixture", prompt=prompt, max_tokens=8)
            async with client.stream("POST", runtime.url, json=request.model_dump()) as response:
                response.raise_for_status()
                return [json.loads(line) async for line in response.aiter_lines() if line]

        first, second = await asyncio.gather(collect("abcdefghij"), collect("ABCDEFGHIJ"))
        assert "".join(item["token"]["text"] for item in first) == "abcdefgh"
        assert "".join(item["token"]["text"] for item in second) == "ABCDEFGH"
        assert {first[0]["replica_id"], second[0]["replica_id"]} == {"fixture-a", "fixture-b"}
        status = (await client.get(runtime.url)).json()
        assert status["active_reservations"] == 0
        assert all(worker["ongoing_requests"] == 0 for worker in status["workers"])


@pytest.mark.asyncio
async def test_real_ray_http_disconnect_drains_worker_and_router(
    runtime: RayFixtureRuntime,
) -> None:
    """Closing an unfinished HTTP stream must cancel the worker and release its routing lease."""
    async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
        request = InferenceRequest(model="fixture", prompt="x" * 100, max_tokens=100)
        async with client.stream("POST", runtime.url, json=request.model_dump()) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    assert json.loads(line)["token"]["text"] == "x"
                    break
        deadline = time.monotonic() + 10
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = (await client.get(runtime.url)).json()
            idle = status["active_reservations"] == 0 and all(
                worker["ongoing_requests"] == 0 for worker in status["workers"]
            )
            if idle and sum(worker["cancelled"] for worker in status["workers"]) >= 1:
                return
            await asyncio.sleep(0.05)
        pytest.fail(f"Ray disconnect did not drain ownership: {status}")


def test_real_ray_streaming_handle_cancellation(runtime: RayFixtureRuntime) -> None:
    """Exercise Ray's public cancel method independently of HTTP client-disconnect detection."""
    before = runtime.handle.status.remote().result(timeout_s=5)
    prior_cancelled = sum(worker["cancelled"] for worker in before["workers"])
    request = InferenceRequest(model="fixture", prompt="y" * 100, max_tokens=100)
    stream = runtime.handle.options(stream=True).generate.remote(request.model_dump())
    assert next(stream)["token"]["text"] == "y"
    stream.cancel()
    deadline = time.monotonic() + 10
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = runtime.handle.status.remote().result(timeout_s=5)
        if status["active_reservations"] == 0 and all(
            worker["ongoing_requests"] == 0 for worker in status["workers"]
        ):
            assert sum(worker["cancelled"] for worker in status["workers"]) > prior_cancelled
            return
        time.sleep(0.05)
    pytest.fail(f"Ray handle cancellation did not drain ownership: {status}")


@pytest.mark.asyncio
async def test_real_ray_engine_adapter(runtime: RayFixtureRuntime) -> None:
    """Verify the ordinary Engine protocol through an actual deployed routing handle."""
    engine = RayServeEngine(runtime.handle)
    request = InferenceRequest(model="fixture", prompt="adapter", max_tokens=4)
    tokens = [token async for token in engine.stream(request)]
    assert "".join(token.text for token in tokens) == "adap"
    assert sum(token.generated_tokens for token in tokens) == 4
    await engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        await anext(engine.stream(request))
