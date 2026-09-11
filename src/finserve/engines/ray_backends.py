"""Ray request routing over separately managed OpenAI-compatible engine processes.

Proxy actors own zero GPUs. Engine containers/processes own device allocation,
batching and KV resources; closing HTTP is not proof that their GPU work drained.
"""

import asyncio
import importlib
import json
import os
import secrets
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, Literal, Self, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.engines.openai_adapter import OpenAICompletionEngine
from finserve.engines.ray_serve import FixtureReplica, OwnedFixtureResponse, RoutedFixture
from finserve.scheduler.policy import RoutingPolicy
from finserve.scheduler.router import ReplicaRouter


class BackendConfiguration(BaseModel):
    """Trusted deployment configuration names actual endpoints, not caller-selected destinations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1, max_length=256)
    backends: dict[str, str] = Field(min_length=1, max_length=8)
    capacity_per_worker: int = Field(default=16, ge=1, le=128, strict=True)
    mode: Literal["least_load", "adaptive"] = "least_load"

    @model_validator(mode="after")
    def validate_endpoints(self) -> Self:
        """Reject duplicate canonical endpoints that would double-count one configured worker."""
        normalized: set[str] = set()
        for name, address in self.backends.items():
            if not name or len(name) > 63 or not all(c.isalnum() or c in "-_" for c in name):
                raise ValueError("invalid backend name")
            url = httpx.URL(address)
            if url.scheme not in {"http", "https"} or not url.host or url.username or url.password:
                raise ValueError("invalid backend endpoint")
            if url.query or url.fragment:
                raise ValueError("backend endpoints cannot contain queries or fragments")
            if "\\" in address or b"%" in url.raw_path:
                raise ValueError("backend paths cannot contain encoded or backslash aliases")
            identity = str(url).rstrip("/")
            if identity in normalized:
                raise ValueError("each backend must have a distinct endpoint")
            normalized.add(identity)
        return self


class OpenAIEngineReplica(FixtureReplica):
    """Reuse fenced lease ownership while the external process performs real model inference."""

    def __init__(self, replica_id: str, capacity: int, model: str, base_url: str) -> None:
        """Resolve the engine credential from the trusted worker environment."""
        super().__init__(replica_id, capacity, 0)
        self.model = model
        self.engine = OpenAICompletionEngine(base_url, api_key=os.getenv("FINSERVE_ENGINE_API_KEY"))
        self.health_url = base_url.rstrip("/") + "/models"
        headers: dict[str, str] = {}
        if key := os.getenv("FINSERVE_ENGINE_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        self.probe = httpx.AsyncClient(
            headers=headers, timeout=1, trust_env=False, follow_redirects=False
        )
        self._health_lock = asyncio.Lock()
        self._healthy = False
        self._checked_at: float | None = None
        self._close_task: asyncio.Task[None] | None = None

    async def tokens(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """Explicit closure propagates Ray cancellation through upstream HTTP."""
        async with aclosing(self.engine.stream(request)) as output:
            async for token in output:
                yield token

    async def _probe_health(self) -> bool:
        """Bound discovery bytes and time; readiness requires the configured model ID."""
        async with asyncio.timeout(1):
            async with self.probe.stream("GET", self.health_url) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > 131072:
                        raise ValueError("model discovery too large")
                    body.extend(chunk)
        document = json.loads(body)
        return any(row.get("id") == self.model for row in document["data"])

    async def state(self) -> dict[str, Any]:
        """Cache discovery for at most one second; acknowledged request counts remain current."""
        if self._close_task is not None:
            return super().snapshot() | {"healthy": False}
        async with self._health_lock:
            if self._checked_at is None or time.monotonic() - self._checked_at >= 1:
                try:
                    self._healthy = await self._probe_health()
                except Exception:
                    self._healthy = False
                self._checked_at = time.monotonic()
        result = super().snapshot()
        result["healthy"] = self._healthy
        return result

    async def close(self) -> None:
        """Drain proxy requests and close pools; external processes remain separately owned."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_pools())
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                continue
        self._close_task.result()

    async def _close_pools(self) -> None:
        """Pool closure is guaranteed even if proxy retirement fails during graceful shutdown."""
        try:
            for lease in tuple(self.active):
                await self.retire(lease)
        finally:
            try:
                await self.engine.close()
            finally:
                await self.probe.aclose()


class ServeOpenAIEngineReplica(OpenAIEngineReplica):
    """Confine Ray's awaited destructor convention to actors, not ordinary Python instances."""

    async def __del__(self) -> None:
        """Ray Serve explicitly awaits this hook on graceful deletion; hard kills cannot drain."""
        await self.close()


class RoutedBackends(RoutedFixture):
    """A single routing authority addresses one named proxy actor per distinct external backend."""

    def __init__(
        self, handles: dict[str, Any], model: str, mode: Literal["least_load", "adaptive"]
    ) -> None:
        """One actor owns routing reservations; external engines scale separately."""
        self.handles, self.model = handles, model
        self.router = ReplicaRouter(RoutingPolicy(mode=mode))
        self._api_key = os.getenv("FINSERVE_RAY_API_KEY")

    async def __call__(self, request: Request) -> Response:
        """Bound the optional HTTP ingress before creating any lease or contacting an engine."""
        credentials = request.headers.getlist("authorization")
        if self._api_key and (
            len(credentials) != 1
            or not secrets.compare_digest(
                credentials[0].encode(), f"Bearer {self._api_key}".encode()
            )
        ):
            return JSONResponse(
                {"detail": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
        if request.method == "GET":
            return JSONResponse(await self.status())
        if request.method != "POST":
            return JSONResponse({"detail": "method not allowed"}, status_code=405)
        body = bytearray()
        try:
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > 131072:
                        return JSONResponse({"detail": "request body too large"}, status_code=413)
                    body.extend(chunk)
            payload = InferenceRequest.model_validate_json(body).model_dump()
        except TimeoutError:
            return JSONResponse({"detail": "request body timed out"}, status_code=408)
        except ValueError:
            return JSONResponse({"detail": "invalid inference request"}, status_code=422)
        return OwnedFixtureResponse(self._http_lines(payload))

    async def read_snapshot(self, handle: Any) -> dict[str, Any]:
        """The engine heartbeat adds model discovery to current local lease counts."""
        return cast(dict[str, Any], await handle.state.remote())

    async def status(self) -> dict[str, Any]:
        """Report actual HTTP proxy occupancy without inventing remote GPU measurements."""
        workers = await asyncio.gather(*(handle.state.remote() for handle in self.handles.values()))
        return {
            "active_reservations": self.router.active_reservations,
            "workers": workers,
            "scope": "CPU proxy leases; GPU resources belong to external engines",
        }


def build_application(args: dict[str, Any]) -> Any:
    """Public Ray Serve application builder; KubeRay passes the same validated args mapping."""
    configuration = BackendConfiguration.model_validate(args)
    serve = importlib.import_module("ray.serve")
    workers = {
        name: serve.deployment(
            name=name,
            num_replicas=1,
            max_ongoing_requests=configuration.capacity_per_worker + 4,
            # Health/admit/retire share this bounded control queue; compute has its own lease cap.
            max_queued_requests=128,
            ray_actor_options={"num_cpus": 0.1, "num_gpus": 0},
        )(ServeOpenAIEngineReplica).bind(
            name, configuration.capacity_per_worker, configuration.model, endpoint
        )
        for name, endpoint in configuration.backends.items()
    }
    return serve.deployment(
        name="engine-routing",
        num_replicas=1,
        max_ongoing_requests=128,
        max_queued_requests=128,
        ray_actor_options={"num_cpus": 0.1, "num_gpus": 0},
    )(RoutedBackends).bind(workers, configuration.model, configuration.mode)
