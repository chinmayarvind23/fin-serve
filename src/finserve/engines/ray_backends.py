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
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any, Literal, Self, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.engines.backend_observations import (
    EngineObservation,
    SharedGpuSampler,
    parse_engine_metrics,
)
from finserve.engines.openai_adapter import OpenAICompletionEngine
from finserve.engines.ray_serve import FixtureReplica, OwnedFixtureResponse, RoutedFixture
from finserve.scheduler.policy import RoutingPolicy
from finserve.scheduler.router import (
    NoReplicaAvailable,
    ReplicaCapacityUnavailable,
    ReplicaRouter,
    RoutingLease,
)
from finserve.telemetry.tracing import from_env as tracing_from_env


class BackendConfiguration(BaseModel):
    """Trusted deployment configuration names actual endpoints, not caller-selected destinations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1, max_length=256)
    backends: dict[str, str] = Field(min_length=1, max_length=8)
    capacity_per_worker: int = Field(default=16, ge=1, le=128, strict=True)
    mode: Literal["least_load", "adaptive"] = "least_load"
    observe_engine_metrics: bool = False
    vllm_structured_outputs: bool = False
    shared_gpu_uuid: str | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_endpoints(self) -> Self:
        """Reject duplicate canonical endpoints that would double-count one configured worker."""
        normalized: set[str] = set()
        if self.shared_gpu_uuid is not None and not self.observe_engine_metrics:
            raise ValueError("physical GPU routing requires backend engine observations")
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

    def __init__(
        self,
        replica_id: str,
        capacity: int,
        model: str,
        base_url: str,
        observe_engine_metrics: bool = False,
        vllm_structured_outputs: bool = False,
    ) -> None:
        """Resolve the engine credential from the trusted worker environment."""
        from finserve.auth import require_credentials

        require_credentials("FINSERVE_ENGINE_API_KEY")
        super().__init__(replica_id, capacity, 0)
        self.model = model
        from finserve.engines.vllm_adapter import VLLMEngine

        adapter = VLLMEngine if vllm_structured_outputs else OpenAICompletionEngine
        self.engine = adapter(base_url, api_key=os.getenv("FINSERVE_ENGINE_API_KEY"))
        self.health_url = base_url.rstrip("/") + "/models"
        self.metrics_url = str(httpx.URL(base_url).copy_with(path="/metrics"))
        self.observe_engine_metrics = observe_engine_metrics
        self._observation: EngineObservation | None = None
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
        # A batch exporter owns a thread; allocate it after validating HTTP resources.
        self.tracing = tracing_from_env(component="engine")
        if self.tracing is not None:
            self.tracer = self.tracing.tracer

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

    async def _probe_metrics(self) -> EngineObservation:
        """Use bounded current vLLM gauges; deployments with other metric schemas opt out."""
        async with asyncio.timeout(1):
            async with self.probe.stream("GET", self.metrics_url) as response:
                response.raise_for_status()
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > 262144:
                        raise ValueError("engine metrics exceed byte limit")
                    body.extend(chunk)
        return parse_engine_metrics(bytes(body), self.model)

    async def state(self) -> dict[str, Any]:
        """Cache discovery for at most one second; acknowledged request counts remain current."""
        if self._close_task is not None:
            return super().snapshot() | {"healthy": False}
        async with self._health_lock:
            if self._checked_at is None or time.monotonic() - self._checked_at >= 1:
                self._checked_at = time.monotonic()
                try:
                    if self.observe_engine_metrics:
                        async with asyncio.TaskGroup() as group:
                            health = group.create_task(self._probe_health())
                            metrics = group.create_task(self._probe_metrics())
                        self._healthy, self._observation = health.result(), metrics.result()
                    else:
                        self._healthy = await self._probe_health()
                except Exception:
                    self._healthy = False
                    self._observation = None
        result = super().snapshot()
        result["healthy"] = self._healthy
        if self._observation is not None:
            observed = self._observation
            # Proxy ownership and backend counts overlap; summing would double-count our calls.
            # External direct calls affect load, but cannot share our lease fence. The engine's
            # compute cap remains authoritative; proxy capacity assumes exclusive routed traffic.
            result["ongoing_requests"] = max(len(self.active), observed.running + observed.waiting)
            result.update(
                engine_running_requests=observed.running,
                engine_waiting_requests=observed.waiting,
                kv_cache_utilization=observed.kv_cache_utilization,
                engine_observation_age_seconds=time.monotonic() - self._checked_at,
            )
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
                try:
                    await self.probe.aclose()
                finally:
                    if self.tracing is not None:
                        await asyncio.to_thread(self.tracing.close)


class ServeOpenAIEngineReplica(OpenAIEngineReplica):
    """Confine Ray's awaited destructor convention to actors, not ordinary Python instances."""

    async def __del__(self) -> None:
        """Ray Serve explicitly awaits this hook on graceful deletion; hard kills cannot drain."""
        await self.close()


class RoutedBackends(RoutedFixture):
    """A single routing authority addresses one named proxy actor per distinct external backend."""

    def __init__(
        self,
        handles: dict[str, Any],
        model: str,
        mode: Literal["least_load", "adaptive"],
        shared_gpu_uuid: str | None = None,
    ) -> None:
        """One actor owns routing reservations; external engines scale separately."""
        from finserve.auth import require_credentials

        require_credentials("FINSERVE_RAY_API_KEY")
        self.handles, self.model = handles, model
        self.router = ReplicaRouter(RoutingPolicy(mode=mode))
        from opentelemetry.trace import NoOpTracer

        self.tracing = tracing_from_env(component="route")
        self.tracer = self.tracing.tracer if self.tracing is not None else NoOpTracer()
        self._close_task: asyncio.Task[None] | None = None
        self._api_key = os.getenv("FINSERVE_RAY_API_KEY")
        self.shared_gpu_uuid = shared_gpu_uuid
        self._gpu_sampler = SharedGpuSampler() if shared_gpu_uuid is not None else None
        self._disabled: set[str] = set()
        self._pending_admissions = 0
        self._admission_wait_count = 0
        self._admission_wait_seconds = 0.0
        self._decisions: deque[dict[str, Any]] = deque(maxlen=256)

    def set_enabled(self, replica_id: str, enabled: bool) -> None:
        """Trusted handle control quarantines before owned engine drain; it never revokes leases."""
        if replica_id not in self.handles or type(enabled) is not bool:
            raise ValueError("unknown backend or invalid enabled flag")
        if enabled:
            self._disabled.discard(replica_id)
        else:
            self._disabled.add(replica_id)
            for snapshot in self.router.snapshots:
                if snapshot.replica_id == replica_id:
                    self.router.update_snapshot(
                        snapshot.model_copy(
                            update={
                                "healthy": False,
                                "received_at": time.monotonic(),
                            }
                        )
                    )

    def record_selection(self, request: InferenceRequest, lease: RoutingLease) -> None:
        """Retain bounded request IDs and actual policy inputs without prompts or credentials."""
        self._decisions.append(
            {
                "request_id": request.request_id,
                "epoch_s": time.time(),
                "decision": lease.decision.model_dump(),
                "snapshots": [
                    snapshot.model_dump(mode="json") for snapshot in self.router.snapshots
                ],
            }
        )

    async def reserve_request(self, request: InferenceRequest, deadline: float) -> RoutingLease:
        """Wait only for temporary saturation, within receipt budget and a fixed waiter cap.

        No lease or worker invocation exists during this wait. Cached native gauges remain
        conservative; refreshing after completion lets them expire without inventing capacity.
        Healthy eligibility is required on every iteration. This is not a fairness guarantee.
        """
        if self._pending_admissions >= 128:
            raise NoReplicaAvailable("Routing admission wait limit reached")
        self._pending_admissions += 1
        started = time.monotonic()
        waited = False
        try:
            while True:
                try:
                    return await super().reserve_request(request, deadline)
                except ReplicaCapacityUnavailable:
                    if not waited:
                        self._admission_wait_count += 1
                        waited = True
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Routing admission deadline exceeded") from None
                    await asyncio.sleep(min(0.05, remaining))
        finally:
            self._pending_admissions -= 1
            if waited:
                self._admission_wait_seconds += time.monotonic() - started

    def admission_statistics(self) -> dict[str, int | float]:
        """Expose fixed-cardinality counters without contacting workers or retaining request IDs."""
        return {
            "pending_admissions": self._pending_admissions,
            "admission_wait_count": self._admission_wait_count,
            "admission_wait_seconds": self._admission_wait_seconds,
        }

    async def refresh_snapshots(self) -> None:
        """Attach one shared-device sample per refresh, preventing invented per-engine VRAM."""
        fields: dict[str, object] = {}
        if self._gpu_sampler is not None and self.shared_gpu_uuid is not None:
            observation = await self._gpu_sampler.get()
            fields = observation.snapshot_fields(self.shared_gpu_uuid)
        await asyncio.gather(
            *(
                self._refresh_one(
                    name, handle, fields | ({"healthy": False} if name in self._disabled else {})
                )
                for name, handle in self.handles.items()
            )
        )

    async def close(self) -> None:
        """Drain only router-owned native sampling; engine process ownership stays external."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_resources())
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                continue
        self._close_task.result()

    async def _close_resources(self) -> None:
        """Exporter shutdown is native work; one retained task owns it through cancellation."""
        try:
            if self._gpu_sampler is not None:
                await self._gpu_sampler.close()
        finally:
            if self.tracing is not None:
                await asyncio.to_thread(self.tracing.close)

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
        parents = request.headers.getlist("traceparent")
        parent = parents[0] if len(parents) == 1 else None
        return OwnedFixtureResponse(self._http_lines(payload, parent))

    async def read_snapshot(self, handle: Any) -> dict[str, Any]:
        """The engine heartbeat adds model discovery to current local lease counts."""
        data = dict(cast(dict[str, Any], await handle.state.remote()))
        if data.get("replica_id") in self._disabled:
            data["healthy"] = False
        return data

    async def status(self) -> dict[str, Any]:
        """Report actual HTTP proxy occupancy without inventing remote GPU measurements."""
        workers = await asyncio.gather(*(handle.state.remote() for handle in self.handles.values()))
        physical: dict[str, object] | None = None
        if self._gpu_sampler is not None:
            observation = await self._gpu_sampler.get()
            physical = observation.sample.model_dump()
            physical["router_observed_at"] = observation.started_at
            physical["configured_shared_uuid"] = self.shared_gpu_uuid
        return {
            "active_reservations": self.router.active_reservations,
            **self.admission_statistics(),
            "workers": workers,
            "physical_gpu_observation": physical,
            "disabled_backends": sorted(self._disabled),
            "recent_decisions": list(self._decisions),
            "scope": "CPU proxy leases; GPU resources belong to external engines",
        }


class ServeRoutedBackends(RoutedBackends):
    """Keep Ray's awaited destructor limited to the actor lifecycle boundary."""

    async def __del__(self) -> None:
        """Graceful actor deletion drains the router-owned collector thread before exit."""
        await self.close()


def build_application(args: dict[str, Any]) -> Any:
    """Public Ray Serve application builder; KubeRay passes the same validated args mapping."""
    from finserve.auth import require_credentials

    require_credentials("FINSERVE_ENGINE_API_KEY", "FINSERVE_RAY_API_KEY")
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
            name,
            configuration.capacity_per_worker,
            configuration.model,
            endpoint,
            configuration.observe_engine_metrics,
            configuration.vllm_structured_outputs,
        )
        for name, endpoint in configuration.backends.items()
    }
    return serve.deployment(
        name="engine-routing",
        num_replicas=1,
        max_ongoing_requests=128,
        max_queued_requests=128,
        ray_actor_options={"num_cpus": 0.1, "num_gpus": 0},
    )(ServeRoutedBackends).bind(
        workers, configuration.model, configuration.mode, configuration.shared_gpu_uuid
    )
