"""A real two-worker Ray Serve fixture for routing and cancellation integration.

Ray is imported only when building the application. Dynamic handle types are confined
to this optional boundary; requests and scheduler state retain strict local contracts.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from contextvars import Context
from typing import Any, Literal, cast

from opentelemetry.trace import NoOpTracer, Tracer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.contracts.routing import ReplicaSnapshot, RoutingRequest
from finserve.scheduler.policy import RoutingPolicy
from finserve.scheduler.router import ReplicaRouter, RoutingLease
from finserve.telemetry.propagation import trace_headers, traced_stream


class FixtureReplica:
    """CPU-only character fixture with worker-enforced capacity and observable cancellation."""

    def __init__(self, replica_id: str, capacity: int, token_delay_seconds: float) -> None:
        """Use independent replica identities so public deployment handles can target selections."""
        self.replica_id = replica_id
        self.model = "fixture"
        self.capacity = capacity
        self.token_delay_seconds = token_delay_seconds
        self.active: set[str] = set()
        self._producers: dict[str, asyncio.Task[Any]] = {}
        self.completed = 0
        self.cancelled = 0
        self.tracer: Tracer = NoOpTracer()

    async def generate(
        self, payload: dict[str, Any], lease_id: str, traceparent: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Reconstitute private trace identity explicitly across the Ray actor boundary."""
        async with aclosing(
            traced_stream(
                self._generate(payload, lease_id), self.tracer, "finserve.engine", traceparent
            )
        ) as output:
            async for token in output:
                yield token

    async def _generate(
        self, payload: dict[str, Any], lease_id: str
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Keep ownership through every yielded token and remove it in all terminal paths."""
        request = InferenceRequest.model_validate(payload)
        if lease_id not in self.active or lease_id in self._producers:
            raise RuntimeError("worker lease is not admitted or already producing")
        producer = asyncio.current_task()
        assert producer is not None
        self._producers[lease_id] = producer
        complete = False
        try:
            async with aclosing(self.tokens(request)) as output:
                async for token in output:
                    yield {"replica_id": self.replica_id, "token": token.model_dump()}
            complete = True
        finally:
            self.active.discard(lease_id)
            self._producers.pop(lease_id, None)
            self.completed += int(complete)
            self.cancelled += int(not complete)

    async def tokens(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """Keep fixture generation replaceable while sharing the verified lease lifecycle."""
        for character in (request.prompt + " ")[: request.max_tokens]:
            await asyncio.sleep(self.token_delay_seconds)
            yield EngineToken(text=character, token_id=ord(character))

    def admit(self, lease_id: str) -> None:
        """Fence generation with an explicit allowlist; a retired queued call cannot self-admit."""
        if lease_id in self.active or len(self.active) >= self.capacity:
            raise RuntimeError("worker admission capacity or lease ownership violated")
        self.active.add(lease_id)

    async def retire(self, lease_id: str) -> None:
        """Cancel and drain an active producer, or revoke a reservation before generation starts."""
        producer = self._producers.get(lease_id)
        if producer is not None:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        self.active.discard(lease_id)

    def snapshot(self) -> dict[str, Any]:
        """Report acknowledged lease IDs; the router stamps receipt time on its own clock."""
        return {
            "replica_id": self.replica_id,
            "model": self.model,
            "capacity": self.capacity,
            "ongoing_requests": len(self.active),
            "reflected_lease_ids": list(self.active),
            "completed": self.completed,
            "cancelled": self.cancelled,
        }


class RoutedFixture:
    """One routing actor owns reservations for both explicitly addressable worker deployments."""

    def __init__(self, first: Any, second: Any, mode: Literal["least_load", "adaptive"]) -> None:
        """Keep Ray's per-deployment scheduling beneath an explicit, single routing authority."""
        self.handles: dict[str, Any] = {"fixture-a": first, "fixture-b": second}
        self.model = "fixture"
        self.router = ReplicaRouter(RoutingPolicy(mode=mode))
        self.tracer: Tracer = NoOpTracer()

    async def _refresh_one(
        self, replica_id: str, handle: Any, shared_fields: dict[str, object] | None = None
    ) -> None:
        """Bound heartbeat latency and quarantine workers without a current snapshot."""
        try:
            started = time.monotonic()
            async with asyncio.timeout(2):
                data = await self.read_snapshot(handle)
            if data.get("replica_id") != replica_id:
                raise ValueError("worker snapshot identity mismatch")
            data.pop("completed", None)
            data.pop("cancelled", None)
            if "engine_observation_age_seconds" in data:
                age = data.pop("engine_observation_age_seconds")
                if not isinstance(age, (float, int)) or age < 0:
                    raise ValueError("invalid engine observation age")
                # Subtract the whole RPC interval conservatively; remote clocks are incomparable.
                data["engine_observed_at"] = started - age
            if shared_fields:
                data.update(shared_fields)
            data["received_at"] = time.monotonic()
            self.router.update_snapshot(ReplicaSnapshot.model_validate(data))
        except Exception:
            self.router.update_snapshot(
                ReplicaSnapshot(
                    replica_id=replica_id,
                    model=self.model,
                    received_at=time.monotonic(),
                    capacity=0,
                    healthy=False,
                )
            )

    async def refresh_snapshots(self) -> None:
        """A hook allows production routing to attach one common physical observation per round."""
        await asyncio.gather(
            *(self._refresh_one(name, handle) for name, handle in self.handles.items())
        )

    async def read_snapshot(self, handle: Any) -> dict[str, Any]:
        """Keep the public heartbeat call replaceable for engine-backed workers."""
        return cast(dict[str, Any], await handle.snapshot.remote())

    def record_selection(self, request: InferenceRequest, lease: RoutingLease) -> None:
        """Production observers may retain bounded prompt-free evidence after atomic reservation."""

    async def _release_after_worker(
        self, lease: RoutingLease, admission: asyncio.Future[Any], result: Any
    ) -> None:
        """Wait for cancellation acknowledgement before returning the routing slot.

        On failed acknowledgement the lease remains reserved, failing closed until
        operator recovery. This fixture does not pretend to provide durable failover.
        """
        handle = self.handles[lease.decision.replica_id]
        async with asyncio.timeout(5):
            await asyncio.gather(admission, return_exceptions=True)
            try:
                if result is not None:
                    result.cancel()
            finally:
                await handle.retire.remote(lease.lease_id)
            self.router.release(lease)

    async def generate(
        self, payload: dict[str, Any], traceparent: str | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Keep route span context out of callers while covering lease cleanup and RPC waits."""
        async with aclosing(
            traced_stream(self._generate(payload), self.tracer, "finserve.route", traceparent)
        ) as output:
            async for token in output:
                yield token

    async def reserve_request(self, request: InferenceRequest, deadline: float) -> RoutingLease:
        """Keep the CPU fixture's immediate admission policy separate from production waiting."""
        await asyncio.wait_for(
            self.refresh_snapshots(),
            timeout=max(0, deadline - time.monotonic()),
        )
        return self.router.reserve(RoutingRequest(model=request.model))

    async def _generate(self, payload: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Route once, stream without retry, and drain cancellation before releasing admission."""
        request = InferenceRequest.model_validate(payload)
        deadline = time.monotonic() + request.timeout_seconds
        lease = await self.reserve_request(request, deadline)
        admission: asyncio.Future[Any] | None = None
        result: Any = None
        try:
            self.record_selection(request, lease)
            worker = self.handles[lease.decision.replica_id]
            handle = worker.options(stream=True)
            admission = cast(
                asyncio.Future[Any], asyncio.ensure_future(worker.admit.remote(lease.lease_id))
            )
            await asyncio.wait_for(asyncio.shield(admission), max(0, deadline - time.monotonic()))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("routing request budget exceeded")
            payload = request.model_copy(update={"timeout_seconds": remaining}).model_dump()
            parent = trace_headers().get("traceparent")
            result = (
                handle.generate.remote(payload, lease.lease_id, parent)
                if parent
                else handle.generate.remote(payload, lease.lease_id)
            )
            iterator = aiter(result)
            while True:
                try:
                    token = await asyncio.wait_for(
                        anext(iterator), max(0, deadline - time.monotonic())
                    )
                except StopAsyncIteration:
                    break
                yield token
        finally:
            if admission is None:
                self.router.release(lease)
            else:
                # Cleanup is a new control operation, not a child of the cancelled Ray request.
                # A fresh context prevents Ray from immediately cancelling its retire RPC.
                cleanup = asyncio.create_task(
                    self._release_after_worker(lease, admission, result), context=Context()
                )
                cancelled_during_cleanup = False
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled_during_cleanup = True
                cleanup.result()
                if cancelled_during_cleanup:
                    raise asyncio.CancelledError

    async def status(self) -> dict[str, Any]:
        """Expose fixture-only lifecycle evidence without prompts or model-generated content."""
        workers = await asyncio.gather(
            *(handle.snapshot.remote() for handle in self.handles.values())
        )
        return {"active_reservations": self.router.active_reservations, "workers": workers}

    async def _http_lines(
        self, payload: dict[str, Any], traceparent: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Encode one JSON line per envelope; explicit generator closure propagates disconnects."""
        import json

        output = self.generate(payload, traceparent)
        try:
            async for envelope in output:
                yield json.dumps(envelope) + "\n"
        finally:
            await output.aclose()

    async def __call__(self, request: Request) -> Response:
        """Provide localhost functional HTTP verification without changing the product ingress."""
        if request.method == "GET":
            return JSONResponse(await self.status())
        payload = InferenceRequest.model_validate(await request.json()).model_dump()
        parents = request.headers.getlist("traceparent")
        parent = parents[0] if len(parents) == 1 else None
        return OwnedFixtureResponse(self._http_lines(payload, parent))


class OwnedFixtureResponse(StreamingResponse):
    """Own the fixture iterator so failing ASGI header/body writes cannot orphan a worker."""

    def __init__(self, output: AsyncGenerator[str, None]) -> None:
        """Retain explicit generation ownership independently of Starlette iteration internals."""
        super().__init__(output, media_type="application/x-ndjson")
        self.output = output

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Close even if response start/body transport raises before normal stream exhaustion."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.output.aclose()


def build_fixture_application(
    mode: Literal["least_load", "adaptive"] = "least_load",
    capacity_per_worker: int = 2,
    token_delay_seconds: float = 0.02,
) -> Any:
    """Build two real CPU worker deployments without importing Ray into ordinary API startup.

    One replica per named deployment makes public handles addressable. It does not
    claim to replace Ray's default replica scheduler or to scale this layout unchanged.
    """
    if capacity_per_worker < 1 or not 0 <= token_delay_seconds <= 1:
        raise ValueError("fixture capacity and token delay are outside supported bounds")
    serve = importlib.import_module("ray.serve")
    workers = [
        serve.deployment(
            name=replica_id,
            num_replicas=1,
            max_ongoing_requests=capacity_per_worker + 4,
            ray_actor_options={"num_cpus": 0.1, "num_gpus": 0},
        )(FixtureReplica).bind(replica_id, capacity_per_worker, token_delay_seconds)
        for replica_id in ("fixture-a", "fixture-b")
    ]
    return serve.deployment(
        name="routing-ingress",
        num_replicas=1,
        max_ongoing_requests=16,
        max_queued_requests=16,
        ray_actor_options={"num_cpus": 0.1, "num_gpus": 0},
    )(RoutedFixture).bind(workers[0], workers[1], mode)


class RayServeEngine:
    """Adapt a deployed routing handle to ingress without owning the shared Ray cluster."""

    def __init__(self, handle: Any) -> None:
        """Accept an already deployed application handle, keeping discovery outside request data."""
        self._handle = handle.options(stream=True)
        self._closed = False

    async def stream(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """Cancel through the routing actor, which retains leases until worker acknowledgement."""
        if self._closed:
            raise RuntimeError("Ray engine adapter is closed")
        parent = trace_headers().get("traceparent")
        result = (
            self._handle.generate.remote(request.model_dump(), parent)
            if parent
            else self._handle.generate.remote(request.model_dump())
        )
        try:
            async for envelope in result:
                yield EngineToken.model_validate(envelope["token"])
        finally:
            result.cancel()

    async def close(self) -> None:
        """Reject new calls; cluster and application shutdown belong to the deployment owner."""
        self._closed = True
