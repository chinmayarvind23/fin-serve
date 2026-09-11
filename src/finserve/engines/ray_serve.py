"""A real two-worker Ray Serve fixture for routing and cancellation integration.

Ray is imported only when building the application. Dynamic handle types are confined
to this optional boundary; requests and scheduler state retain strict local contracts.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal, cast

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.contracts.routing import ReplicaSnapshot, RoutingRequest
from finserve.scheduler.policy import RoutingPolicy
from finserve.scheduler.router import ReplicaRouter, RoutingLease


class FixtureReplica:
    """CPU-only character fixture with worker-enforced capacity and observable cancellation."""

    def __init__(self, replica_id: str, capacity: int, token_delay_seconds: float) -> None:
        """Use independent replica identities so public deployment handles can target selections."""
        self.replica_id = replica_id
        self.capacity = capacity
        self.token_delay_seconds = token_delay_seconds
        self.active: set[str] = set()
        self._producers: dict[str, asyncio.Task[Any]] = {}
        self.completed = 0
        self.cancelled = 0

    async def generate(
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
            for character in (request.prompt + " ")[: request.max_tokens]:
                await asyncio.sleep(self.token_delay_seconds)
                token = EngineToken(text=character, token_id=ord(character))
                yield {"replica_id": self.replica_id, "token": token.model_dump()}
            complete = True
        finally:
            self.active.discard(lease_id)
            self._producers.pop(lease_id, None)
            self.completed += int(complete)
            self.cancelled += int(not complete)

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
            "model": "fixture",
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
        self.router = ReplicaRouter(RoutingPolicy(mode=mode))

    async def _refresh_one(self, replica_id: str, handle: Any) -> None:
        """Bound heartbeat latency and quarantine workers without a current snapshot."""
        try:
            async with asyncio.timeout(2):
                data: dict[str, Any] = await handle.snapshot.remote()
            data.pop("completed", None)
            data.pop("cancelled", None)
            data["received_at"] = time.monotonic()
            self.router.update_snapshot(ReplicaSnapshot.model_validate(data))
        except Exception:
            self.router.update_snapshot(
                ReplicaSnapshot(
                    replica_id=replica_id,
                    model="fixture",
                    received_at=time.monotonic(),
                    capacity=0,
                    healthy=False,
                )
            )

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

    async def generate(self, payload: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Route once, stream without retry, and drain cancellation before releasing admission."""
        request = InferenceRequest.model_validate(payload)
        await asyncio.gather(
            *(self._refresh_one(replica_id, handle) for replica_id, handle in self.handles.items())
        )
        lease = self.router.reserve(RoutingRequest(model=request.model))
        admission: asyncio.Future[Any] | None = None
        result: Any = None
        try:
            worker = self.handles[lease.decision.replica_id]
            handle = worker.options(stream=True)
            admission = cast(
                asyncio.Future[Any], asyncio.ensure_future(worker.admit.remote(lease.lease_id))
            )
            await asyncio.shield(admission)
            result = handle.generate.remote(payload, lease.lease_id)
            async for token in result:
                yield token
        finally:
            if admission is None:
                self.router.release(lease)
            else:
                cleanup = asyncio.create_task(self._release_after_worker(lease, admission, result))
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

    async def _http_lines(self, payload: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Encode one JSON line per envelope; explicit generator closure propagates disconnects."""
        import json

        output = self.generate(payload)
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
        return OwnedFixtureResponse(self._http_lines(payload))


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
        result = self._handle.generate.remote(request.model_dump())
        try:
            async for envelope in result:
                yield EngineToken.model_validate(envelope["token"])
        finally:
            result.cancel()

    async def close(self) -> None:
        """Reject new calls; cluster and application shutdown belong to the deployment owner."""
        self._closed = True
