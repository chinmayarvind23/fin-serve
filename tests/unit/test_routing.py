"""Adversarial capacity and telemetry tests for the single-owner routing invariant."""

from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from finserve.contracts.routing import ReplicaSnapshot, RoutingRequest
from finserve.engines.ray_serve import FixtureReplica, OwnedFixtureResponse, RoutedFixture
from finserve.scheduler.policy import RoutingPolicy
from finserve.scheduler.router import NoReplicaAvailable, ReplicaRouter, RoutingLease

PREFIX = "a" * 64


def snapshot(replica_id: str = "a", /, **changes: object) -> ReplicaSnapshot:
    """Keep deterministic router-local timestamps while varying one failure condition at a time."""
    return ReplicaSnapshot.model_validate(
        {"replica_id": replica_id, "model": "fixture", "received_at": 10, "capacity": 4, **changes}
    )


def test_least_load_has_deterministic_ties_and_normalized_capacity() -> None:
    """Input order must not affect ties, and a larger worker should absorb proportionate work."""
    router = ReplicaRouter()
    router.update_snapshot(snapshot("z"))
    router.update_snapshot(snapshot("a", capacity=8))
    request = RoutingRequest(model="fixture")
    selected = [router.reserve(request, now=10) for _ in range(3)]
    assert [lease.decision.replica_id for lease in selected] == ["a", "z", "a"]
    assert all(not lease.decision.cache_affinity_used for lease in selected)


@pytest.mark.parametrize(
    "changes",
    [
        {"healthy": False},
        {"model": "different"},
        {"received_at": 4},
        {"received_at": 11},
        {"capacity": 0},
        {"ongoing_requests": 4},
        {"queued_requests": 4},
        {"gpu_type": "test-gpu", "gpu_memory_utilization": 0.95},
    ],
)
def test_ineligible_snapshots_never_receive_work(changes: dict[str, object]) -> None:
    """Optimization scores cannot revive a replica rejected by a safety boundary."""
    router = ReplicaRouter(RoutingPolicy(mode="adaptive"))
    router.update_snapshot(snapshot(cached_prefixes=[PREFIX], **changes))
    with pytest.raises(NoReplicaAvailable):
        router.reserve(RoutingRequest(model="fixture", prefix_digest=PREFIX), now=10)
    assert router.active_reservations == 0


def test_gpu_requirement_and_type_are_hard_constraints() -> None:
    """A healthy CPU or wrong accelerator cannot serve a GPU-specific request."""
    router = ReplicaRouter()
    router.update_snapshot(snapshot("cpu"))
    router.update_snapshot(snapshot("gpu", gpu_type="gpu-a"))
    request = RoutingRequest(model="fixture", requires_gpu=True, gpu_type="gpu-a")
    assert router.reserve(request, now=10).decision.replica_id == "gpu"
    with pytest.raises(NoReplicaAvailable):
        router.reserve(RoutingRequest(model="fixture", gpu_type="gpu-b"), now=10)


def test_adaptive_memory_penalty_and_bounded_affinity() -> None:
    """Memory pressure breaks load ties; small verified prefix affinity may then break near ties."""
    router = ReplicaRouter(RoutingPolicy(mode="adaptive"))
    router.update_snapshot(snapshot("a", gpu_type="gpu", gpu_memory_utilization=0.8))
    router.update_snapshot(snapshot("b", gpu_type="gpu", gpu_memory_utilization=0.2))
    lease = router.reserve(RoutingRequest(model="fixture"), now=10)
    assert lease.decision.replica_id == "b"
    router.release(lease)
    router.update_snapshot(snapshot("a", cached_prefixes=[PREFIX]))
    router.update_snapshot(snapshot("b"))
    lease = router.reserve(RoutingRequest(model="fixture", prefix_digest=PREFIX), now=10)
    assert lease.decision.replica_id == "a" and lease.decision.cache_affinity_used
    router.release(lease)
    router.update_snapshot(snapshot("a", ongoing_requests=3, cached_prefixes=[PREFIX]))
    assert (
        router.reserve(
            RoutingRequest(model="fixture", prefix_digest=PREFIX), now=10
        ).decision.replica_id
        == "b"
    )


def test_reflected_leases_are_not_double_counted_and_stale_release_is_conservative() -> None:
    """A lease transfers from local pending to reported ongoing without disappearing or doubling."""
    router = ReplicaRouter()
    router.update_snapshot(snapshot(capacity=2))
    first = router.reserve(RoutingRequest(model="fixture"), now=10)
    router.update_snapshot(
        snapshot(capacity=2, ongoing_requests=1, reflected_lease_ids=[first.lease_id])
    )
    second = router.reserve(RoutingRequest(model="fixture"), now=10)
    assert second.decision.effective_requests == 1
    with pytest.raises(NoReplicaAvailable):
        router.reserve(RoutingRequest(model="fixture"), now=10)
    router.release(first)
    with pytest.raises(NoReplicaAvailable):
        router.reserve(RoutingRequest(model="fixture"), now=10)
    router.update_snapshot(snapshot(capacity=2, received_at=11))
    third = router.reserve(RoutingRequest(model="fixture"), now=11)
    assert third.decision.effective_requests == 1
    router.release(second)
    router.release(third)
    assert router.active_reservations == 0


def test_concurrent_reservations_cannot_exceed_capacity() -> None:
    """Simultaneous threads exercise the atomic check-and-reserve boundary, not just its formula."""
    router = ReplicaRouter()
    router.update_snapshot(snapshot("a", capacity=4))
    router.update_snapshot(snapshot("b", capacity=4))

    def acquire(_: int) -> RoutingLease | None:
        """Preserve successful ownership while allowing excess arrivals to fail explicitly."""
        try:
            return router.reserve(RoutingRequest(model="fixture"), now=10)
        except NoReplicaAvailable:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        leases = [lease for lease in pool.map(acquire, range(100)) if lease is not None]
    assert len(leases) == router.active_reservations == 8
    assert sum(lease.decision.replica_id == "a" for lease in leases) == 4
    for lease in leases:
        router.release(lease)
    assert router.active_reservations == 0


def test_release_requires_ownership_and_removal_waits_for_drain() -> None:
    """Wrong-router, altered, and repeated release cannot free another request's slot."""
    router = ReplicaRouter()
    router.update_snapshot(snapshot())
    lease = router.reserve(RoutingRequest(model="fixture"), now=10)
    with pytest.raises(RuntimeError):
        ReplicaRouter().release(lease)
    with pytest.raises(RuntimeError):
        router.release(RoutingLease(lease.lease_id, lease.decision.model_copy(update={"score": 3})))
    with pytest.raises(RuntimeError):
        router.forget_replica("a")
    router.release(lease)
    with pytest.raises(RuntimeError):
        router.release(lease)
    router.forget_replica("a")
    router.forget_replica("unknown")
    with pytest.raises(NoReplicaAvailable):
        router.reserve(RoutingRequest(model="fixture"), now=10)


def test_snapshot_and_clock_validation() -> None:
    """Bound metadata and refuse invalid clocks or out-of-order load observations."""
    with pytest.raises(ValueError):
        ReplicaRouter(max_replicas=0)
    router = ReplicaRouter(max_replicas=1)
    router.update_snapshot(snapshot())
    with pytest.raises(ValueError):
        router.update_snapshot(snapshot(received_at=9))
    with pytest.raises(ValueError):
        router.update_snapshot(snapshot("b"))
    for now in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            router.reserve(RoutingRequest(model="fixture"), now=now)
    with pytest.raises(ValidationError):
        snapshot(ongoing_requests=0, reflected_lease_ids=["unknown"])
    with pytest.raises(ValidationError):
        RoutingRequest(model="fixture", prefix_digest="raw private prompt")


def test_live_clock_default_and_baseline_ignores_affinity() -> None:
    """Exercise normal clock use and preserve least-load as a genuine cache-blind baseline."""
    import time

    router = ReplicaRouter()
    router.update_snapshot(snapshot("a", received_at=time.monotonic()))
    router.update_snapshot(snapshot("z", received_at=time.monotonic(), cached_prefixes=[PREFIX]))
    lease = router.reserve(RoutingRequest(model="fixture", prefix_digest=PREFIX))
    assert lease.decision.replica_id == "a" and not lease.decision.cache_affinity_used


@pytest.mark.parametrize("failure_stage", ["options", "generate"])
async def test_ray_setup_failure_releases_every_acquired_slot(failure_stage: str) -> None:
    """Synchronous handle setup failure before or after worker admission cannot leak capacity."""
    workers = [FixtureReplica(name, 2, 0) for name in ("fixture-a", "fixture-b")]
    handles: list[MagicMock] = []
    for worker in workers:
        handle = MagicMock()
        handle.snapshot.remote = AsyncMock(side_effect=worker.snapshot)
        handle.admit.remote = AsyncMock(side_effect=worker.admit)
        handle.retire.remote = AsyncMock(side_effect=worker.retire)
        handle.options.return_value = handle
        if failure_stage == "options":
            handle.options.side_effect = RuntimeError("setup failed")
        else:
            handle.generate.remote.side_effect = RuntimeError("setup failed")
        handles.append(handle)
    routed = RoutedFixture(handles[0], handles[1], "least_load")
    with pytest.raises(RuntimeError, match="setup failed"):
        await anext(routed.generate({"model": "fixture", "prompt": "hello"}))
    assert routed.router.active_reservations == 0
    assert all(not worker.active for worker in workers)


async def test_retired_worker_admission_fences_late_queued_generation() -> None:
    """A revoked lease cannot recreate itself when its delayed generation call finally arrives."""
    worker = FixtureReplica("fixture-a", 1, 0)
    worker.admit("lease")
    with pytest.raises(RuntimeError):
        worker.admit("different")
    await worker.retire("lease")
    with pytest.raises(RuntimeError, match="not admitted"):
        await anext(worker.generate({"model": "fixture", "prompt": "hello"}, "lease"))
    assert not worker.active


@pytest.mark.parametrize("body_index", [0, 1, 2])
async def test_owned_ray_response_closes_on_asgi_send_failure(body_index: int) -> None:
    """Header and token write failures must close the explicit body iterator deterministically."""
    active = False

    async def output() -> AsyncGenerator[str, None]:
        """Represent the routed stream's resource ownership without importing Ray."""
        nonlocal active
        active = True
        try:
            yield "one\n"
            yield "two\n"
        finally:
            active = False

    async def receive() -> Message:
        """ASGI2.4 detects this fixture's disconnect through writes, not incoming events."""
        return {"type": "http.request", "body": b"", "more_body": False}

    bodies_sent = 0

    async def send(message: Message) -> None:
        """Fail at the chosen write while an iterator may own a live routing reservation."""
        nonlocal bodies_sent
        bodies_sent += int(message["type"] == "http.response.body")
        if (body_index == 0 and message["type"] == "http.response.start") or (
            body_index > 0 and bodies_sent == body_index
        ):
            raise OSError("client disconnected")

    scope: Scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    with pytest.raises(ClientDisconnect):
        await OwnedFixtureResponse(output())(scope, receive, send)
    assert not active
