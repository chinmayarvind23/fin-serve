"""A stale full gauge may delay admission but never create an unowned worker call."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from finserve.contracts.inference import InferenceRequest
from finserve.contracts.routing import ReplicaSnapshot, RoutingRequest
from finserve.engines.ray_backends import RoutedBackends
from finserve.scheduler.router import NoReplicaAvailable, ReplicaCapacityUnavailable, ReplicaRouter


def full_snapshot(*, healthy: bool = True) -> ReplicaSnapshot:
    """Model a cached native count of four with no current proxy leases to reconcile."""
    return ReplicaSnapshot(
        replica_id="a",
        model="fixture",
        received_at=time.monotonic(),
        capacity=4,
        ongoing_requests=4,
        engine_running_requests=4,
        healthy=healthy,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"healthy": False},
        {"capacity": 0},
        {"model": "different"},
        {"received_at": 0},
        {"engine_observed_at": 0},
        {"gpu_type": "GPU"},
        {"gpu_type": "GPU", "gpu_memory_utilization": 0.99},
    ],
)
def test_only_capacity_exhaustion_is_waitable(changes: dict[str, object]) -> None:
    """Invalid health and zero capacity stay fail-closed even when load also appears full."""
    router = ReplicaRouter()
    for update, expected in [({}, ReplicaCapacityUnavailable), (changes, NoReplicaAvailable)]:
        router = ReplicaRouter()
        router.update_snapshot(full_snapshot().model_copy(update=update))
        with pytest.raises(expected) as failure:
            router.reserve(RoutingRequest(model="fixture"))
        assert type(failure.value) is expected


async def test_generation_after_admission_wait_uses_remaining_budget() -> None:
    """The full routing path forwards only the time left after waiting, then releases its lease."""
    worker, result = MagicMock(), MagicMock()
    worker.options.return_value = worker
    worker.admit.remote = AsyncMock()
    worker.retire.remote = AsyncMock(return_value=True)
    worker.generate.remote.return_value = result

    async def output() -> AsyncGenerator[dict[str, Any], None]:
        """Supply one complete transport envelope without needing the optional Ray runtime."""
        yield {"replica_id": "a", "token": {"text": "x", "generated_tokens": 1}}

    result.__aiter__.side_effect = output
    route = RoutedBackends({"a": worker}, "fixture", "least_load")
    calls = 0

    async def refresh() -> None:
        """Hold the initial native count for one admission poll, then expose available capacity."""
        nonlocal calls
        calls += 1
        route.router.update_snapshot(
            full_snapshot().model_copy(update={"ongoing_requests": 4 if calls == 1 else 0})
        )

    route.refresh_snapshots = refresh
    received = [
        row
        async for row in route.generate(
            InferenceRequest(model="fixture", prompt="x", timeout_seconds=1).model_dump()
        )
    ]
    forwarded = worker.generate.remote.call_args.args[0]
    assert 0 < forwarded["timeout_seconds"] <= 0.96
    assert len(received) == 1 and worker.admit.remote.call_count == 1
    assert route.router.active_reservations == 0
    assert route.admission_statistics()["pending_admissions"] == 0
    await route.close()


async def test_cached_full_count_waits_until_fresh_capacity() -> None:
    """Do not discard the native observation; acquire once its later observation is free."""
    worker = MagicMock()
    route = RoutedBackends({"a": worker}, "fixture", "least_load")
    calls = 0

    async def refresh() -> None:
        """Expire the cached gauge on the second poll, as a completed native request would."""
        nonlocal calls
        calls += 1
        row = full_snapshot()
        if calls > 1:
            row = row.model_copy(update={"ongoing_requests": 0, "engine_running_requests": 0})
        route.router.update_snapshot(row)

    route.refresh_snapshots = refresh
    lease = await route.reserve_request(
        InferenceRequest(model="fixture", prompt="x"), time.monotonic() + 1
    )
    assert calls == 2 and route.router.active_reservations == 1
    stats = route.admission_statistics()
    assert stats["pending_admissions"] == 0 and stats["admission_wait_count"] == 1
    assert stats["admission_wait_seconds"] >= 0.04
    worker.admit.remote.assert_not_called()
    route.router.release(lease)
    await route.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_wait_deadline_or_cancel_acquires_nothing(cancel: bool) -> None:
    """Timeout and caller cancellation cannot leak a reservation or invoke the worker."""
    worker = MagicMock()
    route = RoutedBackends({"a": worker}, "fixture", "least_load")
    route.router.update_snapshot(full_snapshot())
    route.refresh_snapshots = AsyncMock()
    task = asyncio.create_task(
        route.reserve_request(
            InferenceRequest(model="fixture", prompt="x"),
            time.monotonic() + (1 if cancel else 0.02),
        )
    )
    if cancel:
        await asyncio.sleep(0.01)
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert route.admission_statistics()["pending_admissions"] == 0
    assert route.router.active_reservations == 0
    worker.admit.remote.assert_not_called()
    await route.close()


async def test_unhealthy_and_waiter_cap_reject_without_wait() -> None:
    """Only admitted, otherwise eligible requests may wait for temporary saturation."""
    route = RoutedBackends({}, "fixture", "least_load")
    route.router.update_snapshot(full_snapshot(healthy=False))
    route.refresh_snapshots = AsyncMock()
    with pytest.raises(NoReplicaAvailable):
        await route.reserve_request(
            InferenceRequest(model="fixture", prompt="x"), time.monotonic() + 1
        )
    assert route.admission_statistics()["admission_wait_count"] == 0
    gate = asyncio.Event()

    async def blocked_refresh() -> None:
        """Hold every accepted admission so the next call must exercise the real waiter cap."""
        await gate.wait()

    route.refresh_snapshots = blocked_refresh
    pending = [
        asyncio.create_task(
            route.reserve_request(
                InferenceRequest(model="fixture", prompt="x"), time.monotonic() + 5
            )
        )
        for _ in range(128)
    ]
    await asyncio.sleep(0.01)
    with pytest.raises(NoReplicaAvailable, match="wait limit"):
        await route.reserve_request(
            InferenceRequest(model="fixture", prompt="x"), time.monotonic() + 1
        )
    assert route.admission_statistics()["pending_admissions"] == 128
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    assert route.admission_statistics()["pending_admissions"] == 0
    await route.close()
