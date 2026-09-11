"""Durable intent and fencing checks use independent SQLite connections and real restart."""

import asyncio
import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from finserve.contracts.visual import VisualArtifact, VisualAttempt, VisualJobRequest
from finserve.multimodal.benchmark import conditioning_image
from finserve.multimodal.jobs import (
    JobCapacityError,
    JobConflict,
    VisualJobCoordinator,
    VisualJobStore,
)
from finserve.multimodal.visual_rpc import MODEL_REVISION, RPCExecution, VisualRPCClient


def request(seed: int = 17) -> VisualJobRequest:
    """Build public bounded input so storage tests do not need optional numerical dependencies."""
    return VisualJobRequest(image=conditioning_image(), model_revision=MODEL_REVISION, seed=seed)


def artifact() -> VisualArtifact:
    """Provide byte identity independently of the store's hash verification implementation."""
    png = b"\x89PNG\r\n\x1a\nfixture"
    return VisualArtifact(
        png=png,
        sha256=hashlib.sha256(png).hexdigest(),
        model_sha256="a" * 64,
        initialization_ns=0,
        generation_ns=1,
        rendering_ns=1,
    )


def test_idempotent_acceptance_concurrent_and_restart(tmp_path: Path) -> None:
    """Concurrent submissions and coordinator restart must still resolve one durable intent."""
    path = tmp_path / "jobs.db"
    store = VisualJobStore(path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(store.submit, "tenant", "key", request()) for _ in range(8)]
    jobs = [future.result() for future in futures]
    assert len({job.job_id for job in jobs}) == 1
    reopened = VisualJobStore(path)
    assert reopened.submit("tenant", "key", request()) == jobs[0]
    rehydrated = VisualJobRequest.model_validate_json(request().model_dump_json())
    assert reopened.submit("tenant", "key", rehydrated) == jobs[0]
    explicit = request().model_copy(update={"timeout_seconds": 30.0})
    assert reopened.submit("tenant", "key", explicit) == jobs[0]
    with pytest.raises(JobConflict):
        reopened.submit("tenant", "key", request(18))
    assert reopened.submit("other", "key", request()).job_id != jobs[0].job_id


def test_admission_caps_and_exact_replay_when_full(tmp_path: Path) -> None:
    """Bound accepted pending work and retained history before returning acceptance."""
    store = VisualJobStore(tmp_path / "jobs.db", max_pending=1, max_records=2)
    first = store.submit("tenant", "a", request())
    with pytest.raises(JobCapacityError):
        store.submit("tenant", "b", request())
    assert store.submit("tenant", "a", request()) == first
    store.request_cancel("tenant", first.job_id)
    second = store.submit("tenant", "b", request())
    store.request_cancel("tenant", second.job_id)
    with pytest.raises(JobCapacityError):
        store.submit("tenant", "c", request())


def test_atomic_claim_duplicate_completion_and_tenant_isolation(tmp_path: Path) -> None:
    """Only one claimant can commit bytes, and a terminal artifact cannot be changed by replay."""
    store = VisualJobStore(tmp_path / "jobs.db")
    job = store.submit("tenant", "a", request())
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(store.claim_next) for _ in range(4)]
    claimed = [future.result() for future in futures if future.result() is not None]
    assert len(claimed) == 1 and claimed[0] is not None
    attempt = claimed[0][1]
    assert store.complete(attempt, artifact())
    assert not store.complete(attempt, artifact())
    assert store.artifact("tenant", job.job_id) == artifact().png
    for action in (store.get, store.artifact, store.request_cancel):
        with pytest.raises(KeyError):
            action("other", job.job_id)
    assert store.request_cancel("tenant", job.job_id).state == "succeeded"


def test_running_cancel_fences_late_result_and_requires_drain(tmp_path: Path) -> None:
    """A native result arriving after cancellation can never publish or release pending quota."""
    store = VisualJobStore(tmp_path / "jobs.db", max_pending=1)
    job = store.submit("tenant", "a", request())
    claimed = store.claim_next()
    assert claimed is not None
    attempt = claimed[1]
    cancelled = store.request_cancel("tenant", job.job_id)
    assert cancelled.state == "cancel_requested" and cancelled.generation == attempt.generation + 1
    assert not store.complete(attempt, artifact())
    assert not store.finish_cancel(attempt, drained=False)
    with pytest.raises(JobCapacityError):
        store.submit("tenant", "b", request())
    assert store.finish_cancel(attempt, drained=True)
    assert not store.finish_cancel(attempt, drained=True)
    assert store.get("tenant", job.job_id).state == "cancelled"
    with pytest.raises(JobConflict):
        store.artifact("tenant", job.job_id)


def test_restart_preserves_running_and_worker_identity(tmp_path: Path) -> None:
    """Restart exposes ambiguity instead of falsely requeueing possibly live native work."""
    path = tmp_path / "jobs.db"
    store = VisualJobStore(path)
    job = store.submit("tenant", "a", request())
    claimed = store.claim_next()
    assert claimed is not None
    execution = RPCExecution(claimed[1], worker_instance="old-worker")
    store.record_worker(execution)
    reopened = VisualJobStore(path)
    assert reopened.claim_next() is None
    assert reopened.get("tenant", job.job_id).state == "running"
    unresolved = reopened.unresolved()
    assert len(unresolved) == 1 and unresolved[0][1].worker_instance == "old-worker"
    assert unresolved[0][1].attempt == execution.attempt


def test_invalid_artifact_and_invalid_capacity_are_rejected(tmp_path: Path) -> None:
    """Validation failures must not silently create a successful job or invalid admission policy."""
    with pytest.raises(ValueError):
        VisualJobStore(tmp_path / "bad.db", max_pending=0)
    store = VisualJobStore(tmp_path / "jobs.db")
    store.submit("tenant", "a", request())
    claimed = store.claim_next()
    assert claimed is not None
    with pytest.raises(ValueError, match="integrity"):
        store.complete(claimed[1], artifact().model_copy(update={"sha256": "0" * 64}))
    assert store.get("tenant", claimed[1].job_id).state == "running"


def test_stored_artifact_corruption_is_detected(tmp_path: Path) -> None:
    """A corrupt local artifact cannot be returned merely because its job status says success."""
    store = VisualJobStore(tmp_path / "jobs.db")
    job = store.submit("tenant", "a", request())
    claimed = store.claim_next()
    assert claimed is not None and store.complete(claimed[1], artifact())
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE visual_jobs SET artifact=?", (b"corrupt",))
    with pytest.raises(RuntimeError, match="integrity"):
        store.artifact("tenant", job.job_id)


class ControlledClient(VisualRPCClient):
    """Deterministic lifecycle fake for races around SQL awaits, independent from gRPC transport."""

    def __init__(self) -> None:
        """Expose explicit barriers so cancellation tests require no timing assumption."""
        self.started_event = asyncio.Event()
        self.release_generation = asyncio.Event()
        self.cleanup_entered = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.cancel_calls = 0
        self.closed = False

    async def generate(self, execution: RPCExecution) -> VisualArtifact:
        """Record admission before waiting indefinitely for cancellation or an explicit result."""
        execution.started = True
        execution.worker_instance = "controlled-worker"
        execution.admitted.set()
        self.started_event.set()
        await self.release_generation.wait()
        return artifact()

    async def cancel(self, execution: RPCExecution, deadline_seconds: float = 120) -> bool:
        """Hold cleanup at a visible barrier so repeated caller cancellation can be injected."""
        self.cancel_calls += 1
        self.cleanup_entered.set()
        await self.release_cleanup.wait()
        return True

    async def close(self) -> None:
        """Record channel ownership closure separately from native cleanup acknowledgment."""
        self.closed = True


async def test_repeated_cancel_cannot_interrupt_cleanup_persistence(tmp_path: Path) -> None:
    """A second cancel during SQL persistence cannot abandon remote cleanup or terminal commit."""
    entered, release, recorded = threading.Event(), threading.Event(), threading.Event()

    class PausedStore(VisualJobStore):
        """Hold only cleanup's record_worker call; normal admission identity persists first."""

        pause = False

        def record_worker(self, execution: RPCExecution) -> None:
            """Expose the prior unshielded SQL-await race without holding the database lock."""
            if self.pause:
                entered.set()
                assert release.wait(5)
            super().record_worker(execution)
            recorded.set()

    store = PausedStore(tmp_path / "jobs.db")
    client = ControlledClient()
    coordinator = VisualJobCoordinator(store, client)
    coordinator.start()
    job = await coordinator.submit("tenant", "a", request())
    await client.started_event.wait()
    assert await asyncio.to_thread(recorded.wait, 5)
    store.pause = True
    first = asyncio.create_task(coordinator.cancel("tenant", job.job_id))
    assert await asyncio.to_thread(entered.wait, 5)
    second = asyncio.create_task(coordinator.cancel("tenant", job.job_id))
    await asyncio.sleep(0)
    release.set()
    await client.cleanup_entered.wait()
    assert client.cancel_calls == 1
    client.release_cleanup.set()
    results = await asyncio.gather(first, second)
    assert all(result.state == "cancelled" for result in results)
    await coordinator.close()
    assert client.closed


async def test_close_drains_despite_repeated_caller_cancellation(tmp_path: Path) -> None:
    """Shutdown owns cleanup and channel close even when the caller task is cancelled repeatedly."""
    store = VisualJobStore(tmp_path / "jobs.db")
    client = ControlledClient()
    coordinator = VisualJobCoordinator(store, client)
    coordinator.start()
    job = await coordinator.submit("tenant", "a", request())
    await client.started_event.wait()
    closing = asyncio.create_task(coordinator.close())
    await client.cleanup_entered.wait()
    closing.cancel()
    await asyncio.sleep(0)
    closing.cancel()
    assert not client.closed
    client.release_cleanup.set()
    await closing
    assert client.cancel_calls == 1 and client.closed
    assert store.get("tenant", job.job_id).state == "cancelled"


async def test_dispatch_failure_rejects_new_jobs_and_closes_channel(tmp_path: Path) -> None:
    """A dead dispatch loop must report degraded readiness rather than keep accepting work."""

    class BrokenStore(VisualJobStore):
        """Make dispatch failure deterministic while preserving durable submit behavior."""

        def claim_next(self) -> tuple[str, VisualAttempt] | None:
            """Represent a database failure whose private detail must not appear in readiness."""
            raise RuntimeError("private database detail")

    client = ControlledClient()
    coordinator = VisualJobCoordinator(BrokenStore(tmp_path / "jobs.db"), client)
    coordinator.start()
    for _ in range(200):
        if coordinator.failure_type is not None:
            break
        await asyncio.sleep(0.01)
    assert coordinator.failure_type == "RuntimeError" and not coordinator.ready
    with pytest.raises(RuntimeError, match="unavailable"):
        await coordinator.submit("tenant", "a", request())
    with pytest.raises(RuntimeError, match="private database detail"):
        await coordinator.close()
    assert client.closed
