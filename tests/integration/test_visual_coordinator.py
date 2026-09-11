"""Durable coordinator checks across real loopback gRPC and independently reopened SQLite."""

import asyncio
import hashlib
import threading
from pathlib import Path

import pytest

from finserve.contracts.visual import VisualArtifact, VisualJob, VisualJobRequest
from finserve.multimodal.benchmark import conditioning_image
from finserve.multimodal.jobs import VisualJobCoordinator, VisualJobStore
from finserve.multimodal.visual_rpc import MODEL_REVISION, VisualRPCClient, VisualWorker

KEY = "coordinator-test-only-service-key"


def request() -> VisualJobRequest:
    """Use one public image and immutable implementation revision across process boundaries."""
    return VisualJobRequest(image=conditioning_image(), model_revision=MODEL_REVISION)


def artifact(_: VisualJobRequest) -> VisualArtifact:
    """Return deterministic bytes while leaving transport, SQL and ownership behavior real."""
    png = b"\x89PNG\r\n\x1a\nfixture"
    return VisualArtifact(
        png=png,
        sha256=hashlib.sha256(png).hexdigest(),
        model_sha256="a" * 64,
        initialization_ns=0,
        generation_ns=1,
        rendering_ns=1,
    )


async def status(store: VisualJobStore, job_id: str, expected: str) -> VisualJob:
    """Bound status polling without binding the durable job lifetime to this reader."""
    for _ in range(200):
        job = await asyncio.to_thread(store.get, "tenant", job_id)
        if job.state == expected:
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job did not reach {expected}")


async def test_durable_queue_survives_restart_and_poll_disconnect(tmp_path: Path) -> None:
    """Reopen committed queued intent and finish it without any surviving original HTTP owner."""
    pytest.importorskip("grpc")
    path = tmp_path / "jobs.db"
    accepted = VisualJobStore(path).submit("tenant", "a", request())
    store = VisualJobStore(path)
    worker = VisualWorker(KEY, artifact)
    server, port = await worker.start()
    coordinator = VisualJobCoordinator(store, VisualRPCClient(f"127.0.0.1:{port}", KEY))
    coordinator.start()
    try:
        completed = await status(store, accepted.job_id, "succeeded")
        assert completed.artifact_sha256 == artifact(request()).sha256
        assert VisualJobStore(path).artifact("tenant", accepted.job_id) == artifact(request()).png
        assert (await coordinator.submit("tenant", "a", request())).job_id == accepted.job_id
        with pytest.raises(RuntimeError):
            coordinator.start()
    finally:
        await coordinator.close()
        await server.stop(0)
        await worker.close()


async def test_explicit_cancel_drains_and_fences_late_native_result(tmp_path: Path) -> None:
    """A reader can disappear, but explicit cancellation alone prevents publishing late bytes."""
    pytest.importorskip("grpc")
    started, release = threading.Event(), threading.Event()

    def blocked(value: VisualJobRequest) -> VisualArtifact:
        """Represent native work that outlives the cancelled request transport."""
        started.set()
        assert release.wait(10)
        return artifact(value)

    store = VisualJobStore(tmp_path / "jobs.db")
    worker = VisualWorker(KEY, blocked)
    server, port = await worker.start()
    coordinator = VisualJobCoordinator(store, VisualRPCClient(f"127.0.0.1:{port}", KEY))
    coordinator.start()
    cancelling: asyncio.Task[VisualJob] | None = None
    try:
        accepted = await coordinator.submit("tenant", "a", request())
        assert await asyncio.to_thread(started.wait, 5)
        # Read durable admission identity, rather than assume native start implies client receipt.
        for _ in range(200):
            unresolved = await asyncio.to_thread(store.unresolved)
            if unresolved and unresolved[0][1].worker_instance:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("worker admission was not persisted")
        cancelling = asyncio.create_task(coordinator.cancel("tenant", accepted.job_id))
        await status(store, accepted.job_id, "cancel_requested")
        assert worker.active == 1 and not cancelling.done()
        release.set()
        assert (await cancelling).state == "cancelled"
        assert store.get("tenant", accepted.job_id).artifact_sha256 is None
        await worker.idle.wait()
        second = await coordinator.submit("tenant", "b", request())
        assert (await status(store, second.job_id, "succeeded")).artifact_sha256
    finally:
        release.set()
        if cancelling is not None:
            await asyncio.gather(cancelling, return_exceptions=True)
        await coordinator.close()
        await server.stop(0)
        await worker.close()


async def test_queued_cancel_and_close_before_dispatch(tmp_path: Path) -> None:
    """Unclaimed cancellation needs no remote cleanup and preserves other queued intent."""
    pytest.importorskip("grpc")
    store = VisualJobStore(tmp_path / "jobs.db")
    client = VisualRPCClient("127.0.0.1:1", KEY)
    coordinator = VisualJobCoordinator(store, client)
    first = await coordinator.submit("tenant", "a", request())
    second = await coordinator.submit("tenant", "b", request())
    assert (await coordinator.cancel("tenant", first.job_id)).state == "cancelled"
    await coordinator.close()
    assert store.get("tenant", second.job_id).state == "queued"


async def test_unreachable_worker_preserves_ambiguous_attempt(tmp_path: Path) -> None:
    """A missing admission identity cannot become a false cleanup acknowledgment."""
    pytest.importorskip("grpc")
    store = VisualJobStore(tmp_path / "jobs.db")
    coordinator = VisualJobCoordinator(store, VisualRPCClient("127.0.0.1:1", KEY))
    coordinator.start()
    try:
        job = await coordinator.submit("tenant", "a", request())
        result = await status(store, job.job_id, "cancel_requested")
        assert result.artifact_sha256 is None
        assert len(VisualJobStore(store.path).unresolved()) == 1
    finally:
        await coordinator.close()


async def test_native_failure_is_terminal_only_after_drain(tmp_path: Path) -> None:
    """A typed worker failure resolves only after the instance barrier confirms no native owner."""
    pytest.importorskip("grpc")

    def failed(_: VisualJobRequest) -> VisualArtifact:
        """Simulate a private model failure whose sensitive detail must not enter job metadata."""
        raise ValueError("private model detail")

    store = VisualJobStore(tmp_path / "jobs.db")
    worker = VisualWorker(KEY, failed)
    server, port = await worker.start()
    coordinator = VisualJobCoordinator(store, VisualRPCClient(f"127.0.0.1:{port}", KEY))
    coordinator.start()
    try:
        job = await coordinator.submit("tenant", "a", request())
        failed_job = await status(store, job.job_id, "failed")
        assert failed_job.failure_type == "RuntimeError"
        assert worker.active == 0 and worker.retained_task_count == 0
    finally:
        await coordinator.close()
        await server.stop(0)
        await worker.close()


async def test_close_during_claim_before_rpc_start(tmp_path: Path) -> None:
    """Shutdown during async DB work must cancel claimed intent without ever issuing an RPC."""
    pytest.importorskip("grpc")
    entered, release = threading.Event(), threading.Event()

    class PausedStore(VisualJobStore):
        """Pause the coordinator's first status read at the exact pre-invocation boundary."""

        def get(self, tenant: str, job_id: str) -> VisualJob:
            """Make the startup race deterministic without holding a database transaction lock."""
            entered.set()
            assert release.wait(5)
            return super().get(tenant, job_id)

    store = PausedStore(tmp_path / "jobs.db")
    job = store.submit("tenant", "a", request())
    coordinator = VisualJobCoordinator(store, VisualRPCClient("127.0.0.1:1", KEY))
    coordinator.start()
    assert await asyncio.to_thread(entered.wait, 5)
    closing = asyncio.create_task(coordinator.close())
    await asyncio.sleep(0)
    release.set()
    await closing
    assert VisualJobStore(store.path).get("tenant", job.job_id).state == "cancelled"
