"""Real localhost protobuf/gRPC checks; native CPU JAX has one explicit functional test."""

import asyncio
import base64
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from finserve.contracts.visual import VisualArtifact, VisualAttempt, VisualJobRequest
from finserve.multimodal.benchmark import conditioning_image
from finserve.multimodal.visual_rpc import (
    MODEL_REVISION,
    SERVICE,
    RPCExecution,
    VisualRPCClient,
    VisualWorker,
)
from finserve.multimodal.visual_wire import DESCRIPTOR_BASE64, message_types

KEY = "test-worker-key-not-a-real-secret"


def attempt(job_id: str = "visual-test", seconds: float = 30) -> RPCExecution:
    """Use public deterministic pixels and a unique attempt rather than private image fixtures."""
    return RPCExecution(
        VisualAttempt(
            job_id=job_id,
            generation=1,
            request=VisualJobRequest(
                image=conditioning_image(), model_revision=MODEL_REVISION, timeout_seconds=seconds
            ),
        )
    )


def artifact(_: VisualJobRequest) -> VisualArtifact:
    """Supply bounded deterministic bytes for transport tests without numerical work."""
    png = b"\x89PNG\r\n\x1a\nfixture"
    return VisualArtifact(
        png=png,
        sha256=hashlib.sha256(png).hexdigest(),
        model_sha256="a" * 64,
        initialization_ns=1,
        generation_ns=2,
        rendering_ns=3,
    )


@pytest.fixture
async def pair() -> AsyncIterator[tuple[VisualWorker, VisualRPCClient, int]]:
    """Run an actual ephemeral loopback server and always drain owned compute before shutdown."""
    pytest.importorskip("grpc")
    worker = VisualWorker(KEY, artifact)
    server, port = await worker.start()
    client = VisualRPCClient(f"127.0.0.1:{port}", KEY)
    try:
        yield worker, client, port
    finally:
        await client.close()
        await server.stop(0)
        await worker.close()


async def test_real_stream_identity_integrity_and_replay(
    pair: tuple[VisualWorker, VisualRPCClient, int],
) -> None:
    """Verify the binary terminal artifact and prohibit duplicate native executions."""
    grpc = importlib.import_module("grpc")
    worker, client, _ = pair
    execution = attempt()
    result = await client.generate(execution)
    assert result == artifact(execution.attempt.request)
    assert execution.worker_instance == worker.instance
    assert await client.cancel(execution)
    assert worker.retained_task_count == 0
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await client.generate(execution)
    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS


async def test_real_auth_invalid_shape_revision_and_message_cap(
    pair: tuple[VisualWorker, VisualRPCClient, int],
) -> None:
    """Reject untrusted metadata and malformed or oversized binary input before compute."""
    grpc = importlib.import_module("grpc")
    worker, _, port = pair
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    wire = message_types()
    call = channel.unary_stream(
        f"/{SERVICE}/GenerateVisual",
        request_serializer=wire["GenerateRequest"].SerializeToString,
        response_deserializer=wire["VisualEvent"].FromString,
    )
    cases = [
        (b"a" * 192, MODEL_REVISION, (), grpc.StatusCode.UNAUTHENTICATED),
        (
            b"a",
            MODEL_REVISION,
            (("authorization", f"Bearer {KEY}"),),
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (b"a" * 192, "wrong", (("authorization", f"Bearer {KEY}"),), grpc.StatusCode.NOT_FOUND),
        (
            b"a" * 140000,
            MODEL_REVISION,
            (("authorization", f"Bearer {KEY}"),),
            grpc.StatusCode.RESOURCE_EXHAUSTED,
        ),
    ]
    try:
        for pixels, revision, metadata, status in cases:
            request = wire["GenerateRequest"](
                job_id="bad",
                generation=1,
                image_rgb=pixels,
                model_revision=revision,
                seed=17,
                timeout_seconds=1,
            )
            with pytest.raises(grpc.aio.AioRpcError) as error:
                _ = [event async for event in call(request, timeout=1, metadata=metadata)]
            assert error.value.code() == status
        assert worker.active == 0
    finally:
        await channel.close()


@pytest.mark.parametrize("deadline", [False, True])
async def test_real_cancel_and_deadline_hold_native_capacity(deadline: bool) -> None:
    """A cancelled transport must not free capacity while its blocking native call is running."""
    grpc = pytest.importorskip("grpc")
    started, release = threading.Event(), threading.Event()

    def blocked(request: VisualJobRequest) -> VisualArtifact:
        """Represent uninterruptible native compute without relying on machine timing."""
        started.set()
        assert release.wait(10)
        return artifact(request)

    worker = VisualWorker(KEY, blocked)
    server, port = await worker.start()
    client = VisualRPCClient(f"127.0.0.1:{port}", KEY)
    execution = attempt(seconds=0.1 if deadline else 30)
    task = asyncio.create_task(client.generate(execution))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        async with asyncio.timeout(2):
            await execution.admitted.wait()
        if deadline:
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await task
            assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert worker.active == 1
        barrier = asyncio.create_task(client.cancel(execution))
        await asyncio.sleep(0.02)
        assert not barrier.done()
        with pytest.raises(grpc.aio.AioRpcError) as saturated:
            await client.generate(attempt("second"))
        assert saturated.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        release.set()
        assert await barrier
        async with asyncio.timeout(2):
            await worker.idle.wait()
        assert (await client.generate(attempt("third"))).png
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()
        await server.stop(0)
        await worker.close()


async def test_cancel_revokes_delayed_start_and_restart_is_unknown(
    pair: tuple[VisualWorker, VisualRPCClient, int],
) -> None:
    """The barrier closes queued-start races; a new instance cannot acknowledge old work."""
    grpc = importlib.import_module("grpc")
    worker, client, _ = pair
    execution = attempt()
    assert not await client.cancel(execution)
    execution.worker_instance = "previous-worker-instance"
    assert not await client.cancel(execution)
    execution.worker_instance = worker.instance
    assert await client.cancel(execution)
    with pytest.raises(grpc.aio.AioRpcError) as error:
        await client.generate(execution)
    assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
    assert worker.active == 0


def test_proto_descriptor_matches_pinned_protoc(tmp_path: Path) -> None:
    """Regenerate protobuf schema in CI to prove the checked-in descriptor is interoperable."""
    pytest.importorskip("grpc_tools")
    contracts = Path(__file__).resolve().parents[2] / "src/finserve/contracts"
    target = tmp_path / "visual.pb"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I",
            str(contracts),
            f"--descriptor_set_out={target}",
            str(contracts / "visual.proto"),
        ],
        check=True,
        timeout=30,
    )
    assert target.read_bytes() == base64.b64decode(DESCRIPTOR_BASE64)


async def test_real_jax_worker_outputs_valid_png() -> None:
    """Run one actual CPU numerical request across a real binary RPC and inspect output pixels."""
    pytest.importorskip("grpc")
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    image = pytest.importorskip("PIL.Image")
    import io

    worker = VisualWorker(KEY)
    server, port = await worker.start()
    client = VisualRPCClient(f"127.0.0.1:{port}", KEY)
    try:
        result = await client.generate(attempt())
        raster = image.open(io.BytesIO(result.png))
        assert raster.size == (128, 128) and raster.mode == "RGB"
        assert result.generation_ns > 0 and result.rendering_ns > 0
        assert hashlib.sha256(result.png).hexdigest() == result.sha256
        second = await client.generate(attempt("second-native"))
        assert second.sha256 == result.sha256 and second.model_sha256 == result.model_sha256
    finally:
        await client.close()
        await server.stop(0)
        await worker.close()


@pytest.mark.parametrize("mode", ["sequence", "hash", "empty", "failure"])
async def test_client_rejects_invalid_real_wire_terminal(mode: str) -> None:
    """Malformed peer streams cannot create a successful coordinator artifact."""
    pytest.importorskip("grpc")
    wire = message_types()

    class BadWorker(VisualWorker):
        """Keep real gRPC transport while injecting protocol violations at the producer boundary."""

        async def _produce(self, record: Any, request: VisualJobRequest, context: Any) -> None:
            """Emit one deliberate violation with otherwise valid protobuf framing."""
            if mode == "empty":
                return
            await context.write(
                wire["VisualEvent"](
                    sequence=3 if mode == "sequence" else 0,
                    admitted=wire["Admitted"](worker_instance=self.instance),
                )
            )
            if mode == "hash":
                value = artifact(request).model_copy(update={"sha256": "0" * 64})
                await context.write(
                    wire["VisualEvent"](sequence=1, artifact=wire["Artifact"](**value.model_dump()))
                )
            elif mode == "failure":
                await context.write(
                    wire["VisualEvent"](sequence=1, failure=wire["Failure"](code="bounded_failure"))
                )

    worker = BadWorker(KEY, artifact)
    server, port = await worker.start()
    client = VisualRPCClient(f"127.0.0.1:{port}", KEY)
    try:
        with pytest.raises(RuntimeError):
            await client.generate(attempt())
    finally:
        await client.close()
        await server.stop(0)
        await worker.close()


async def test_tombstone_capacity_and_invalid_local_configuration() -> None:
    """Never evict revocations to admit work; reject unsupported remote destinations explicitly."""
    grpc = pytest.importorskip("grpc")
    for key, capacity in (("short", 1), (KEY, 0)):
        with pytest.raises(ValueError):
            VisualWorker(key, max_attempts=capacity)
    with pytest.raises(ValueError):
        VisualRPCClient("example.com:50061", KEY)
    worker = VisualWorker(KEY, artifact, max_attempts=1)
    with pytest.raises(ValueError):
        await worker.start("0.0.0.0:50061")
    server, port = await worker.start()
    client = VisualRPCClient(f"127.0.0.1:{port}", KEY)
    try:
        first = attempt("first")
        first.worker_instance = worker.instance
        assert await client.cancel(first)
        second = attempt("second")
        second.worker_instance = worker.instance
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await client.cancel(second)
        assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        with pytest.raises(grpc.aio.AioRpcError):
            await client.generate(second)
        assert worker.active == 0
    finally:
        await client.close()
        await server.stop(0)
        await worker.close()


async def test_server_budget_without_client_deadline() -> None:
    """An internal caller cannot bypass the declared budget by omitting its transport deadline."""
    grpc = pytest.importorskip("grpc")
    release = threading.Event()

    def blocked(request: VisualJobRequest) -> VisualArtifact:
        """Hold native compute past its deadline to distinguish timeout from resource release."""
        assert release.wait(5)
        return artifact(request)

    worker = VisualWorker(KEY, blocked)
    server, port = await worker.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    wire = message_types()
    stream = channel.unary_stream(
        f"/{SERVICE}/GenerateVisual",
        request_serializer=wire["GenerateRequest"].SerializeToString,
        response_deserializer=wire["VisualEvent"].FromString,
    )(
        wire["GenerateRequest"](
            job_id="budget",
            generation=1,
            image_rgb=b"x" * 192,
            model_revision=MODEL_REVISION,
            seed=17,
            timeout_seconds=0.05,
        ),
        metadata=(("authorization", f"Bearer {KEY}"),),
    )
    try:
        assert (await stream.read()).WhichOneof("payload") == "admitted"
        async with asyncio.timeout(2):
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stream.read()
            assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        assert worker.active == 1 and worker.retained_task_count == 1
        release.set()
        await worker.idle.wait()
        assert worker.retained_task_count == 0
    finally:
        release.set()
        stream.cancel()
        await channel.close()
        await server.stop(0)
        await worker.close()


async def test_separate_worker_process_native_png_and_sigterm(tmp_path: Path) -> None:
    """Prove the CLI hosts real CPU generation outside the client process and exits on SIGTERM."""
    pytest.importorskip("grpc")
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    if os.name == "nt":
        pytest.skip("SIGTERM drain contract is verified in the isolated Linux CPU worker")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "finserve.multimodal.visual_rpc",
        "--address",
        "127.0.0.1:0",
        env=os.environ | {"JAX_PLATFORMS": "cpu", "FINSERVE_VISUAL_WORKER_KEY": KEY},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    client: VisualRPCClient | None = None
    try:
        assert process.stdout is not None
        async with asyncio.timeout(10):
            ready = json.loads(await process.stdout.readline())
        assert ready["event"] == "visual_rpc_ready" and ready["model_revision"] == MODEL_REVISION
        client = VisualRPCClient(f"127.0.0.1:{ready['port']}", KEY)
        execution = attempt("process-smoke")
        result = await client.generate(execution)
        assert execution.worker_instance == ready["worker_instance"]
        assert result.png.startswith(b"\x89PNG\r\n\x1a\n")
        (tmp_path / "worker-output.png").write_bytes(result.png)
        process.terminate()
        async with asyncio.timeout(10):
            assert await process.wait() == 0
    finally:
        if client is not None:
            await client.close()
        if process.returncode is None:
            process.kill()
            await process.wait()
