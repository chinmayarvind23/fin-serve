"""Authenticated loopback gRPC worker with explicit native-work ownership and cancel barriers."""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import secrets
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from finserve.contracts.visual import VisualArtifact, VisualAttempt, VisualJobRequest
from finserve.multimodal.jax_generator import JAXVisualReference, VisualRequest, output_png
from finserve.multimodal.visual_wire import message_types

MODEL_REVISION = (
    "jax-visual-"
    + hashlib.sha256(Path(__file__).with_name("jax_generator.py").read_bytes()).hexdigest()
)
MAX_MESSAGE_BYTES = 131072
SERVICE = "finserve.visual.v1.VisualWorker"


@dataclass
class RPCExecution:
    """Retain admission identity even when the owning client task is cancelled mid-stream."""

    attempt: VisualAttempt
    worker_instance: str | None = None
    admitted: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    started: bool = False


@dataclass
class _AttemptRecord:
    """Never evict a tombstone: delayed starts cannot revive a cancelled attempt in this process."""

    revoked: bool = False
    task: asyncio.Task[VisualArtifact] | None = None
    drained: bool = False


async def _drain(task: asyncio.Task[VisualArtifact]) -> None:
    """Repeated cancellation cannot release native compute before its thread actually completes."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:
            break
    if not task.cancelled():
        task.exception()


class NativeVisualGenerator:
    """Reuse only the most recent seed's compiled model to put a hard bound on the model cache."""

    def __init__(self) -> None:
        """Defer JAX initialization until actual work is accepted in the isolated process."""
        self._model: JAXVisualReference | None = None

    def __call__(self, request: VisualJobRequest) -> VisualArtifact:
        """Measure initialization, synchronized generation and rendering separately on CPU."""
        started = time.perf_counter_ns()
        if self._model is None or self._model.seed != request.seed:
            self._model = JAXVisualReference(request.seed)
        initialized = time.perf_counter_ns()
        output = self._model.generate(VisualRequest(images=[request.image]))
        generated = time.perf_counter_ns()
        png = output_png(output)
        return VisualArtifact(
            png=png,
            sha256=hashlib.sha256(png).hexdigest(),
            model_sha256=output.model_sha256,
            initialization_ns=initialized - started,
            generation_ns=generated - initialized,
            rendering_ns=time.perf_counter_ns() - generated,
        )


class VisualWorker:
    """A single CPU compute owner; RPC cleanup and native execution have separate lifetimes."""

    def __init__(
        self,
        credential: str,
        generator: Callable[[VisualJobRequest], VisualArtifact] | None = None,
        *,
        max_attempts: int = 4096,
    ) -> None:
        """Bound lifetime attempt metadata; exhaustion requires a controlled process replacement."""
        if len(credential) < 24 or not 1 <= max_attempts <= 65536:
            raise ValueError("worker credential or attempt capacity is invalid")
        self._credential = credential
        self._generator = generator or NativeVisualGenerator()
        self._max_attempts = max_attempts
        self._records: dict[tuple[str, int], _AttemptRecord] = {}
        self._active = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self.instance = str(uuid4())
        self._grpc = importlib.import_module("grpc")
        self._wire = message_types()

    @property
    def active(self) -> int:
        """Expose owned compute for deterministic cleanup checks, not GPU utilization claims."""
        return self._active

    @property
    def retained_task_count(self) -> int:
        """Expose native tasks to detect retained result bytes or tracebacks after drain."""
        return sum(record.task is not None for record in self._records.values())

    async def _authorize(self, context: Any) -> None:
        """Use one internal bearer value and never put incoming metadata into error details."""
        values = [value for key, value in context.invocation_metadata() if key == "authorization"]
        if len(values) != 1 or not secrets.compare_digest(values[0], f"Bearer {self._credential}"):
            await context.abort(
                self._grpc.StatusCode.UNAUTHENTICATED, "worker authentication failed"
            )

    def _request(self, message: Any) -> VisualAttempt:
        """Validate protobuf defaults and exact pixel length before numerical allocation."""
        pixels = bytes(message.image_rgb)
        if len(pixels) != 192:
            raise ValueError("image must have exactly 192 RGB bytes")
        image = [
            [tuple(pixels[(y * 8 + x) * 3 : (y * 8 + x + 1) * 3]) for x in range(8)]
            for y in range(8)
        ]
        return VisualAttempt.model_validate(
            {
                "job_id": message.job_id,
                "generation": message.generation,
                "request": {
                    "image": image,
                    "model_revision": message.model_revision,
                    "seed": message.seed,
                    "timeout_seconds": message.timeout_seconds,
                },
            }
        )

    async def generate(self, message: Any, context: Any) -> None:
        """Hold admission across writes and drain native work if transport disappears."""
        await self._authorize(context)
        try:
            attempt = self._request(message)
        except ValueError:
            await context.abort(self._grpc.StatusCode.INVALID_ARGUMENT, "invalid visual request")
            return
        if attempt.request.model_revision != MODEL_REVISION:
            await context.abort(self._grpc.StatusCode.NOT_FOUND, "unknown visual revision")
        key = (attempt.job_id, attempt.generation)
        if key in self._records:
            await context.abort(self._grpc.StatusCode.ALREADY_EXISTS, "attempt cannot be replayed")
        if self._active or len(self._records) >= self._max_attempts:
            await context.abort(
                self._grpc.StatusCode.RESOURCE_EXHAUSTED, "worker capacity exhausted"
            )
        record = self._records[key] = _AttemptRecord()
        self._active += 1
        self.idle.clear()
        try:
            async with asyncio.timeout(attempt.request.timeout_seconds):
                await self._produce(record, attempt.request, context)
        except TimeoutError:
            await context.abort(self._grpc.StatusCode.DEADLINE_EXCEEDED, "visual budget exceeded")
        except asyncio.CancelledError:
            raise
        except Exception:
            if not context.done():
                await context.abort(self._grpc.StatusCode.INTERNAL, "visual stream failed")
        finally:
            if record.task is not None:
                await _drain(record.task)
                record.task = None
            record.drained = True
            self._active -= 1
            self.idle.set()

    async def _produce(
        self, record: _AttemptRecord, request: VisualJobRequest, context: Any
    ) -> None:
        """Include both writes in the declared budget and send bounded model failures."""
        await context.write(
            self._wire["VisualEvent"](
                sequence=0, admitted=self._wire["Admitted"](worker_instance=self.instance)
            )
        )
        if record.revoked:
            await context.abort(self._grpc.StatusCode.CANCELLED, "attempt revoked")
        record.task = asyncio.create_task(asyncio.to_thread(self._generator, request))
        try:
            artifact = await asyncio.shield(record.task)
        except Exception:
            await context.write(
                self._wire["VisualEvent"](
                    sequence=1, failure=self._wire["Failure"](code="generation_failed")
                )
            )
            return
        if not record.revoked:
            await context.write(
                self._wire["VisualEvent"](
                    sequence=1, artifact=self._wire["Artifact"](**artifact.model_dump())
                )
            )

    async def cancel(self, message: Any, context: Any) -> Any:
        """Revoke delayed starts; instance mismatch is ambiguity, never cleanup ACK."""
        await self._authorize(context)
        if message.worker_instance != self.instance:
            return self._wire["AttemptStatus"](worker_instance=self.instance, state="unknown")
        key = (str(message.job_id), int(message.generation))
        if not key[0] or len(key[0]) > 128 or key[1] < 1:
            await context.abort(self._grpc.StatusCode.INVALID_ARGUMENT, "invalid attempt")
        if key not in self._records:
            if len(self._records) >= self._max_attempts:
                await context.abort(self._grpc.StatusCode.RESOURCE_EXHAUSTED, "attempt table full")
            self._records[key] = _AttemptRecord(revoked=True, drained=True)
        record = self._records[key]
        record.revoked = True
        if record.task is not None:
            await _drain(record.task)
            record.task = None
        # No task can start after revocation, including a generate handler paused in context.write.
        return self._wire["AttemptStatus"](worker_instance=self.instance, state="drained")

    async def close(self) -> None:
        """Shutdown owns all native tasks even if gRPC has already terminated their handlers."""
        for record in self._records.values():
            record.revoked = True
            if record.task is not None:
                await _drain(record.task)
                record.task = None

    async def start(self, address: str = "127.0.0.1:0") -> tuple[Any, int]:
        """Bind only loopback; remote production use requires an explicit TLS deployment design."""
        if not address.startswith("127.0.0.1:"):
            raise ValueError("visual worker currently supports loopback only")
        server = self._grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ],
            maximum_concurrent_rpcs=32,
        )
        handlers = {
            "GenerateVisual": self._grpc.unary_stream_rpc_method_handler(
                self.generate,
                request_deserializer=self._wire["GenerateRequest"].FromString,
                response_serializer=self._wire["VisualEvent"].SerializeToString,
            ),
            "CancelAttempt": self._grpc.unary_unary_rpc_method_handler(
                self.cancel,
                request_deserializer=self._wire["AttemptRequest"].FromString,
                response_serializer=self._wire["AttemptStatus"].SerializeToString,
            ),
        }
        server.add_generic_rpc_handlers(
            (self._grpc.method_handlers_generic_handler(SERVICE, handlers),)
        )
        port = server.add_insecure_port(address)
        if not port:
            raise RuntimeError("visual worker failed to bind")
        await server.start()
        return server, port


class VisualRPCClient:
    """Own one loopback channel; status polling never owns or cancels an accepted generation."""

    def __init__(self, address: str, credential: str) -> None:
        """Bound protobuf transport and require a trusted configured destination and service key."""
        if not address.startswith("127.0.0.1:") or len(credential) < 24:
            raise ValueError("visual client requires loopback and an internal credential")
        self._grpc = importlib.import_module("grpc")
        self._wire = message_types()
        self._metadata = (("authorization", f"Bearer {credential}"),)
        self._channel = self._grpc.aio.insecure_channel(
            address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
                ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
            ],
        )
        self._generate = self._channel.unary_stream(
            f"/{SERVICE}/GenerateVisual",
            request_serializer=self._wire["GenerateRequest"].SerializeToString,
            response_deserializer=self._wire["VisualEvent"].FromString,
        )
        self._cancel = self._channel.unary_unary(
            f"/{SERVICE}/CancelAttempt",
            request_serializer=self._wire["AttemptRequest"].SerializeToString,
            response_deserializer=self._wire["AttemptStatus"].FromString,
        )

    async def generate(self, execution: RPCExecution) -> VisualArtifact:
        """Require admitted+terminal frames and verify artifact bytes before success."""
        attempt = execution.attempt
        execution.started = True
        request = self._wire["GenerateRequest"](
            job_id=attempt.job_id,
            generation=attempt.generation,
            image_rgb=bytes(
                channel for row in attempt.request.image for rgb in row for channel in rgb
            ),
            model_revision=attempt.request.model_revision,
            seed=attempt.request.seed,
            timeout_seconds=attempt.request.timeout_seconds,
        )
        call = self._generate(
            request, timeout=attempt.request.timeout_seconds, metadata=self._metadata
        )
        artifact: VisualArtifact | None = None
        sequence = 0
        try:
            async for event in call:
                kind = event.WhichOneof("payload")
                if event.sequence != sequence or sequence > 1:
                    raise RuntimeError("invalid visual event sequence")
                if sequence == 0 and kind == "admitted" and event.admitted.worker_instance:
                    execution.worker_instance = str(event.admitted.worker_instance)
                    execution.admitted.set()
                elif sequence == 1 and kind == "artifact":
                    artifact = VisualArtifact(
                        **{
                            name: getattr(event.artifact, name)
                            for name in VisualArtifact.model_fields
                        }
                    )
                    if hashlib.sha256(artifact.png).hexdigest() != artifact.sha256:
                        raise RuntimeError("visual artifact integrity failed")
                else:
                    raise RuntimeError("visual worker failed or violated protocol")
                sequence += 1
            if artifact is None:
                raise RuntimeError("visual stream ended without artifact")
            return artifact
        finally:
            call.cancel()

    async def cancel(self, execution: RPCExecution, deadline_seconds: float = 120) -> bool:
        """Require same-instance drain ACK; transport cancellation alone proves nothing."""
        if execution.worker_instance is None:
            return False
        status = await self._cancel(
            self._wire["AttemptRequest"](
                job_id=execution.attempt.job_id,
                generation=execution.attempt.generation,
                worker_instance=execution.worker_instance,
            ),
            timeout=deadline_seconds,
            metadata=self._metadata,
        )
        return status.worker_instance == execution.worker_instance and status.state == "drained"

    async def close(self) -> None:
        """Channel shutdown is separate from worker drain and must follow coordinator cleanup."""
        await self._channel.close()


async def _serve(address: str, credential: str) -> None:
    """Keep CPU worker lifetime independent from HTTP clients and drain on process shutdown."""
    worker = VisualWorker(credential)
    server, port = await worker.start(address)
    task = asyncio.current_task()
    if task is not None and os.name != "nt":
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    print(
        json.dumps(
            {
                "event": "visual_rpc_ready",
                "port": port,
                "worker_instance": worker.instance,
                "model_revision": MODEL_REVISION,
            }
        ),
        flush=True,
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(0)
        await worker.close()


def main() -> None:
    """Run the optional worker with credentials from environment, never command-line history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:50061")
    args = parser.parse_args()
    try:
        asyncio.run(_serve(args.address, os.environ["FINSERVE_VISUAL_WORKER_KEY"]))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass


if __name__ == "__main__":
    main()
