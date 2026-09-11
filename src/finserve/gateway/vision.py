"""Single-image ingress owns preprocessing, generation and disconnect cleanup explicitly."""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send

from finserve.contracts.inference import EngineToken
from finserve.contracts.vision import MAX_VISION_BODY_BYTES, VisionRequest
from finserve.engines.vision_openai import VisionEngine
from finserve.gateway.admission import Admission
from finserve.multimodal.images import PreparedImage, decode_inline_png, prepare_png


def vision_error(code: str, status: int) -> JSONResponse:
    """Never echo validation inputs, upstream bodies, credentials or image contents."""
    return JSONResponse({"error": {"code": code, "retryable": status in (429, 503)}}, status)


async def drain[T](task: asyncio.Task[T]) -> T:
    """Repeated cancellation cannot return admission while a native CPU thread still runs."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


@dataclass
class VisionLease:
    """Exactly one release covers failures before and after response iteration begins."""

    admission: Admission
    released: bool = False

    def release(self) -> None:
        """Make response and route cleanup idempotent without hiding counter underflow."""
        if not self.released:
            self.released = True
            self.admission.release()


class VisionResponse(StreamingResponse):
    """Own generation even if ASGI fails to send headers or a streamed body."""

    def __init__(
        self, iterator: AsyncGenerator[str], lease: VisionLease, request_id: str, deadline: float
    ) -> None:
        """Retain the iterator before headers so early disconnects cannot orphan a lease."""
        super().__init__(
            iterator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )
        self.iterator, self.lease = iterator, lease
        self.deadline = deadline

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Drain owned cleanup under cancellation, including transport send failures."""

        async def bounded_send(message: Message) -> None:
            """A stalled downstream write spends the same deadline as upstream generation."""
            async with asyncio.timeout_at(self.deadline):
                await send(message)

        try:
            await super().__call__(scope, receive, bounded_send)
        finally:
            try:
                await drain(asyncio.create_task(self.iterator.aclose()))
            finally:
                self.lease.release()


@dataclass
class VisionServing:
    """One explicit image-capable model and bounded local capacity; no text-engine fallback."""

    engine: VisionEngine
    model: str
    api_key: str | None
    admission: Admission

    async def infer(self, http: Request) -> Response:
        """Authenticate and cap bytes before JSON, then hold capacity through native cleanup."""
        received = asyncio.get_running_loop().time()
        if self.api_key is not None:
            headers = http.headers.getlist("authorization")
            if len(headers) != 1 or not hmac.compare_digest(
                headers[0].encode(), ("Bearer " + self.api_key).encode()
            ):
                return vision_error("UNAUTHORIZED", 401)
        if not self.admission.acquire():
            return vision_error("OVERLOADED", 429)
        lease, transferred = VisionLease(self.admission), False
        try:
            response = await self._admitted(http, received, lease)
            transferred = isinstance(response, VisionResponse)
            return response
        finally:
            if not transferred:
                lease.release()

    async def _admitted(self, http: Request, received: float, lease: VisionLease) -> Response:
        """Bound authenticated slow uploads as well as image decoding under one lease."""
        try:
            data = bytearray()
            async with asyncio.timeout(10):
                async for chunk in http.stream():
                    if len(data) + len(chunk) > MAX_VISION_BODY_BYTES:
                        return vision_error("BODY_TOO_LARGE", 413)
                    data.extend(chunk)
            request = VisionRequest.model_validate_json(data)
        except (ValidationError, ValueError):
            return vision_error("INVALID_VISION_REQUEST", 422)
        except TimeoutError:
            return vision_error("BODY_TIMEOUT", 408)
        if request.model != self.model:
            return vision_error("MODEL_MODALITY_UNAVAILABLE", 404)
        deadline = received + request.timeout_seconds
        try:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError
            image = await self._prepare(request, deadline)
            if request.stream:
                response = VisionResponse(
                    self.events(request, image, deadline), lease, request.request_id, deadline
                )
                return response
            text, count, reason = "", 0, "stop"
            async for token in self.tokens(request, image, deadline):
                text += token.text
                count += token.generated_tokens
                reason = token.finish_reason or reason
            return JSONResponse(
                {
                    "id": request.request_id,
                    "model": request.model,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": reason,
                        }
                    ],
                    "usage": {"completion_tokens": count},
                    "image_sha256": image.sha256,
                }
            )
        except ValueError:
            return vision_error("INVALID_PNG", 422)
        except TimeoutError:
            return vision_error("DEADLINE_EXCEEDED", 504)
        except Exception:
            return vision_error("VISION_ENGINE_FAILED", 502)

    async def _prepare(self, request: VisionRequest, deadline: float) -> PreparedImage:
        """Offload bounded image decoding while retaining its slot until the thread finishes."""
        task = asyncio.create_task(
            asyncio.to_thread(prepare_png, decode_inline_png(request.image_png_base64))
        )
        try:
            async with asyncio.timeout_at(deadline):
                return await asyncio.shield(task)
        finally:
            # Drain does not replace the original timeout/cancellation with decoder errors.
            if not task.done():
                try:
                    await drain(task)
                except Exception:
                    pass

    async def tokens(
        self, request: VisionRequest, image: PreparedImage, deadline: float
    ) -> AsyncGenerator[EngineToken]:
        """Bound each upstream await; no timeout context may span consumer work at a yield."""
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError
        iterator = self.engine.stream(
            request.model_copy(update={"timeout_seconds": remaining}), image
        )
        terminal = False
        try:
            while True:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError
                try:
                    async with asyncio.timeout_at(deadline):
                        token = await anext(iterator)
                except StopAsyncIteration:
                    if not terminal:
                        raise RuntimeError("Vision engine omitted terminal accounting") from None
                    return
                if terminal or (token.generated_tokens and token.finish_reason is None):
                    raise RuntimeError("Invalid vision accounting sequence")
                terminal = token.finish_reason is not None
                if terminal and (token.text or token.generated_tokens > request.max_tokens):
                    raise RuntimeError("Invalid terminal vision accounting")
                yield token
        finally:
            await drain(asyncio.create_task(iterator.aclose()))

    async def events(
        self, request: VisionRequest, image: PreparedImage, deadline: float
    ) -> AsyncGenerator[str]:
        """Emit escaped chat SSE and withhold DONE on any incomplete engine response."""
        iterator = self.tokens(request, image, deadline)
        terminal: EngineToken | None = None
        try:
            async for token in iterator:
                if token.finish_reason:
                    terminal = token
                else:
                    yield _sse(
                        {
                            "id": request.request_id,
                            "model": request.model,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token.text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            if terminal is None:
                raise RuntimeError("Missing terminal vision event")
            yield _sse(
                {
                    "id": request.request_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": terminal.finish_reason}],
                    "usage": {"completion_tokens": terminal.generated_tokens},
                    "image_sha256": image.sha256,
                }
            )
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield _sse(
                {
                    "error": {
                        "code": "DEADLINE_EXCEEDED"
                        if isinstance(exc, TimeoutError)
                        else "VISION_ENGINE_FAILED",
                        "retryable": False,
                    }
                }
            )
        finally:
            await drain(asyncio.create_task(iterator.aclose()))

    async def close(self) -> None:
        """Application lifespan calls this after its HTTP request owners have drained."""
        await self.engine.close()


def _sse(payload: dict[str, object]) -> str:
    """JSON escaping prevents generated text from injecting event boundaries."""
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


def register_vision_routes(
    app: FastAPI, engine: VisionEngine, model: str, api_key: str | None = None, capacity: int = 1
) -> VisionServing:
    """Return explicit lifespan ownership; outer body middleware must allow this route's cap."""
    serving = VisionServing(engine, model, api_key, Admission(capacity))
    app.add_api_route(
        "/v1/vision/completions", serving.infer, methods=["POST"], response_model=None
    )
    return serving
