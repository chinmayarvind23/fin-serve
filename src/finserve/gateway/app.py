"""FastAPI streaming ingress with bounded leases, explicit errors, and honest timing."""

import asyncio
import hmac
import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry.context import Context
from opentelemetry.trace import NoOpTracer, StatusCode, Tracer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import ValidationError
from starlette.types import Receive, Scope, Send

from finserve.cache.redis_state import RateLimiter
from finserve.contracts.inference import ChatRequest, EngineToken, InferenceRequest
from finserve.contracts.vision import VISION_MODEL
from finserve.engines.base import Engine
from finserve.engines.fixture import FixtureEngine
from finserve.engines.vision_openai import VisionEngine
from finserve.gateway.admission import Admission
from finserve.gateway.body_limit import BodyLimit
from finserve.multimodal.jobs import VisualJobCoordinator, VisualJobStore
from finserve.telemetry.metrics import Metrics
from finserve.telemetry.propagation import next_in_span
from finserve.telemetry.tracing import TraceRuntime
from finserve.telemetry.tracing import from_env as tracing_from_env


def error_response(code: str, status: int, request_id: str = "") -> JSONResponse:
    """Stable codes allow clients to classify failure without exposing backend details."""
    return JSONResponse(
        {
            "error": {
                "code": code,
                "message": code.replace("_", " ").lower(),
                "retryable": status in {429, 503},
                "request_id": request_id,
            }
        },
        status_code=status,
    )


def sse(payload: dict[str, object]) -> str:
    """JSON escaping prevents model text from injecting SSE control fields."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


@dataclass
class Lease:
    """Idempotent ownership covers disconnects before an async generator ever starts."""

    admission: Admission
    metrics: Metrics
    released: bool = False
    started: bool = False

    def release(self) -> None:
        """Response and generator both clean up, but only the first returns capacity."""
        if not self.released:
            self.released = True
            self.admission.release()
            self.metrics.active.dec()
            if not self.started:
                self.metrics.requests.labels(outcome="cancelled_before_stream").inc()


class OwnedStreamingResponse(StreamingResponse):
    """ASGI send failures must close generation even when framework iteration aborts."""

    def __init__(self, stream: AsyncGenerator[str], lease: Lease, request_id: str) -> None:
        """Keep an explicit iterator reference instead of relying on generator finalization."""
        super().__init__(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )
        self.stream = stream
        self.lease = lease

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Header/body transport errors follow the same bounded cleanup path."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.stream.aclose()
            finally:
                self.lease.release()


@dataclass
class Serving:
    """Application-scoped dependencies avoid mutable process-global request state."""

    engine: Engine
    admission: Admission
    metrics: Metrics
    model: str
    api_key: str | None
    tracer: Tracer
    rate_limiter: RateLimiter | None = None

    async def quota_response(self, request_id: str, deadline: float) -> Response | None:
        """Rate limiting is an optional pre-admission hop with explicit failure semantics."""
        if self.rate_limiter is None:
            return None
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return error_response("DEADLINE_EXCEEDED", 504, request_id)
        try:
            async with asyncio.timeout(min(remaining, 0.3)):
                decision = await self.rate_limiter.allow(self.api_key or "anonymous")
        except Exception:
            if time.perf_counter() >= deadline:
                return error_response("DEADLINE_EXCEEDED", 504, request_id)
            self.metrics.requests.labels(outcome="quota_unavailable").inc()
            return error_response("QUOTA_UNAVAILABLE", 503, request_id)
        if decision.allowed:
            return None
        self.metrics.requests.labels(outcome="rate_limited").inc()
        response = error_response("RATE_LIMITED", 429, request_id)
        response.headers["Retry-After"] = str(max(1, (decision.retry_after_ms + 999) // 1000))
        return response

    async def events(
        self, request: InferenceRequest, received: float, chat: bool, lease: Lease | None = None
    ) -> AsyncGenerator[str]:
        """Never retry visible output; close the iterator on failure or client cancellation."""
        outcome, count, first = "cancelled", 0, None
        finish_reason = "length"
        lease = lease or Lease(self.admission, self.metrics)
        lease.started = True
        iterator: AsyncIterator[EngineToken] | None = None
        span = self.tracer.start_span(
            "finserve.inference",
            context=Context(),
            attributes={
                "gen_ai.request.model": request.model,
                "finserve.max_tokens": request.max_tokens,
                "finserve.prompt_characters": len(request.prompt),
            },
        )
        try:
            remaining = max(0, request.timeout_seconds - (time.perf_counter() - received))
            if remaining <= 0:
                raise TimeoutError
            iterator = self.engine.stream(request)
            async with asyncio.timeout(remaining):
                while True:
                    try:
                        token = await next_in_span(iterator, span)
                    except StopAsyncIteration:
                        break
                    if token.finish_reason is not None:
                        finish_reason = token.finish_reason
                    count += token.generated_tokens
                    self.metrics.tokens.inc(token.generated_tokens)
                    if token.text and first is None:
                        first = time.perf_counter() - received
                        self.metrics.ttft.observe(first)
                    choice: dict[str, object] = {"index": 0, "finish_reason": None}
                    choice["delta" if chat else "text"] = (
                        {"content": token.text} if chat else token.text
                    )
                    yield sse(
                        {
                            "id": request.request_id,
                            "model": request.model,
                            "object": "chat.completion.chunk" if chat else "text_completion",
                            "choices": [choice],
                        }
                    )
            yield sse(
                {
                    "id": request.request_id,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "delta" if chat else "text": {} if chat else "",
                        }
                    ],
                    "usage": {"completion_tokens": count},
                    "finserve": {"server_ttft_seconds": first},
                }
            )
            yield "data: [DONE]\n\n"
            outcome = "success"
        except Exception as exc:
            outcome = "timeout" if isinstance(exc, TimeoutError) else "engine_failed"
            yield sse(
                {
                    "error": {
                        "code": outcome.upper(),
                        "request_id": request.request_id,
                        "retryable": False,
                    }
                }
            )
        finally:
            try:
                close = getattr(iterator, "aclose", None)
                if close is not None:
                    await close()
            finally:
                lease.release()
                self.metrics.requests.labels(outcome=outcome).inc()
                self.metrics.duration.observe(time.perf_counter() - received)
                span.set_attribute("finserve.outcome", outcome)
                span.set_attribute("gen_ai.usage.output_tokens", count)
                span.set_status(StatusCode.OK if outcome == "success" else StatusCode.ERROR)
                span.end()

    async def respond(
        self, payload: InferenceRequest, request: Request, chat: bool = False
    ) -> Response:
        """Validate before admission; the stream owns its lease until consumption finishes."""
        received = float(
            request.scope.get("state", {}).get("finserve_received", time.perf_counter())
        )
        if self.api_key is not None and not hmac.compare_digest(
            request.headers.get("authorization", "").encode(), f"Bearer {self.api_key}".encode()
        ):
            return error_response("UNAUTHORIZED", 401, payload.request_id)
        if payload.model != self.model:
            return error_response("MODEL_NOT_FOUND", 404, payload.request_id)
        quota_error = await self.quota_response(
            payload.request_id, received + payload.timeout_seconds
        )
        if quota_error is not None:
            return quota_error
        if not self.admission.acquire():
            self.metrics.requests.labels(outcome="overloaded").inc()
            return error_response("OVERLOADED", 429, payload.request_id)
        self.metrics.active.inc()
        lease = Lease(self.admission, self.metrics)
        stream = self.events(payload, received, chat, lease)
        try:
            if payload.stream:
                return OwnedStreamingResponse(stream, lease, payload.request_id)
            return await collect_response(stream, payload, chat)
        except BaseException:
            lease.release()
            raise


async def collect_response(
    stream: AsyncGenerator[str], payload: InferenceRequest, chat: bool
) -> Response:
    """Reuse stream accounting for non-stream clients so terminal semantics stay identical."""
    output: list[str] = []
    usage: dict[str, object] = {}
    finish_reason = "length"
    async with aclosing(stream):
        async for frame in stream:
            if frame == "data: [DONE]\n\n":
                continue
            data = json.loads(frame[6:])
            if "error" in data:
                return JSONResponse(data, status_code=502)
            for choice in data.get("choices", []):
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                output.append(
                    choice.get("delta", {}).get("content", "") if chat else choice.get("text", "")
                )
            usage = data.get("usage", usage)
    content = "".join(output)
    choice = {
        "index": 0,
        "finish_reason": finish_reason,
        "message" if chat else "text": {"role": "assistant", "content": content}
        if chat
        else content,
    }
    return JSONResponse(
        {"id": payload.request_id, "model": payload.model, "choices": [choice], "usage": usage}
    )


def create_app(
    engine: Engine | None = None,
    *,
    model: str = "reference",
    max_concurrency: int = 16,
    api_key: str | None = None,
    tracing: TraceRuntime | None = None,
    revision: str = "unrecorded",
    rate_limiter: RateLimiter | None = None,
    visual_jobs: VisualJobCoordinator | None = None,
    vision_engine: VisionEngine | None = None,
    vision_model: str = VISION_MODEL,
    vision_capacity: int = 1,
) -> FastAPI:
    """Tests inject engines; deployments select an explicit backend through the factory."""
    if (visual_jobs is not None or vision_engine is not None) and not api_key:
        raise ValueError("multimodal serving requires an API credential")
    serving = Serving(
        engine or FixtureEngine(),
        Admission(max_concurrency),
        Metrics(),
        model,
        api_key,
        tracing.tracer if tracing else NoOpTracer(),
        rate_limiter,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        """Close pooled clients when shutdown completes; Uvicorn handles request draining."""
        async with AsyncExitStack() as cleanup:
            # Every resource closes even if another close or the coordinator's startup fails.
            if tracing is not None:
                cleanup.push_async_callback(asyncio.to_thread, tracing.close)
            if rate_limiter is not None:
                cleanup.push_async_callback(rate_limiter.close)
            cleanup.push_async_callback(serving.engine.close)
            if vision_engine is not None:
                cleanup.push_async_callback(vision_engine.close)
            if visual_jobs is not None:
                cleanup.push_async_callback(visual_jobs.close)
                visual_jobs.start()
            yield

    app = FastAPI(title="FinServe", version="0.1.0", lifespan=lifespan)
    # Vision authenticates and acquires its own image slot before reading its larger bounded body.
    app.add_middleware(
        BodyLimit,
        delegated_paths=frozenset({"/v1/vision/completions"})
        if vision_engine is not None
        else frozenset(),
    )
    app.state.serving = serving
    if vision_engine is not None:
        from finserve.gateway.vision import register_vision_routes

        app.state.vision = register_vision_routes(
            app, vision_engine, vision_model, api_key, vision_capacity
        )
    if visual_jobs is not None and api_key is not None:
        from finserve.gateway.visual_routes import register_visual_routes

        register_visual_routes(app, visual_jobs, api_key)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        """Liveness makes no claim about downstream GPU readiness."""
        return {"status": "ok", "model": model, "revision": revision}

    @app.get("/metrics")
    async def metrics() -> Response:
        """Expose fixed-cardinality operational metrics without prompt data."""
        return Response(
            generate_latest(serving.metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @app.post("/v1/completions", response_model=None)
    async def completions(payload: InferenceRequest, request: Request) -> Response:
        """The OpenAI text surface exposes only the supported generation contract."""
        return await serving.respond(payload, request)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(payload: ChatRequest, request: Request) -> Response:
        """Reject aggregate prompt overflow rather than truncating conversation silently."""
        try:
            inference = payload.to_inference()
        except ValidationError:
            return error_response("CONTEXT_TOO_LARGE", 422)
        return await serving.respond(inference, request, chat=True)

    return app


def from_env() -> FastAPI:
    """Explicit fixture/reference modes prevent a fake engine masquerading as a real model."""
    backend = os.getenv("FINSERVE_ENGINE", "fixture")
    engine: Engine
    if backend == "pytorch":
        from finserve.engines.pytorch_reference import PyTorchReferenceEngine

        engine = PyTorchReferenceEngine(
            cached=os.getenv("FINSERVE_KV_CACHE", "1") == "1",
            device=os.getenv("FINSERVE_DEVICE", "cpu"),
        )
    elif backend == "fixture":
        engine = FixtureEngine()
    elif backend in {"vllm", "sglang"}:
        from finserve.engines.openai_adapter import OpenAICompletionEngine

        engine = OpenAICompletionEngine(
            os.environ["FINSERVE_ENGINE_URL"], api_key=os.getenv("FINSERVE_ENGINE_API_KEY")
        )
    elif backend == "ray-http":
        from finserve.engines.ray_http import RayHTTPEngine

        engine = RayHTTPEngine(
            os.environ["FINSERVE_ENGINE_URL"], api_key=os.getenv("FINSERVE_RAY_API_KEY")
        )
    else:
        raise ValueError(f"Unsupported FINSERVE_ENGINE: {backend}")
    visual_jobs = None
    if visual_target := os.getenv("FINSERVE_VISUAL_GRPC_TARGET"):
        from finserve.multimodal.visual_rpc import VisualRPCClient

        visual_jobs = VisualJobCoordinator(
            VisualJobStore(Path(os.environ["FINSERVE_VISUAL_DB"])),
            VisualRPCClient(visual_target, os.environ["FINSERVE_VISUAL_SERVICE_KEY"]),
        )
    rate_limiter = None
    if redis_url := os.getenv("FINSERVE_REDIS_URL"):
        from finserve.cache.redis_state import from_url

        rate_limiter = from_url(
            redis_url,
            os.environ["FINSERVE_REDIS_KEY_SECRET"].encode(),
            limit=int(os.getenv("FINSERVE_RATE_LIMIT", "120")),
            window_ms=int(os.getenv("FINSERVE_RATE_WINDOW_MS", "60000")),
        )
    vision_engine = None
    if vision_url := os.getenv("FINSERVE_VISION_ENGINE_URL"):
        from finserve.engines.vision_openai import OpenAIVisionEngine

        vision_engine = OpenAIVisionEngine(
            vision_url, api_key=os.getenv("FINSERVE_VISION_ENGINE_KEY")
        )
    capacity = int(os.getenv("FINSERVE_MAX_CONCURRENCY", "16"))
    vision_capacity = int(os.getenv("FINSERVE_VISION_CAPACITY", "1"))
    # Exporter threads are acquired last and explicitly returned if construction fails.
    tracing = tracing_from_env()
    try:
        return create_app(
            engine,
            model=os.getenv("FINSERVE_MODEL", "reference"),
            max_concurrency=capacity,
            api_key=os.getenv("FINSERVE_API_KEY"),
            tracing=tracing,
            revision=os.getenv("FINSERVE_REVISION", "unrecorded"),
            rate_limiter=rate_limiter,
            visual_jobs=visual_jobs,
            vision_engine=vision_engine,
            vision_model=os.getenv("FINSERVE_VISION_MODEL", VISION_MODEL),
            vision_capacity=vision_capacity,
        )
    except BaseException:
        if tracing is not None:
            tracing.close()
        raise
