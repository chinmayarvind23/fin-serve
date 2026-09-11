"""The ordinary inference API pins each stream to a durable warm-route generation."""

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from contextvars import ContextVar

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.engines.openai_adapter import OpenAICompletionEngine
from finserve.engines.vllm_adapter import VLLMEngine
from finserve.gateway.app import create_app, error_response
from finserve.reliability.warm_routes import RouteSnapshot, WarmRouteStore

request_route: ContextVar[RouteSnapshot | None] = ContextVar("finserve_warm_route", default=None)


class WarmRouteEngine:
    """HTTP proxy clients own zero GPUs; backend processes remain independently managed."""

    # Preserve requests through the pinned route; each selected adapter checks its own capability.
    supports_output_constraints = True

    def __init__(self, store: WarmRouteStore, deployment_id: str) -> None:
        """The registry's fixed endpoint cap also bounds the lifetime client pool."""
        self.store, self.deployment_id = store, deployment_id
        self.clients: dict[str, OpenAICompletionEngine] = {}

    async def _client(self, snapshot: RouteSnapshot) -> OpenAICompletionEngine:
        """Use registered revisions and keep resolved credentials in memory."""
        if snapshot.revision_id not in self.clients:
            backend = await asyncio.to_thread(self.store.backend, snapshot.revision_id)
            configuration = backend.configuration
            key = os.getenv(configuration.credential_env) if configuration.credential_env else None
            if configuration.credential_env and not key:
                raise ValueError("configured backend credential is unavailable")
            # Concurrent first requests may finish lookup together; create only one pool.
            if snapshot.revision_id not in self.clients:
                adapter = (
                    VLLMEngine if backend.revision.engine == "vllm" else OpenAICompletionEngine
                )
                self.clients[snapshot.revision_id] = adapter(configuration.base_url, api_key=key)
        return self.clients[snapshot.revision_id]

    async def stream(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """New requests see cutover; an admitted stream keeps its pinned revision."""
        snapshot = request_route.get()
        if snapshot is None or snapshot.deployment_id != self.deployment_id:
            raise RuntimeError("inference must be pinned by route middleware")
        client = await self._client(snapshot)
        async with aclosing(client.stream(request)) as stream:
            async for token in stream:
                yield token

    async def close(self) -> None:
        """Close every registered proxy pool after application request draining finishes."""
        await asyncio.gather(*(client.close() for client in self.clients.values()))


class WarmRouteMiddleware:
    """Response identity and engine selection share the exact same request snapshot."""

    def __init__(self, app: ASGIApp, store: WarmRouteStore, deployment_id: str) -> None:
        """Pure ASGI middleware preserves context through streaming and cancellation cleanup."""
        self.app, self.store, self.deployment_id = app, store, deployment_id

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Stamp server-owned route identity; never accept routing metadata from caller headers."""
        if scope["type"] != "http" or scope["path"] not in {
            "/v1/completions",
            "/v1/chat/completions",
        }:
            await self.app(scope, receive, send)
            return
        try:
            scope.setdefault("state", {}).setdefault("finserve_received", time.perf_counter())
            async with asyncio.timeout(5):
                snapshot = await asyncio.to_thread(self.store.snapshot, self.deployment_id)
        except Exception:
            await error_response("ROUTE_UNAVAILABLE", 503)(scope, receive, send)
            return
        token = request_route.set(snapshot)

        async def identity_send(message: Message) -> None:
            """Headers describe the route actually pinned before this response began streaming."""
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-finserve-revision", snapshot.revision_id.encode()),
                        (b"x-finserve-revision-digest", snapshot.revision_digest.encode()),
                        (b"x-finserve-route-generation", str(snapshot.generation).encode()),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, identity_send)
        finally:
            request_route.reset(token)


def create_warm_app(
    store: WarmRouteStore,
    deployment_id: str,
    *,
    api_key: str | None = None,
    max_concurrency: int = 16,
) -> FastAPI:
    """Reuse admission, SSE and cancellation behavior while replacing only route selection."""
    state = store.snapshot(deployment_id)
    backend = store.backend(state.revision_id)
    app = create_app(
        WarmRouteEngine(store, deployment_id),
        model=backend.configuration.model,
        api_key=api_key,
        max_concurrency=max_concurrency,
    )
    app.add_middleware(WarmRouteMiddleware, store=store, deployment_id=deployment_id)
    return app
