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
from finserve.registry.model_assets import owned_disk
from finserve.reliability.warm_drain import ADMISSION_PROTOCOL, AdmissionLease, AdmissionTransport
from finserve.reliability.warm_routes import RouteSnapshot, WarmRouteStore

request_route: ContextVar[RouteSnapshot | None] = ContextVar("finserve_warm_route", default=None)
request_admission: ContextVar[AdmissionLease | None] = ContextVar(
    "finserve_warm_admission", default=None
)


class WarmRouteEngine:
    """HTTP proxy clients own zero GPUs; backend processes remain independently managed."""

    # Preserve requests through the pinned route; each selected adapter checks its own capability.
    supports_output_constraints = True

    def __init__(self, store: WarmRouteStore, deployment_id: str) -> None:
        """Bound cached pools while keeping admitted backend streams locally owned."""
        self.store, self.deployment_id = store, deployment_id
        self.clients: dict[str, OpenAICompletionEngine] = {}
        self.client_users: dict[str, int] = {}
        self.client_limit = 32
        self.client_lock = asyncio.Lock()

    async def _client(self, snapshot: RouteSnapshot) -> OpenAICompletionEngine:
        """Use registered revisions and keep resolved credentials in memory."""
        async with self.client_lock:
            if snapshot.revision_id not in self.clients:
                admission = request_admission.get()
                backend = (
                    admission.backend
                    if admission is not None
                    else await asyncio.to_thread(self.store.backend, snapshot.revision_id)
                )
                configuration = backend.configuration
                key = (
                    os.getenv(configuration.credential_env)
                    if configuration.credential_env
                    else None
                )
                if configuration.credential_env and not key:
                    raise ValueError("configured backend credential is unavailable")
                await self._evict_retired()
                if len(self.clients) >= self.client_limit:
                    raise RuntimeError("warm backend client limit reached")
                # The client lock serializes allocation and retired pool closure.
                if snapshot.revision_id not in self.clients:
                    adapter = (
                        VLLMEngine if backend.revision.engine == "vllm" else OpenAICompletionEngine
                    )
                    self.clients[snapshot.revision_id] = adapter(
                        configuration.base_url, api_key=key, transport=AdmissionTransport()
                    )
            return self.clients[snapshot.revision_id]

    async def _evict_retired(self) -> None:
        """Under the client lock, close only idle pools whose retirement is irreversible."""
        if len(self.clients) < self.client_limit:
            return
        for revision_id, client in tuple(self.clients.items()):
            retired = await asyncio.to_thread(self.store.is_retired, revision_id)
            if retired and not self.client_users.get(revision_id, 0):
                # New users wait for this lock and cannot obtain the closing adapter.
                await client.close()
                del self.clients[revision_id]
                return

    async def stream(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """New requests see cutover; an admitted stream keeps its pinned revision."""
        snapshot = request_route.get()
        if snapshot is None or snapshot.deployment_id != self.deployment_id:
            raise RuntimeError("inference must be pinned by route middleware")
        admission = request_admission.get()
        if snapshot.admission_protocol == ADMISSION_PROTOCOL and (
            admission is None or admission.snapshot != snapshot
        ):
            raise RuntimeError("durable route requires its exact admission lease")
        revision_id = snapshot.revision_id
        self.client_users[revision_id] = self.client_users.get(revision_id, 0) + 1
        try:
            client = await self._client(snapshot)
            async with aclosing(client.stream(request)) as stream:
                async for token in stream:
                    yield token
        finally:
            # An uncertain HTTP close pins the local pool as well as its durable obligation.
            if admission is None or admission.verified_closed:
                remaining = self.client_users[revision_id] - 1
                if remaining:
                    self.client_users[revision_id] = remaining
                else:
                    self.client_users.pop(revision_id, None)

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
        admission = None
        try:
            scope.setdefault("state", {}).setdefault("finserve_received", time.perf_counter())
            async with asyncio.timeout(5):
                if self.store.admission_protocol == ADMISSION_PROTOCOL:
                    serving = getattr(getattr(scope.get("app"), "state", None), "serving", None)
                    engine = getattr(serving, "engine", None)
                    if (
                        not isinstance(engine, WarmRouteEngine)
                        or engine.store.identity != self.store.identity
                        or engine.deployment_id != self.deployment_id
                    ):
                        raise RuntimeError("durable admission requires the bound warm engine")
                    admission = await owned_disk(lambda: self.store.admit(self.deployment_id))
                    snapshot = admission.snapshot
                else:
                    snapshot = await asyncio.to_thread(self.store.snapshot, self.deployment_id)
        except Exception:
            await error_response("ROUTE_UNAVAILABLE", 503)(scope, receive, send)
            return
        token = request_route.set(snapshot)
        admission_token = request_admission.set(admission)

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
            request_admission.reset(admission_token)
            if admission is not None and admission.verified_closed:
                await owned_disk(lambda: self.store.finish_admission(admission))


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
