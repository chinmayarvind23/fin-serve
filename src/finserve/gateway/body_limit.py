"""Bound actual ASGI bytes, including chunked bodies with no Content-Length."""

import time

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyLimit:
    """Buffer at most the limit before routing so oversized inputs cannot reach JSON parsing."""

    def __init__(
        self, app: ASGIApp, max_bytes: int = 131072, delegated_paths: frozenset[str] = frozenset()
    ) -> None:
        """A fixed small cap covers text requests; media gets separate upload contracts."""
        self.app = app
        self.max_bytes = max_bytes
        self.delegated_paths = delegated_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Preserve disconnect semantics after replaying the bounded request body once."""
        if scope["type"] == "http":
            scope.setdefault("state", {}).setdefault("finserve_received", time.perf_counter())
        if (
            scope["type"] != "http"
            or scope["method"] not in {"POST", "PUT", "PATCH"}
            or scope["path"] in self.delegated_paths
        ):
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_bytes:
                response = JSONResponse({"error": {"code": "BODY_TOO_LARGE"}}, status_code=413)
                await response(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        replayed = False

        async def replay() -> Message:
            """Replay only once; subsequent receives must observe real client disconnects."""
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
