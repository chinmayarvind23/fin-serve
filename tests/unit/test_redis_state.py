"""Failure and privacy checks for the optional Redis state boundary."""

import asyncio
import time
from typing import cast

import httpx
import pytest

from finserve.cache.redis_state import RedisClient, RedisState, from_url
from finserve.gateway.app import create_app


class StubRedis:
    """Fault injection for adapter policy; real Redis integration checks Lua atomicity."""

    def __init__(self) -> None:
        """Expose deterministic outputs and close observations to focused tests."""
        self.result: object = [1, 999]
        self.value: object = None
        self.failure = False
        self.closed = False
        self.calls = 0

    async def eval(self, script: str, numkeys: int, *args: str | int) -> object:
        """Count transitions so unauthenticated requests cannot consume quota."""
        self.calls += 1
        if self.failure:
            raise ConnectionError("private Redis address")
        return self.result

    async def get(self, key: str) -> object:
        """A failed optional read should leave serving available."""
        if self.failure:
            raise ConnectionError("private Redis address")
        return self.value

    async def set(self, key: str, value: bytes, *, ex: int) -> object:
        """Writes must carry an explicit TTL and bounded bytes."""
        if self.failure:
            raise ConnectionError("private Redis address")
        self.value = value
        return True

    async def aclose(self) -> None:
        """Make shutdown pool ownership observable."""
        self.closed = True


def state(client: StubRedis) -> RedisState:
    """The in-memory test boundary implements the same small optional-client protocol."""
    return RedisState(cast(RedisClient, client), b"test-secret-at-least-16-bytes")


async def test_gateway_quota_order_and_failure() -> None:
    """Auth comes first; denied or unavailable quotas never leak capacity or server details."""
    backend = StubRedis()
    app = create_app(api_key="secret", rate_limiter=state(backend))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            payload = {"prompt": "hello", "stream": False}
            assert (await client.post("/v1/completions", json=payload)).status_code == 401
            assert backend.calls == 0
            headers = {"Authorization": "Bearer secret"}
            backend.result = [0, 1500]
            denied = await client.post("/v1/completions", json=payload, headers=headers)
            assert denied.status_code == 429 and denied.headers["retry-after"] == "2"
            backend.failure = True
            unavailable = await client.post("/v1/completions", json=payload, headers=headers)
            assert unavailable.status_code == 503 and "private" not in unavailable.text
            backend.failure = False
            backend.result = [1, 500]
            assert (
                await client.post("/v1/completions", json=payload, headers=headers)
            ).status_code == 200
    assert backend.closed


@pytest.mark.parametrize("result", [[2, 10], [1, -1], [1], "invalid"])
async def test_invalid_quota_output(result: object) -> None:
    """Malformed Redis replies cannot silently allow a request."""
    backend = StubRedis()
    backend.result = result
    with pytest.raises(ValueError):
        await state(backend).allow("principal")


async def test_cache_limits_and_failure_policy() -> None:
    """Only bounded bytes qualify as hits, while any outage remains a cache miss."""
    backend = StubRedis()
    cache = state(backend)
    assert await cache.remember(b"identity", b"answer")
    assert await cache.cached(b"identity") == b"answer"
    backend.value = b"x" * 1_048_577
    assert await cache.cached(b"identity") is None
    backend.failure = True
    assert await cache.cached(b"identity") is None
    assert not await cache.remember(b"identity", b"answer")
    for value, ttl in ((b"answer", 0), (b"x" * 1_048_577, 60)):
        with pytest.raises(ValueError):
            await cache.remember(b"identity", value, ttl)
    assert "sensitive prompt" not in cache.key("response", b"sensitive prompt")
    assert cache.key("response", b"a") != cache.key("quota", b"a")


async def test_quota_deadline() -> None:
    """A stalled client is canceled by the total deadline, independent of socket timeouts."""

    class Stalled(StubRedis):
        async def eval(self, script: str, numkeys: int, *args: str | int) -> object:
            """Wait forever until the adapter's deadline cancels this coroutine."""
            await asyncio.Event().wait()
            return [1, 100]

    cache = RedisState(Stalled(), b"test-secret-long-enough", timeout_seconds=0.001)
    with pytest.raises(TimeoutError):
        await cache.allow("principal")


async def test_gateway_deadline_includes_quota_and_expired_body() -> None:
    """A short request budget interrupts a stalled quota, and an expired budget skips it."""

    class Stalled(StubRedis):
        async def eval(self, script: str, numkeys: int, *args: str | int) -> object:
            """Track cancellation rather than asserting a scheduler-sensitive wall duration."""
            self.calls += 1
            try:
                await asyncio.Event().wait()
            finally:
                self.closed = True

    backend = Stalled()
    app = create_app(rate_limiter=state(backend))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        result = await client.post(
            "/v1/completions", json={"prompt": "hello", "timeout_seconds": 0.005}
        )
    assert result.status_code == 504 and backend.closed
    calls = backend.calls
    response = await app.state.serving.quota_response("expired", time.perf_counter() - 1)
    assert response.status_code == 504 and backend.calls == calls


async def test_lifespan_failure_still_closes_redis() -> None:
    """Exceptions crossing the lifespan yield must not skip network resource shutdown."""
    backend = StubRedis()
    app = create_app(rate_limiter=state(backend))
    with pytest.raises(RuntimeError, match="lifespan failure"):
        async with app.router.lifespan_context(app):
            raise RuntimeError("lifespan failure")
    assert backend.closed


def test_invalid_redis_url() -> None:
    """Reject non-Redis endpoints before importing or constructing an optional client."""
    with pytest.raises(ValueError):
        from_url("https://invalid", b"test-secret-long-enough")


@pytest.mark.parametrize(
    "secret,limit,window,timeout",
    [
        (b"short", 1, 100, 1),
        (b"a" * 16, 0, 100, 1),
        (b"a" * 16, 1, 99, 1),
        (b"a" * 16, 1, 100, float("nan")),
    ],
)
def test_configuration_bounds(secret: bytes, limit: int, window: int, timeout: float) -> None:
    """Reject weak key isolation and unbounded/invalid state budgets before connecting."""
    with pytest.raises(ValueError):
        RedisState(StubRedis(), secret, limit=limit, window_ms=window, timeout_seconds=timeout)
