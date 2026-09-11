"""Real Redis atomicity/TTL checks; opt in using an isolated FINSERVE_TEST_REDIS_URL."""

import asyncio
import os
import secrets

import pytest

from finserve.cache.redis_state import from_url


async def test_real_atomic_quota_and_expiring_cache() -> None:
    """Concurrent callers share one limit and expiry releases it without clearing unrelated keys."""
    url = os.getenv("FINSERVE_TEST_REDIS_URL")
    if not url:
        pytest.skip("isolated Redis not configured")
    cache = from_url(url, secrets.token_bytes(32), limit=7, window_ms=1000)
    try:
        decisions = await asyncio.gather(*(cache.allow("tenant") for _ in range(20)))
        assert sum(item.allowed for item in decisions) == 7
        assert all(0 <= item.retry_after_ms <= 1000 for item in decisions)
        assert await cache.remember(b"job-digest", b"complete", ttl_seconds=1)
        assert await cache.cached(b"job-digest") == b"complete"
        await asyncio.sleep(1.05)
        assert (await cache.allow("tenant")).allowed
        assert await cache.cached(b"job-digest") is None
    finally:
        await cache.close()
