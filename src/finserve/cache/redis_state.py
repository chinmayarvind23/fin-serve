"""Optional bounded Redis quotas and ephemeral response bytes; Redis is never durable truth."""

import asyncio
import hashlib
import hmac
import importlib
from typing import Protocol, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

# Redis executes this counter/expiry transition atomically across gateway replicas.
# The window starts with its first request; denied attempts cannot extend the window.
QUOTA_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('PEXPIRE', KEYS[1], ARGV[2]) end
local ttl = redis.call('PTTL', KEYS[1])
if ttl < 0 then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  ttl = tonumber(ARGV[2])
end
return {count <= tonumber(ARGV[1]) and 1 or 0, ttl}
"""


class QuotaDecision(BaseModel):
    """Retry delay is server-side remaining window time, never a client clock subtraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    allowed: bool
    retry_after_ms: int = Field(ge=0)


class RateLimiter(Protocol):
    """Gateway quota failures are explicit unavailability; capacity admission stays separate."""

    async def allow(self, subject: str) -> QuotaDecision:
        """Count one authorized logical request before engine admission."""
        ...

    async def close(self) -> None:
        """Release any owned network pool on application shutdown."""
        ...


class RedisClient(Protocol):
    """Keep the optional Redis dependency behind a narrow, independently testable boundary."""

    async def eval(self, script: str, numkeys: int, *args: str | int) -> object:
        """Execute a bounded atomic state transition."""
        ...

    async def get(self, key: str) -> object:
        """Fetch opaque bytes; expiration is controlled by Redis."""
        ...

    async def set(self, key: str, value: bytes, *, ex: int) -> object:
        """Store only ephemeral values with a mandatory expiry."""
        ...

    async def aclose(self) -> None:
        """Close the async pool rather than relying on garbage collection."""
        ...


class RedisState:
    """Separate availability policy: quota fails closed; optional cache failures become misses."""

    def __init__(
        self,
        client: RedisClient,
        secret: bytes,
        *,
        limit: int = 120,
        window_ms: int = 60_000,
        timeout_seconds: float = 0.25,
    ) -> None:
        """One configured service credential is one quota subject; no multi-tenant auth implied."""
        if len(secret) < 16 or not 1 <= limit <= 1_000_000 or not 100 <= window_ms <= 86_400_000:
            raise ValueError("invalid Redis quota configuration")
        if not 0 < timeout_seconds <= 5:
            raise ValueError("invalid Redis deadline")
        self.client, self.secret = client, secret
        self.limit, self.window_ms, self.timeout_seconds = limit, window_ms, timeout_seconds

    def key(self, scope: str, identity: bytes) -> str:
        """HMAC avoids putting prompts, credentials or guessable content digests in key names."""
        digest = hmac.new(
            self.secret, scope.encode() + b"\0" + identity, hashlib.sha256
        ).hexdigest()
        return f"finserve:{scope}:{digest}"

    async def allow(self, subject: str) -> QuotaDecision:
        """Both network and total deadlines bound the Redis hop before inference begins."""
        async with asyncio.timeout(self.timeout_seconds):
            result = await self.client.eval(
                QUOTA_SCRIPT,
                1,
                self.key("quota", f"{self.limit}:{self.window_ms}:{subject}".encode()),
                self.limit,
                self.window_ms,
            )
        values = TypeAdapter(tuple[int, int]).validate_python(result, strict=False)
        if values[0] not in {0, 1} or values[1] < 0:
            raise ValueError("invalid Redis quota response")
        return QuotaDecision(allowed=values[0] == 1, retry_after_ms=values[1])

    async def cached(self, identity: bytes) -> bytes | None:
        """Only bounded bytes are a cache hit; callers retain authoritative durable evidence."""
        try:
            async with asyncio.timeout(self.timeout_seconds):
                value = await self.client.get(self.key("response", identity))
            return value if isinstance(value, bytes) and len(value) <= 1_048_576 else None
        except Exception:
            return None

    async def remember(self, identity: bytes, value: bytes, ttl_seconds: int = 300) -> bool:
        """Cache write failure cannot fail a completed request or create a permanent record."""
        if len(value) > 1_048_576 or not 1 <= ttl_seconds <= 3600:
            raise ValueError("invalid cache value or TTL")
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return bool(
                    await self.client.set(self.key("response", identity), value, ex=ttl_seconds)
                )
        except Exception:
            return False

    async def close(self) -> None:
        """Return network resources at shutdown without an unbounded external await."""
        async with asyncio.timeout(5):
            await self.client.aclose()


def from_url(url: str, secret: bytes, *, limit: int = 120, window_ms: int = 60_000) -> RedisState:
    """Production deployments provide TLS and credentials through a secret-bearing Redis URL."""
    if urlsplit(url).scheme not in {"redis", "rediss"}:
        raise ValueError("Redis URL must use redis or rediss")
    module = importlib.import_module("redis.asyncio")
    client = cast(
        RedisClient,
        module.from_url(
            url,
            socket_timeout=0.25,
            socket_connect_timeout=0.25,
            max_connections=32,
            decode_responses=False,
            retry_on_timeout=False,
        ),
    )
    return RedisState(client, secret, limit=limit, window_ms=window_ms)
