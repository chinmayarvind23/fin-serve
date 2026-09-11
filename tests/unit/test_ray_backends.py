"""Trusted endpoint and fail-closed health contracts without importing the optional Ray runtime."""

import asyncio
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from finserve.engines.ray_backends import BackendConfiguration, OpenAIEngineReplica


@pytest.mark.parametrize(
    "second",
    [
        "http://EXAMPLE.com/v1/",
        "http://example.com:80/v1",
        "http://example.com/x/../v1",
    ],
)
def test_duplicate_canonical_urls_rejected(second: str) -> None:
    """Alias spellings of one configured endpoint cannot manufacture extra routing capacity."""
    with pytest.raises(ValidationError, match="distinct"):
        BackendConfiguration(
            model="exact-model", backends={"a": "http://example.com/v1", "b": second}
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://example.com/v1",
        "http://user:password@example.com/v1",
        "http://example.com/v1?q=1",
        "http://example.com/v1#ignored",
        "http://example.com/v%31",
        "http://example.com/v1\\other",
    ],
)
def test_invalid_or_ambiguous_endpoint_rejected(endpoint: str) -> None:
    """Reject credentials, unused URL parts and ambiguous paths in trusted configuration."""
    with pytest.raises(ValidationError):
        BackendConfiguration(model="exact-model", backends={"a": endpoint})


@pytest.mark.parametrize(
    "document,code,healthy",
    [
        ({"data": [{"id": "exact-model"}]}, 200, True),
        ({"data": [{"id": "exact-model-other"}]}, 200, False),
        ({"data": []}, 200, False),
        ({"data": [1]}, 200, False),
        ({"unrelated": "value"}, 200, False),
        ({"data": [{"id": "exact-model"}]}, 503, False),
        ({"data": [{"id": "exact-model"}]}, 302, False),
    ],
)
async def test_exact_model_discovery_and_bad_health_fail_closed(
    document: object,
    code: int,
    healthy: bool,
) -> None:
    """A responding HTTP port is insufficient: routing requires its exact configured model ID."""
    worker = OpenAIEngineReplica("a", 2, "exact-model", "http://example.com/v1")
    await worker.probe.aclose()

    def respond(request: httpx.Request) -> httpx.Response:
        """Return controlled discovery while still using HTTPX response/status parsing."""
        assert request.url.path == "/v1/models"
        return httpx.Response(code, json=document)

    worker.probe = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    try:
        snapshot = await worker.state()
        assert snapshot["healthy"] is healthy
        assert snapshot["model"] == "exact-model" and snapshot["capacity"] == 2
    finally:
        await worker.close()


@pytest.mark.parametrize("failure", ["unreachable", "oversized", "invalid_json"])
async def test_discovery_transport_and_payload_failures(failure: str) -> None:
    """Unavailable or oversized discovery must not leave a replica eligible for new requests."""
    worker = OpenAIEngineReplica("a", 1, "exact-model", "http://example.com/v1")
    await worker.probe.aclose()

    def respond(request: httpx.Request) -> httpx.Response:
        """Inject independent transport/byte/syntax failures before any engine generation."""
        if failure == "unreachable":
            raise httpx.ConnectError("private address", request=request)
        return httpx.Response(200, content=b"x" * (131073 if failure == "oversized" else 1))

    worker.probe = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    try:
        assert (await worker.state())["healthy"] is False
    finally:
        await worker.close()


async def test_health_singleflight_cache_and_initial_probe() -> None:
    """Probe near clock origin, share concurrent discovery and expire the cached result."""
    worker = OpenAIEngineReplica("a", 1, "exact-model", "http://example.com/v1")
    await worker.probe.aclose()
    calls = 0

    async def respond(_: httpx.Request) -> httpx.Response:
        """Yield once to exercise the health lock, not just sequential cache reuse."""
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return httpx.Response(200, json={"data": [{"id": "exact-model"}]})

    worker.probe = httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False)
    try:
        with patch("finserve.engines.ray_backends.time.monotonic", return_value=0.1):
            snapshots = await asyncio.gather(*(worker.state() for _ in range(4)))
        assert all(snapshot["healthy"] for snapshot in snapshots) and calls == 1
        with patch("finserve.engines.ray_backends.time.monotonic", return_value=1.2):
            assert (await worker.state())["healthy"] is True
        assert calls == 2
    finally:
        await worker.close()


async def test_closed_worker_is_ineligible_and_close_is_idempotent() -> None:
    """A cached healthy result cannot admit more proxy work once shutdown has begun."""
    worker = OpenAIEngineReplica("a", 1, "exact-model", "http://example.com/v1")
    worker.admit("reserved")
    await asyncio.gather(worker.close(), worker.close())
    assert worker.probe.is_closed and not worker.active
    assert (await worker.state())["healthy"] is False


def test_valid_distinct_endpoints_and_invalid_names() -> None:
    """Distinct configured processes are valid while malformed deployment identities fail early."""
    valid = BackendConfiguration(
        model="exact-model",
        backends={"a": "http://example.com:8001/v1", "b": "http://example.com:8002/v1"},
    )
    assert len(valid.backends) == 2
    for name in ("", "a/b", "x" * 64):
        with pytest.raises(ValidationError):
            BackendConfiguration(model="exact-model", backends={name: "http://example.com/v1"})


async def test_trusted_engine_key_is_attached_to_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery uses the configured upstream credential rather than trusting public port access."""
    monkeypatch.setenv("FINSERVE_ENGINE_API_KEY", "fixture-engine-key")
    worker = OpenAIEngineReplica("a", 1, "exact-model", "http://example.com/v1")
    try:
        assert worker.probe.headers["authorization"] == "Bearer fixture-engine-key"
    finally:
        await worker.close()
