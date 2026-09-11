"""Exercise stream, errors, cancellation cleanup and overload through ASGI."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.engines.fixture import FixtureEngine
from finserve.gateway.app import create_app


async def test_stream_and_nonstream() -> None:
    """Streaming and aggregated replies share exact engine token accounting."""
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post("/v1/completions", json={"prompt": "hello", "max_tokens": 3})
        assert "data: [DONE]" in response.text
        assert '"completion_tokens":3' in response.text
        assert '"server_ttft_seconds":' in response.text
        result = await c.post("/v1/completions", json={"prompt": "hello", "stream": False})
        assert result.json()["choices"][0]["text"] == "hello "
        chat = await c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert chat.json()["choices"][0]["message"]["content"] == "user: hi "
        assert app.state.serving.admission.active == 0
        metrics = await c.get("/metrics")
        assert 'finserve_requests_total{outcome="success"} 3.0' in metrics.text


class SlowEngine(FixtureEngine):
    """A suspended engine makes timeout and disconnect behavior deterministic."""

    async def stream(self, request: InferenceRequest) -> AsyncIterator[EngineToken]:
        """Suspend before generation to reproduce pre-first-token timeout."""
        await asyncio.sleep(10)
        yield EngineToken(text="late")


async def test_timeout_and_overload() -> None:
    """Timeout releases the lease and rejected work does not acquire an extra slot."""
    app = create_app(SlowEngine(), max_concurrency=1)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post("/v1/completions", json={"prompt": "hi", "timeout_seconds": 0.01})
        assert "TIMEOUT" in response.text
        assert "[DONE]" not in response.text
        assert app.state.serving.admission.active == 0
        assert app.state.serving.admission.acquire()
        response = await c.post("/v1/completions", json={"prompt": "hi"})
        assert response.status_code == 429
        app.state.serving.admission.release()


async def test_auth_validation_and_body_limit() -> None:
    """Validate auth and both declared and actual body size without trusting clients."""
    app = create_app(api_key="test-secret")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.post("/v1/completions", json={"prompt": "hi"})).status_code == 401
        c.headers["Authorization"] = "Bearer test-secret"
        assert (await c.post("/v1/completions", json={"prompt": ""})).status_code == 422
        assert (
            await c.post("/v1/completions", json={"prompt": "hi", "model": "bad"})
        ).status_code == 404
        response = await c.post("/v1/completions", content=b"x" * 131073)
        assert response.status_code == 413


async def test_cancel_releases_admission() -> None:
    """Cancellation before any token must close engine work and return capacity."""
    app = create_app(SlowEngine())
    serving = app.state.serving
    assert serving.admission.acquire()
    serving.metrics.active.inc()
    stream = serving.events(InferenceRequest(prompt="hi"), 0.0, False)
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert serving.admission.active == 0


class FailedEngine(FixtureEngine):
    """A synchronous iterator-construction failure exercises pre-iteration cleanup."""

    def stream(self, request: InferenceRequest) -> AsyncIterator[EngineToken]:
        """Backend setup failures must use the same terminal lease cleanup."""
        raise OSError("private backend failure")


@pytest.mark.parametrize("streaming", [True, False])
async def test_engine_failure_cleanup(streaming: bool) -> None:
    """Both response modes release immediately and redact internal error details."""
    app = create_app(FailedEngine())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        result = await client.post("/v1/completions", json={"prompt": "hi", "stream": streaming})
        assert "ENGINE_FAILED" in result.text
        assert "private backend" not in result.text
        assert app.state.serving.admission.active == 0
