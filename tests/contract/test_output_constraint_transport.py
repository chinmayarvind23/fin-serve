"""Exercise constraint dispatch and terminal failure through actual HTTPX/ASGI boundaries."""

import json

import httpx
import pytest

from finserve.contracts.inference import ChatMessage, ChatRequest, EngineToken, InferenceRequest
from finserve.contracts.output_constraint import ObjectField, OutputConstraint
from finserve.engines.fixture import FixtureEngine
from finserve.engines.openai_adapter import EngineProtocolError, OpenAICompletionEngine
from finserve.engines.ray_http import RayHTTPEngine
from finserve.engines.ray_serve import FixtureReplica
from finserve.engines.sglang_adapter import SGLangEngine
from finserve.engines.vllm_adapter import VLLMEngine
from finserve.gateway.app import create_app


def mock_engine(text: str, *, finish: str = "stop") -> tuple[VLLMEngine, list[dict[str, object]]]:
    """Return retained raw fragments and declared usage while recording upstream wire fields."""
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Serve either native chat or completions so both public request mappings are exercised."""
        payload = json.loads(request.content)
        requests.append(payload)
        content = {"delta": {"content": text}} if "messages" in payload else {"text": text}
        terminal: dict[str, object] = {"delta": {}} if "messages" in payload else {"text": ""}
        frames: list[dict[str, object]] = [
            {"choices": [{"index": 0, **content, "finish_reason": None}]},
            {"choices": [{"index": 0, **terminal, "finish_reason": finish}]},
            {"choices": [], "usage": {"completion_tokens": 3}},
        ]
        wire = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=wire + "data: [DONE]\n\n"
        )

    return VLLMEngine("http://engine.test/v1", transport=httpx.MockTransport(respond)), requests


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_gateway_forwards_shape_and_keeps_raw_content(chat: bool) -> None:
    """Formatting constraints travel through real ASGI chat conversion and the native adapter."""
    raw = '{\n "amount": -2.5\n}'
    constraint = OutputConstraint(
        kind="json_object", fields=(ObjectField(name="amount", type="number"),)
    )
    engine, outbound = mock_engine(raw)
    app = create_app(engine=engine)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway.test"
        ) as client:
            payload = (
                {"messages": [{"role": "user", "content": "Return amount."}]}
                if chat
                else {"prompt": "Return amount."}
            )
            response = await client.post(
                "/v1/chat/completions" if chat else "/v1/completions",
                json={**payload, "output_constraint": constraint.model_dump(), "stream": False},
            )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert (choice["message"]["content"] if chat else choice["text"]) == raw
    assert outbound[0]["structured_outputs"] == constraint.vllm_parameters()
    assert "output_constraint" not in outbound[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(("raw", "finish"), [('{"amount":', "length"), ("0.20", "stop")])
async def test_partial_or_invalid_output_never_publishes_success(raw: str, finish: str) -> None:
    """Preserve fragments and fail instead of repairing or accepting truncated output."""
    constraint = (
        OutputConstraint(kind="decimal")
        if finish == "stop"
        else OutputConstraint(
            kind="json_object", fields=(ObjectField(name="amount", type="number"),)
        )
    )
    engine, _ = mock_engine(raw, finish=finish)
    seen: list[EngineToken] = []
    try:
        with pytest.raises(EngineProtocolError, match="shape"):
            async for token in engine.stream(
                InferenceRequest(prompt="Answer", output_constraint=constraint)
            ):
                seen.append(token)
        assert "".join(token.text for token in seen) == raw
        assert all(token.finish_reason is None for token in seen)
    finally:
        await engine.close()


@pytest.mark.asyncio
async def test_unsupported_engines_and_ray_fixture_do_not_generate() -> None:
    """Unknown native capabilities and reference implementations refuse before yielding text."""
    request = InferenceRequest(prompt="hello", output_constraint=OutputConstraint(kind="yes_no"))
    for engine in (
        OpenAICompletionEngine("http://unused.test/v1"),
        SGLangEngine("http://unused.test/v1"),
        FixtureEngine(),
    ):
        try:
            with pytest.raises(ValueError, match="support"):
                await anext(engine.stream(request))
        finally:
            await engine.close()
    replica = FixtureReplica("fixture", 1, 0)
    with pytest.raises(ValueError, match="support"):
        await anext(replica.tokens(request))


@pytest.mark.asyncio
async def test_gateway_rejects_unsupported_shape_before_success_stream() -> None:
    """Fixture deployments report a bounded 422 instead of accepting an unimplemented feature."""
    app = create_app()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://gateway.test"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=ChatRequest(
                    messages=[ChatMessage(role="user", content="Answer yes or no")],
                    output_constraint=OutputConstraint(kind="yes_no"),
                ).model_dump(),
            )
    assert response.status_code == 422
    assert "UNSUPPORTED_OUTPUT_CONSTRAINT" in response.text


@pytest.mark.asyncio
async def test_ray_relay_preserves_the_typed_constraint() -> None:
    """The internal NDJSON hop transports the same caller contract rather than native API fields."""
    observed: list[InferenceRequest] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Validate the actual outbound body and return two well-formed engine envelopes."""
        observed.append(InferenceRequest.model_validate_json(request.content))
        frames: list[dict[str, object]] = [
            {"replica_id": "vllm", "token": {"text": "yes", "generated_tokens": 0}},
            {
                "replica_id": "vllm",
                "token": {
                    "text": "",
                    "generated_tokens": 1,
                    "finish_reason": "stop",
                },
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "application/x-ndjson"},
            content="".join(json.dumps(frame) + "\n" for frame in frames),
        )

    engine = RayHTTPEngine("http://ray.test/generate", transport=httpx.MockTransport(respond))
    request = InferenceRequest(
        prompt="Answer yes or no", output_constraint=OutputConstraint(kind="yes_no")
    )
    try:
        tokens = [token async for token in engine.stream(request)]
        assert observed == [request]
        assert "".join(token.text for token in tokens) == "yes"
    finally:
        await engine.close()
