"""The frozen benchmark chat wire format works through real HTTP gateway and warm routes."""

import asyncio
import json
import socket
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from finserve.benchmark.runner import RunConfig, request_one
from finserve.benchmark.workload import WorkItem
from finserve.contracts.deployment import Revision
from finserve.contracts.inference import ChatStreamOptions
from finserve.engines.openai_adapter import OpenAICompletionEngine
from finserve.gateway.app import create_app
from finserve.gateway.warm_route_app import create_warm_app
from finserve.reliability.warm_routes import BackendConfiguration, WarmBackend, WarmRouteStore


@asynccontextmanager
async def live(app: FastAPI) -> AsyncGenerator[str]:
    """Own an ephemeral listener and stop only this test's Uvicorn task."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.setblocking(False)
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("fixture server failed")
                    await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 10)


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_usage_option_rejects_coercion(value: object) -> None:
    """Usage suppression and coercible values cannot silently alter this API's usage contract."""
    with pytest.raises(ValueError, match="include_usage"):
        ChatStreamOptions.model_validate({"include_usage": value})
    with pytest.raises(ValueError):
        ChatStreamOptions.model_validate({"include_usage": True, "unknown": True})
    assert ChatStreamOptions(include_usage=True).include_usage is True


@pytest.mark.parametrize("warm", [False, True])
async def test_exact_chat_mapping_over_gateway_and_backend_http(tmp_path: Path, warm: bool) -> None:
    """Role messages and include_usage survive real sockets on both sides of either gateway."""
    seen: list[dict[str, object]] = []
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat(request: Request) -> StreamingResponse:
        """The independent backend fixture emits role, content, terminal and usage events."""
        seen.append(await request.json())
        events: list[dict[str, object]] = [
            {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {"choices": [{"index": 0, "delta": {"content": "42"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 2}},
        ]

        async def stream() -> AsyncGenerator[str]:
            """Use genuine SSE boundaries and an authoritative tokenizer count."""
            for event in events:
                yield "data: " + json.dumps(event) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    async with live(backend) as backend_url:
        configuration = BackendConfiguration(base_url=backend_url + "/v1", model="fixture")
        if warm:
            store = WarmRouteStore(tmp_path / "routes.sqlite")
            revision = Revision(
                revision_id="known-good",
                model_revision="fixture",
                tokenizer_revision="fixture",
                source_revision="fixture",
                image_digest="sha256:" + "a" * 64,
                config_digest=configuration.digest(),
                engine="fixture",
                engine_config="fixture",
            )
            store.register(WarmBackend(configuration=configuration, revision=revision))
            store.bootstrap("deployment", revision.revision_id)
            gateway = create_warm_app(store, "deployment")
        else:
            gateway = create_app(OpenAICompletionEngine(configuration.base_url), model="fixture")
        async with live(gateway) as gateway_url, httpx.AsyncClient() as client:
            config = RunConfig(
                model="fixture",
                request_api="chat",
                system_prompt="Exact format.",
                chat_template_sha256="a" * 64,
            )
            row = await request_one(
                client,
                gateway_url + "/v1/chat/completions",
                WorkItem(case_id="heldout", prompt="Return only 6 times 7.", max_tokens=8),
                0,
                time.perf_counter(),
                config,
                "measured",
            )
            assert row.success and row.output == "42" and row.generated_tokens == 2
            assert row.first_content_s is not None
        assert seen == [
            {
                "model": "fixture",
                "messages": [
                    {"role": "system", "content": "Exact format."},
                    {"role": "user", "content": "Return only 6 times 7."},
                ],
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": True,
                "n": 1,
                "stream_options": {"include_usage": True},
            }
        ]
