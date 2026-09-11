"""Strict runtime readiness records content, terminal framing and authoritative token accounting."""

import asyncio
import time

import httpx

from finserve.benchmark.metrics import RequestRecord
from finserve.engines.openai_adapter import CompletionState, EngineProtocolError, sse_events
from finserve.http_ownership import HTTPClosureError, own_response


async def completion_probe(
    client: httpx.AsyncClient, base_url: str, model: str, budget_seconds: float
) -> RequestRecord:
    """Retain each offered smoke response while enforcing the production completion protocol."""
    sent = time.perf_counter()
    first: float | None = None
    output = ""
    status: int | None = None
    error: str | None = None
    done = False
    state = CompletionState(maximum_tokens=1)
    try:
        async with asyncio.timeout(budget_seconds):
            async with client.stream(
                "POST",
                base_url + "/completions",
                follow_redirects=False,
                headers={"accept": "text/event-stream", "accept-encoding": "identity"},
                json={
                    "model": model,
                    "prompt": "Hello",
                    "temperature": 0,
                    "max_tokens": 1,
                    "stream": True,
                    "n": 1,
                    "stream_options": {"include_usage": True},
                },
                timeout=budget_seconds,
            ) as response:
                own_response(response)
                status = response.status_code
                response.raise_for_status()
                if (
                    status != 200
                    or response.headers.get("content-encoding", "identity").lower() != "identity"
                ):
                    raise EngineProtocolError("runtime returned an unsupported status or encoding")
                if (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                    != "text/event-stream"
                ):
                    raise EngineProtocolError("runtime did not return an SSE response")
                async for event in sse_events(response, 65536):
                    if event == "[DONE]":
                        state.final_token()
                        done = True
                        break
                    text = state.consume(event)
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        output += text
                if not done:
                    raise EngineProtocolError("runtime smoke omitted terminal framing")
    except (
        httpx.HTTPError,
        TimeoutError,
        ValueError,
        EngineProtocolError,
        HTTPClosureError,
    ) as exc:
        error = type(exc).__name__
    except asyncio.CancelledError:
        error = "CancelledError"
    return RequestRecord(
        logical_id=0,
        case_id="runtime-smoke",
        family="RUNTIME_SMOKE",
        phase="runtime-smoke",
        scheduled_s=sent,
        send_s=sent,
        first_content_s=first,
        complete_s=time.perf_counter(),
        success=done and error is None,
        status_code=status,
        error=error,
        output=output,
        generated_tokens=state.completion_tokens,
    )
