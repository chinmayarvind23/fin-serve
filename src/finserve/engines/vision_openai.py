"""Real multimodal Chat Completions transport with authoritative terminal accounting."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finserve.contracts.inference import EngineToken
from finserve.contracts.vision import VisionRequest
from finserve.engines.openai_adapter import (
    CompletionState,
    CompletionUsage,
    EngineProtocolError,
    EngineUnavailableError,
    sse_events,
)
from finserve.multimodal.images import PreparedImage


class VisionEngine(Protocol):
    """Image eligibility is explicit; a raw text engine cannot silently accept this contract."""

    def stream(
        self, request: VisionRequest, image: PreparedImage
    ) -> AsyncGenerator[EngineToken, None]:
        """Stream one already validated image under the caller's remaining deadline."""
        ...

    async def close(self) -> None:
        """Release owned transport after active request owners have drained."""
        ...


class ChatDelta(BaseModel):
    """Accept normal assistant role/content deltas; tool and reasoning protocols are separate."""

    model_config = ConfigDict(extra="ignore", strict=True)
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: None = None
    function_call: None = None
    refusal: None = None


class ChatChoice(BaseModel):
    """Only one generation is budgeted, and every finish reason must have defined semantics."""

    model_config = ConfigDict(extra="ignore", strict=True)
    index: Literal[0]
    delta: ChatDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatFrame(BaseModel):
    """The final usage-only frame is distinct from assistant text deltas."""

    model_config = ConfigDict(extra="ignore", strict=True)
    choices: list[ChatChoice] = Field(max_length=1)
    usage: CompletionUsage | None = None


@dataclass
class ChatState(CompletionState):
    """Reuse accounting invariants while parsing the actual chat delta wire shape."""

    def consume(self, data: str) -> str:
        """Translate validated chat fields to the shared count state without guessing tokens."""
        try:
            frame = ChatFrame.model_validate(json.loads(data, object_pairs_hook=unique_object))
        except (ValidationError, ValueError):
            raise EngineProtocolError("Engine returned an invalid chat event") from None
        choices = [
            {"index": 0, "text": choice.delta.content or "", "finish_reason": choice.finish_reason}
            for choice in frame.choices
        ]
        # Shared state enforces finish-before-usage, total bounds and no data after usage.
        from finserve.engines.openai_adapter import CompletionFrame

        translated = CompletionFrame.model_validate({"choices": choices, "usage": frame.usage})
        return super().consume(translated.model_dump_json())


class OpenAIVisionEngine:
    """Own only HTTP resources; pinned vLLM owns image processing, vision encoder and decoding."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        maximum_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        """Trust only startup endpoint configuration, never media/request supplied destinations."""
        url = httpx.URL(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
            or type(maximum_response_bytes) is not int
            or not 1 <= maximum_response_bytes <= 4 * 1024 * 1024
        ):
            raise ValueError("Invalid vision engine endpoint or response budget")
        self._url = str(url).rstrip("/") + "/chat/completions"
        headers = {"accept": "text/event-stream", "accept-encoding": "identity"}
        if api_key:
            headers["authorization"] = "Bearer " + api_key
        self._client = httpx.AsyncClient(
            transport=transport,
            headers=headers,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=2),
        )
        self._maximum_response_bytes = maximum_response_bytes

    async def stream(
        self, request: VisionRequest, image: PreparedImage
    ) -> AsyncGenerator[EngineToken, None]:
        """Send inline PNG plus text through the model's real chat template, with no retry."""
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image.data_url()}},
                        {"type": "text", "text": request.prompt},
                    ],
                }
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            "n": 1,
            "stream_options": {"include_usage": True},
        }
        deadline = asyncio.get_running_loop().time() + request.timeout_seconds
        outbound = self._client.build_request(
            "POST", self._url, json=payload, timeout=request.timeout_seconds
        )
        try:
            async with asyncio.timeout_at(deadline):
                response = await self._client.send(outbound, stream=True)
            try:
                validate_chat_response(response)
                state = ChatState(maximum_tokens=request.max_tokens)
                events = sse_events(response, self._maximum_response_bytes)
                while True:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("Vision deadline expired")
                    try:
                        async with asyncio.timeout_at(deadline):
                            event = await anext(events)
                    except StopAsyncIteration:
                        raise EngineProtocolError("Chat stream ended without DONE") from None
                    if event == "[DONE]":
                        yield state.final_token()
                        return
                    text = state.consume(event)
                    if text:
                        yield EngineToken(text=text, generated_tokens=0)
            finally:
                await response.aclose()
        except httpx.TimeoutException:
            raise TimeoutError("Vision HTTP operation timed out") from None
        except httpx.HTTPError:
            raise EngineUnavailableError("Vision HTTP transport failed") from None

    async def close(self) -> None:
        """Close the reusable HTTP pool after application request owners are cancelled/drained."""
        await self._client.aclose()


def validate_chat_response(response: httpx.Response) -> None:
    """Reject backend error bodies and compressed SSE without reading sensitive payloads."""
    if response.status_code != 200:
        raise EngineUnavailableError(f"Vision engine returned HTTP status {response.status_code}")
    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "text/event-stream":
        raise EngineProtocolError("Vision engine response is not SSE")
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise EngineProtocolError("Vision SSE compression is unsupported")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Ambiguous duplicate JSON fields cannot decide final usage or change media semantics."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
