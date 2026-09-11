"""Bounded OpenAI completion streaming shared by independently deployed GPU engines."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.contracts.output_constraint import MAXIMUM_CONSTRAINED_OUTPUT_BYTES

MAX_EVENT_CHARACTERS = 65536


class EngineProtocolError(RuntimeError):
    """Expose only bounded static errors, never backend bodies, prompts, or credentials."""


class EngineUnavailableError(RuntimeError):
    """Distinguish upstream availability failures from malformed successful streams."""


class CompletionChoice(BaseModel):
    """Accept backend extensions but enforce one text completion and explicit terminal reasons."""

    model_config = ConfigDict(extra="ignore", strict=True)
    index: Literal[0]
    text: str
    finish_reason: Literal["stop", "length"] | None = None


class CompletionUsage(BaseModel):
    """Token totals must be integers supplied by the engine, never inferred from SSE frames."""

    model_config = ConfigDict(extra="ignore", strict=True)
    completion_tokens: int = Field(ge=0)


class CompletionFrame(BaseModel):
    """Validate the minimal shared vLLM/SGLang shape without binding to extension fields."""

    model_config = ConfigDict(extra="ignore", strict=True)
    choices: list[CompletionChoice] = Field(max_length=1)
    usage: CompletionUsage | None = None


class BoundedByteStream(httpx.AsyncByteStream):
    """Cap bytes before HTTPX's line decoder can buffer an unterminated SSE line."""

    def __init__(self, source: AsyncIterator[bytes], maximum: int) -> None:
        """Wrap decoded response bytes so compressed responses cannot bypass the budget."""
        self._source = source
        self._maximum = maximum

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Reject the first oversized chunk before passing it into the line accumulator."""
        received = 0
        async for chunk in self._source:
            received += len(chunk)
            if received > self._maximum:
                raise EngineProtocolError("Engine stream exceeded the response byte limit")
            yield chunk


async def sse_events(response: httpx.Response, maximum_bytes: int) -> AsyncIterator[str]:
    """Decode arbitrary network chunks and standard multi-line SSE data fields.

    The synthetic response reuses HTTPX's Unicode/CRLF line decoder while enforcing
    a byte budget before buffering. The original response context owns socket closure.
    """
    bounded = httpx.Response(
        200,
        stream=BoundedByteStream(response.aiter_bytes(), maximum_bytes),
        headers={"content-type": "text/event-stream; charset=utf-8"},
    )
    data: list[str] = []
    size = 0
    async for line in bounded.aiter_lines():
        if not line:
            if data:
                yield "\n".join(data)
            data, size = [], 0
        elif line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            size += len(value) + 1
            if size > MAX_EVENT_CHARACTERS:
                raise EngineProtocolError("Engine SSE event exceeded the event limit")
            data.append(value)
        # Comments and optional event/id/retry fields do not contain completion data.
    if data:
        raise EngineProtocolError("Engine stream ended inside an SSE event")


@dataclass
class CompletionState:
    """Track terminal accounting independently of visible text frame boundaries."""

    maximum_tokens: int
    finish_reason: Literal["stop", "length"] | None = None
    completion_tokens: int | None = None
    visible_text: bool = False

    def consume(self, data: str) -> str:
        """Reject malformed, duplicate, or out-of-order frames before trusting their counts."""
        try:
            frame = CompletionFrame.model_validate_json(data)
        except ValidationError:
            raise EngineProtocolError("Engine returned an invalid completion event") from None
        if self.completion_tokens is not None:
            raise EngineProtocolError("Engine sent data after final usage")
        text = ""
        if frame.choices:
            if self.finish_reason is not None:
                raise EngineProtocolError("Engine sent a choice after completion finished")
            choice = frame.choices[0]
            text = choice.text
            self.visible_text = self.visible_text or bool(text)
            self.finish_reason = choice.finish_reason
        if frame.usage is not None:
            if self.finish_reason is None:
                raise EngineProtocolError("Engine sent final usage before completion finished")
            count = frame.usage.completion_tokens
            if count > self.maximum_tokens or (self.visible_text and count == 0):
                raise EngineProtocolError("Engine reported an inconsistent completion token count")
            self.completion_tokens = count
        return text

    def final_token(self) -> EngineToken:
        """Publish accounting only after DONE confirms a fully received completion stream."""
        if self.finish_reason is None or self.completion_tokens is None:
            raise EngineProtocolError("Engine completion omitted final usage or finish reason")
        return EngineToken(
            text="", generated_tokens=self.completion_tokens, finish_reason=self.finish_reason
        )


class OpenAICompletionEngine:
    """Own a reusable HTTP pool; backend processes own batching, GPU scheduling, and KV memory."""

    supports_output_constraints = False

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        maximum_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        """Accept endpoint configuration only at construction, disabling redirects and env proxies.

        A transport injection permits socket-free contract tests. The adapter owns its
        client and authentication header; it never accepts destination URLs in requests.
        """
        url = httpx.URL(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Engine base URL must be HTTP(S) without credentials, query, or fragment"
            )
        if maximum_response_bytes < 1:
            raise ValueError("Engine response byte limit must be positive")
        self._url = str(url).rstrip("/") + "/completions"
        self._chat_url = str(url).rstrip("/") + "/chat/completions"
        self._maximum_response_bytes = maximum_response_bytes
        headers = {"accept": "text/event-stream", "accept-encoding": "identity"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            headers=headers,
            transport=transport,
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def stream(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """Preserve raw completion prompts or send original chat roles through the backend template.

        Visible deltas carry zero counts; the final empty event carries the authoritative
        total. Deadlines cover pool wait, headers, and every frame. Generator closure or
        cancellation exits the HTTP stream context and releases its pooled connection.
        """
        payload: dict[str, object] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            "n": 1,
            "stream_options": {"include_usage": True},
        }
        if request.output_constraint is not None:
            if not self.supports_output_constraints:
                raise ValueError("Engine does not support output constraints")
            payload["structured_outputs"] = request.output_constraint.vllm_parameters()
        state: CompletionState
        constrained_text: list[str] = []
        constrained_bytes = 0
        if request.messages is None:
            payload["prompt"] = request.prompt
            url, state = self._url, CompletionState(maximum_tokens=request.max_tokens)
        else:
            # The shared parser depends only on completion accounting, never vision/Pillow/JAX.
            from finserve.engines.chat_protocol import ChatState

            payload["messages"] = [message.model_dump() for message in request.messages]
            url, state = self._chat_url, ChatState(maximum_tokens=request.max_tokens)
        deadline = asyncio.get_running_loop().time() + request.timeout_seconds
        from finserve.telemetry.propagation import trace_headers

        outbound = self._client.build_request(
            "POST", url, json=payload, timeout=request.timeout_seconds, headers=trace_headers()
        )
        try:
            async with asyncio.timeout_at(deadline):
                response = await self._client.send(outbound, stream=True)
            try:
                self._validate_response(response)
                events = sse_events(response, self._maximum_response_bytes)
                while True:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("Engine request deadline expired")
                    try:
                        # No timeout context spans a yield: it must not cancel consumer work.
                        async with asyncio.timeout_at(deadline):
                            event = await anext(events)
                    except StopAsyncIteration:
                        raise EngineProtocolError("Engine stream ended without DONE") from None
                    if event == "[DONE]":
                        if (
                            request.output_constraint is not None
                            and not request.output_constraint.accepts("".join(constrained_text))
                        ):
                            raise EngineProtocolError(
                                "Engine output did not satisfy requested shape"
                            )
                        yield state.final_token()
                        return
                    text = state.consume(event)
                    if text:
                        if request.output_constraint is not None:
                            constrained_bytes += len(text.encode("utf-8"))
                            if constrained_bytes > MAXIMUM_CONSTRAINED_OUTPUT_BYTES:
                                raise EngineProtocolError("Constrained output exceeded byte limit")
                            constrained_text.append(text)
                        yield EngineToken(text=text, generated_tokens=0)
            finally:
                await response.aclose()
        except httpx.TimeoutException:
            raise TimeoutError("Engine HTTP operation timed out") from None
        except httpx.HTTPError:
            raise EngineUnavailableError("Engine HTTP transport failed") from None

    @staticmethod
    def _validate_response(response: httpx.Response) -> None:
        """Fail without echoing upstream error bodies, which can contain prompts or secrets."""
        if response.status_code != 200:
            raise EngineUnavailableError(f"Engine returned HTTP status {response.status_code}")
        media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "text/event-stream":
            raise EngineProtocolError("Engine response is not an SSE stream")
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise EngineProtocolError("Engine SSE compression is unsupported")

    async def close(self) -> None:
        """Close owned sockets at application shutdown; callers first cancel active requests."""
        await self._client.aclose()
