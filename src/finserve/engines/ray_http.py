"""Bound the internal Ray NDJSON boundary without importing Ray into the public gateway."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finserve.contracts.inference import EngineToken, InferenceRequest
from finserve.engines.openai_adapter import (
    BoundedByteStream,
    EngineProtocolError,
    EngineUnavailableError,
)


class RayEnvelope(BaseModel):
    """A request stays on one replica and preserves engine-supplied terminal accounting."""

    model_config = ConfigDict(extra="forbid", strict=True)
    replica_id: str = Field(min_length=1, max_length=128)
    token: EngineToken


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Duplicate JSON fields cannot silently replace replica identity or accounting values."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


async def envelopes(response: httpx.Response, maximum: int) -> AsyncIterator[RayEnvelope]:
    """Require valid UTF-8 and complete newline framing before accepting an envelope."""
    pending = bytearray()
    async for chunk in BoundedByteStream(response.aiter_bytes(), maximum):
        pending.extend(chunk)
        while (end := pending.find(b"\n")) >= 0:
            raw = bytes(pending[:end])
            del pending[: end + 1]
            try:
                value = json.loads(
                    raw.decode("utf-8", errors="strict"), object_pairs_hook=unique_object
                )
                envelope = RayEnvelope.model_validate(value, strict=True)
            except (ValueError, ValidationError):
                raise EngineProtocolError("Invalid Ray token envelope") from None
            yield envelope
    if pending:
        raise EngineProtocolError("Ray stream ended inside an envelope")


class RayHTTPEngine:
    """Own a bounded HTTP pool to an operator-configured private routing application."""

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        maximum_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        """Forbid URL credentials and redirects; callers cannot select another destination."""
        destination = httpx.URL(url)
        if (
            destination.scheme not in {"http", "https"}
            or not destination.host
            or destination.username
            or destination.password
            or destination.query
            or destination.fragment
            or type(maximum_response_bytes) is not int
            or not 1 <= maximum_response_bytes <= 4 * 1024 * 1024
        ):
            raise ValueError("Invalid Ray endpoint or response limit")
        self._url = destination
        self._maximum = maximum_response_bytes
        headers = {"accept": "application/x-ndjson", "accept-encoding": "identity"}
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            headers=headers,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )

    async def stream(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """Delay final accounting until clean EOF so partial transport failure cannot pass."""
        deadline = asyncio.get_running_loop().time() + request.timeout_seconds
        outbound = self._client.build_request(
            "POST", self._url, json=request.model_dump(), timeout=request.timeout_seconds
        )
        try:
            async with asyncio.timeout_at(deadline):
                response = await self._client.send(outbound, stream=True)
            try:
                if response.status_code != 200:
                    raise EngineUnavailableError("Ray routing request failed")
                if (
                    response.headers.get("content-type", "").split(";", 1)[0].strip()
                    != "application/x-ndjson"
                    or response.headers.get("content-encoding", "identity") != "identity"
                ):
                    raise EngineProtocolError("Invalid Ray response content type or encoding")
                frames = envelopes(response, self._maximum)
                selected: str | None = None
                terminal: EngineToken | None = None
                visible = False
                while True:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise TimeoutError("Ray request deadline expired")
                    try:
                        # Keep timeout ownership inside each await, never across a consumer yield.
                        async with asyncio.timeout_at(deadline):
                            envelope = await anext(frames)
                    except StopAsyncIteration:
                        break
                    token = envelope.token
                    if terminal is not None or selected not in {None, envelope.replica_id}:
                        raise EngineProtocolError(
                            "Ray stream changed replica or continued after finish"
                        )
                    selected = envelope.replica_id
                    visible = visible or bool(token.text)
                    if token.finish_reason is not None:
                        if (
                            token.generated_tokens > request.max_tokens
                            or (visible and token.generated_tokens == 0)
                            or token.text
                        ):
                            raise EngineProtocolError("Invalid Ray terminal accounting")
                        terminal = token
                    elif token.generated_tokens != 0:
                        raise EngineProtocolError("Ray delta carried premature token accounting")
                    else:
                        yield token
                if terminal is None:
                    raise EngineProtocolError("Ray stream omitted terminal accounting")
                yield terminal
            finally:
                await response.aclose()
        except httpx.TimeoutException:
            raise TimeoutError("Ray HTTP operation timed out") from None
        except httpx.HTTPError:
            raise EngineUnavailableError("Ray HTTP transport failed") from None

    async def close(self) -> None:
        """Close the gateway-owned pool; the deployment owner manages Ray's cluster lifecycle."""
        await self._client.aclose()
