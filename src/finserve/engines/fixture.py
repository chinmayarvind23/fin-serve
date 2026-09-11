"""Deterministic transport fixture, explicitly unsuitable for model quality claims."""

import asyncio
from collections.abc import AsyncIterator

from finserve.contracts.inference import EngineToken, InferenceRequest


class FixtureEngine:
    """A character token vocabulary makes fixture counts exact and network tests cheap."""

    async def stream(self, request: InferenceRequest) -> AsyncIterator[EngineToken]:
        """Echo bounded prompt characters; yield control so disconnects can cancel work."""
        for char in (request.prompt + " ")[: request.max_tokens]:
            await asyncio.sleep(0)
            yield EngineToken(text=char, token_id=ord(char))

    async def close(self) -> None:
        """The fixture owns no external resources."""
