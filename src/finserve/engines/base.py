"""The stream iterator owns request-local resources and must close on cancellation."""

from collections.abc import AsyncIterator
from typing import Protocol

from finserve.contracts.inference import EngineToken, InferenceRequest


class Engine(Protocol):
    """Transport-independent streaming keeps routing separate from token scheduling."""

    def stream(self, request: InferenceRequest) -> AsyncIterator[EngineToken]:
        """Yield generated text and authoritative counts; release resources on close."""
        ...

    async def close(self) -> None:
        """Drain engine-owned clients when the application lifespan exits."""
        ...
