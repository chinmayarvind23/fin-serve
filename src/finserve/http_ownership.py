"""Retain local response ownership until asynchronous transport cleanup finishes."""

import asyncio
from collections.abc import AsyncIterator

import httpx


class HTTPClosureError(RuntimeError):
    """The local cleanup task ended without verifying transport release; retry remains unsafe."""


class OwnedCloseStream(httpx.AsyncByteStream):
    """Cancellation drains local stream cleanup; it makes no remote inference-stop claim."""

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        """Wrap a response before iteration can trigger HTTPX's implicit end-of-body close."""
        self.stream = stream
        self.close_task: asyncio.Task[None] | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Delegate incremental bytes without buffering or changing their timing."""
        async for chunk in self.stream:
            yield chunk

    async def aclose(self) -> None:
        """Shield the exact cleanup task from initial and repeated caller cancellation."""
        if self.close_task is None:
            self.close_task = asyncio.create_task(self.stream.aclose())
        try:
            await asyncio.shield(self.close_task)
        except asyncio.CancelledError:
            while not self.close_task.done():
                try:
                    await asyncio.shield(self.close_task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if self.close_task.cancelled() or self.close_task.exception() is not None:
                raise HTTPClosureError("local HTTP cleanup failed") from None
            raise
        except Exception:
            raise HTTPClosureError("local HTTP cleanup failed") from None


def own_response(response: httpx.Response) -> None:
    """Install cleanup ownership before reading, including HTTP error and partial-body paths."""
    if not isinstance(response.stream, httpx.AsyncByteStream):
        raise TypeError("asynchronous response stream required")
    response.stream = OwnedCloseStream(response.stream)
