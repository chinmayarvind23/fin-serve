"""Local response cleanup has one retained task and never silently loses ownership."""

import asyncio

import httpx
import pytest

from finserve.http_ownership import HTTPClosureError, OwnedCloseStream, own_response


async def test_close_replay_owns_one_task() -> None:
    """Implicit EOF and context exit may both close the same response without duplicate cleanup."""
    calls = 0

    class Stream(httpx.AsyncByteStream):
        """Count the exact underlying resource-close operation."""

        async def aclose(self) -> None:
            """A second wrapper call must reuse this completed cleanup task."""
            nonlocal calls
            calls += 1

    stream = OwnedCloseStream(Stream())
    await stream.aclose()
    await stream.aclose()
    assert calls == 1


@pytest.mark.parametrize("cancelled", [False, True])
async def test_close_failure_is_explicitly_unresolved(cancelled: bool) -> None:
    """Failed or self-cancelled cleanup cannot become a retryable producer cancellation."""

    class Stream(httpx.AsyncByteStream):
        """Fail cleanup without any caller cancellation."""

        async def aclose(self) -> None:
            """No successful local transport-release observation exists."""
            if cancelled:
                raise asyncio.CancelledError
            raise OSError("fixture cleanup failure")

    with pytest.raises(HTTPClosureError):
        await OwnedCloseStream(Stream()).aclose()


def test_sync_response_is_not_an_owned_async_transport() -> None:
    """Reject an incompatible caller-supplied stream before attempting asynchronous cleanup."""
    response = httpx.Response(200, stream=httpx.SyncByteStream())
    with pytest.raises(TypeError):
        own_response(response)
