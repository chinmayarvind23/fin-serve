"""Check actual attention mechanics, deterministic parity, and cancellation ownership."""

import asyncio
import threading
from unittest.mock import patch

import pytest

from finserve.contracts.inference import InferenceRequest

torch = pytest.importorskip("torch")

from finserve.engines.pytorch_reference import (  # noqa: E402
    BOS_TOKEN_ID,
    MAX_CONTEXT,
    KVCache,
    PyTorchReferenceEngine,
    TinyCausalDecoder,
)


def test_future_tokens_do_not_change_prefix_logits() -> None:
    """A changed suffix must have no influence on earlier causal positions."""
    decoder = TinyCausalDecoder()
    first, _ = decoder.forward([BOS_TOKEN_ID, 3, 5, 7])
    changed, _ = decoder.forward([BOS_TOKEN_ID, 3, 20, 21])
    torch.testing.assert_close(first[:2], changed[:2], rtol=0, atol=0)
    assert not torch.equal(first[2:], changed[2:])


def test_cached_logits_match_full_prefix_and_cache_grows() -> None:
    """Both rectangular cache attention and multi-token continuation match the oracle."""
    decoder = TinyCausalDecoder()
    prefix = [BOS_TOKEN_ID, 3, 5, 7]
    _, cache = decoder.forward(prefix)
    cached, cache = decoder.forward([11, 13], cache)
    full, _ = decoder.forward([*prefix, 11, 13])
    assert cache.length == 6
    torch.testing.assert_close(cached, full[-2:], rtol=1e-12, atol=1e-12)


def test_cache_is_request_local_and_rng_is_private() -> None:
    """Model construction must not reseed host randomness or share mutable prefix state."""
    before = torch.random.get_rng_state().clone()
    decoder = TinyCausalDecoder()
    assert torch.equal(before, torch.random.get_rng_state())
    _, first_cache = decoder.forward([BOS_TOKEN_ID, 3])
    saved = first_cache.keys.clone()
    decoder.forward([BOS_TOKEN_ID, 9, 8])
    decoder.forward([4], first_cache)
    assert first_cache.length == 2
    assert torch.equal(first_cache.keys, saved)


def test_streamed_cached_uncached_greedy_parity() -> None:
    """Compare real generated IDs over multiple prompts, including Unicode replacement."""

    async def run() -> None:
        """Exercise the public stream protocol in one event loop without extra test plugins."""
        cached = PyTorchReferenceEngine()
        uncached = PyTorchReferenceEngine(cached=False)
        for prompt in ("Finance", "", "risk \u03bb"):
            request = InferenceRequest(prompt=prompt or " ", max_tokens=16)
            first = [token async for token in cached.stream(request)]
            second = [token async for token in uncached.stream(request)]
            assert first == second
            assert len(first) == 16
            assert all(token.generated_tokens == 1 and len(token.text) == 1 for token in first)
        await cached.close()
        await uncached.close()

    asyncio.run(run())


def test_cancellation_drains_active_step_without_generating_more() -> None:
    """Cancellation waits for native work and does not orphan a producer loop."""

    async def run() -> None:
        """Coordinate a blocked worker explicitly instead of relying on fragile timing sleeps."""
        engine = PyTorchReferenceEngine()
        entered = threading.Event()
        release = threading.Event()
        original = engine.decoder.next_token

        def blocked(tokens: list[int], cache: KVCache | None) -> tuple[int, KVCache]:
            """Hold one native operation so cancellation arrives during actual worker ownership."""
            entered.set()
            assert release.wait(5)
            return original(tokens, cache)

        with patch.object(engine.decoder, "next_token", side_effect=blocked) as mocked:
            stream = engine.stream(InferenceRequest(prompt="hello", max_tokens=8))
            task = asyncio.ensure_future(anext(stream))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert engine.active_steps == 1
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await stream.aclose()
            assert mocked.call_count == 1
            assert engine.active_steps == 0
        await engine.close()

    asyncio.run(run())


def test_validation_and_close() -> None:
    """Reject unsupported sampling, excessive context, and reuse after shutdown explicitly."""

    async def run() -> None:
        """Validate errors at first iteration, matching async-generator execution semantics."""
        engine = PyTorchReferenceEngine()
        for request, match in (
            (InferenceRequest(prompt="hello", temperature=0.5), "temperature"),
            (InferenceRequest(prompt="x" * MAX_CONTEXT, max_tokens=1), "fit"),
        ):
            with pytest.raises(ValueError, match=match):
                await anext(engine.stream(request))
        await engine.close()
        await engine.close()
        with pytest.raises(RuntimeError, match="closed"):
            await anext(engine.stream(InferenceRequest(prompt="hello")))

    asyncio.run(run())


def test_close_drains_work_even_if_shutdown_is_cancelled() -> None:
    """A shutdown cancellation must not cancel the thread wrapper and lose kernel ownership."""

    async def run() -> None:
        """Hold generation while closing so the test observes the resource lifetime directly."""
        engine = PyTorchReferenceEngine()
        entered, release = threading.Event(), threading.Event()
        original = engine.decoder.next_token

        def blocked(tokens: list[int], cache: KVCache | None) -> tuple[int, KVCache]:
            """Model one kernel that finishes only when the test releases it."""
            entered.set()
            assert release.wait(5)
            return original(tokens, cache)

        with patch.object(engine.decoder, "next_token", side_effect=blocked):
            stream = engine.stream(InferenceRequest(prompt="hello", max_tokens=2))
            first = asyncio.ensure_future(anext(stream))
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(engine.close())
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            await asyncio.sleep(0)
            assert not closing.done()
            release.set()
            await first
            with pytest.raises(asyncio.CancelledError):
                await closing
            with pytest.raises(RuntimeError, match="closed during"):
                await anext(stream)
            assert engine.active_steps == 0
            await stream.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("tokens", [[], [-1], [BOS_TOKEN_ID + 1]])
def test_decoder_rejects_invalid_input(tokens: list[int]) -> None:
    """Fail before indexing tensors when callers violate the internal token contract."""
    with pytest.raises(ValueError):
        TinyCausalDecoder().forward(tokens)
