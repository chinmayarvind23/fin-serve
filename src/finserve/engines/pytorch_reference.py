"""An untrained decoder for studying causal attention, never financial answers.

The cache stores projected keys/values, not output tokens. Full-prefix recomputation
is intentionally retained as the correctness oracle for the cached implementation.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from finserve.contracts.inference import EngineToken, InferenceRequest

if TYPE_CHECKING:
    from torch import Tensor

ALPHABET = "\n" + "".join(chr(value) for value in range(32, 127))
BOS_TOKEN_ID = len(ALPHABET)
MAX_CONTEXT = 4096


@dataclass(frozen=True)
class KVCache:
    """Keep request-local projections; sharing these would leak prompt state."""

    keys: Tensor
    values: Tensor

    @property
    def length(self) -> int:
        """Derive the absolute next position from the actual cache extent."""
        return int(self.keys.shape[0])


class TinyCausalDecoder:
    """One fixed random attention layer with a residual feed-forward transform.

    Float64 and a small width prioritize an inspectable parity oracle over speed.
    This is neither trained nor representative of production model performance.
    """

    def __init__(self, device: str = "cpu", seed: int = 17) -> None:
        """Load optional PyTorch lazily and seed a private CPU RNG, preserving global RNG."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install finserve[reference] to use the PyTorch engine") from exc
        self._torch = torch
        self.device = torch.device(device)
        self.width = 32
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def weights(rows: int, columns: int, scale: float) -> Tensor:
            """Initialize on CPU so device selection does not change the seeded model."""
            return (
                torch.randn(rows, columns, generator=generator, dtype=torch.float64) * scale
            ).to(self.device)

        self.embedding = weights(BOS_TOKEN_ID + 1, self.width, 0.2)
        self.positions = weights(MAX_CONTEXT, self.width, 0.05)
        self.query = weights(self.width, self.width, 1 / math.sqrt(self.width))
        self.key = weights(self.width, self.width, 1 / math.sqrt(self.width))
        self.value = weights(self.width, self.width, 1 / math.sqrt(self.width))
        self.feed_forward = weights(self.width, self.width, 1 / math.sqrt(self.width))
        self.output = weights(self.width, len(ALPHABET), 1 / math.sqrt(self.width))

    def forward(self, token_ids: list[int], cache: KVCache | None = None) -> tuple[Tensor, KVCache]:
        """Compute logits only for supplied tokens, masking with absolute positions.

        A cached one-token query must attend every preceding key. Applying a local
        triangular mask to that rectangular matrix would incorrectly expose only
        the first key. Explicit absolute positions also support multi-token chunks.
        """
        offset = 0 if cache is None else cache.length
        if not token_ids or offset + len(token_ids) > MAX_CONTEXT:
            raise ValueError("Decoder input must be nonempty and fit the context window")
        if any(token < 0 or token > BOS_TOKEN_ID for token in token_ids):
            raise ValueError("Token ID is outside the reference vocabulary")
        torch = self._torch
        with torch.inference_mode():
            ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            positions = torch.arange(offset, offset + len(token_ids), device=self.device)
            hidden = self.embedding[ids] + self.positions[positions]
            queries, keys, values = hidden @ self.query, hidden @ self.key, hidden @ self.value
            if cache is not None:
                keys = torch.cat((cache.keys, keys), dim=0)
                values = torch.cat((cache.values, values), dim=0)
            scores = (queries @ keys.T) / math.sqrt(self.width)
            key_positions = torch.arange(keys.shape[0], device=self.device)
            future = key_positions.unsqueeze(0) > positions.unsqueeze(1)
            scores = scores.masked_fill(future, float("-inf"))
            attended = torch.softmax(scores, dim=-1) @ values
            residual = hidden + attended
            hidden = residual + torch.tanh(residual @ self.feed_forward)
            return hidden @ self.output, KVCache(keys=keys, values=values)

    def next_token(self, token_ids: list[int], cache: KVCache | None) -> tuple[int, KVCache]:
        """Use greedy decoding so cache experiments share one deterministic oracle."""
        logits, next_cache = self.forward(token_ids, cache)
        return int(logits[-1].argmax().item()), next_cache


class PyTorchReferenceEngine:
    """Async adapter with isolated caches and at most one in-flight step per stream."""

    def __init__(self, cached: bool = True, device: str = "cpu") -> None:
        """Select the cache ablation while retaining identical model weights and tokenizer."""
        self.decoder = TinyCausalDecoder(device=device)
        self.cached = cached
        self._closed = False
        self._pending: set[asyncio.Task[tuple[int, KVCache]]] = set()

    @property
    def active_steps(self) -> int:
        """Expose native operations still consuming resources for shutdown verification."""
        return len(self._pending)

    @staticmethod
    def encode(prompt: str) -> list[int]:
        """Use one printable character per token; unsupported Unicode maps to '?'."""
        unknown = ALPHABET.index("?")
        mapping = {character: index for index, character in enumerate(ALPHABET)}
        return [BOS_TOKEN_ID, *(mapping.get(character, unknown) for character in prompt)]

    async def _step(self, tokens: list[int], cache: KVCache | None) -> tuple[int, KVCache]:
        """Offload blocking tensor work and drain it before propagating cancellation.

        Cancelling an asyncio thread wrapper cannot interrupt a native kernel. We
        wait for the current bounded step; no background generation loop survives.
        """
        worker = asyncio.create_task(asyncio.to_thread(self.decoder.next_token, tokens, cache))
        self._pending.add(worker)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Repeated disconnect/deadline cancellation still must not abandon the kernel.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not worker.cancelled():
                worker.exception()  # Retrieve a late worker failure; cancellation stays terminal.
            raise
        finally:
            self._pending.discard(worker)

    async def stream(self, request: InferenceRequest) -> AsyncGenerator[EngineToken, None]:
        """Prefill once, then project one token per step when caching is enabled.

        Ingress owns deadlines/admission. Closing this iterator drops its private
        cache; the adapter never retries after emitting externally visible output.
        """
        if self._closed:
            raise RuntimeError("Reference engine is closed")
        if request.temperature != 0:
            raise ValueError("The educational reference engine supports temperature=0 only")
        tokens = self.encode(request.prompt)
        if len(tokens) + request.max_tokens > MAX_CONTEXT:
            raise ValueError(f"Prompt plus output must fit {MAX_CONTEXT} reference tokens")
        cache: KVCache | None = None
        try:
            for _ in range(request.max_tokens):
                if self._closed:
                    raise RuntimeError("Reference engine closed during generation")
                inputs = tokens[-1:] if self.cached and cache is not None else tokens.copy()
                token, next_cache = await self._step(inputs, cache if self.cached else None)
                cache = next_cache if self.cached else None
                tokens.append(token)
                yield EngineToken(text=ALPHABET[token], token_id=token, generated_tokens=1)
        finally:
            cache = None
            tokens.clear()

    async def close(self) -> None:
        """Reject new steps and drain active native work without unloading shared tensors."""
        self._closed = True
        if self._pending:
            drained = asyncio.gather(*tuple(self._pending), return_exceptions=True)
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                while not drained.done():
                    try:
                        await asyncio.shield(drained)
                    except asyncio.CancelledError:
                        continue
                raise
