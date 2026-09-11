"""Untrained image-conditioned visual-token decoding with a fixed-shape JAX/Flax path."""

from __future__ import annotations

import hashlib
import importlib
import io
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

IMAGE_SIZE = 8
VISUAL_TOKENS = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_SIZE = 32
PALETTE: tuple[tuple[int, int, int], ...] = (
    (15, 23, 42),
    (51, 65, 85),
    (100, 116, 139),
    (203, 213, 225),
    (248, 250, 252),
    (239, 68, 68),
    (249, 115, 22),
    (250, 204, 21),
    (132, 204, 22),
    (34, 197, 94),
    (20, 184, 166),
    (6, 182, 212),
    (59, 130, 246),
    (99, 102, 241),
    (168, 85, 247),
    (236, 72, 153),
)
Channel = Annotated[int, Field(ge=0, le=255, strict=True)]
RGB = tuple[Channel, Channel, Channel]
ImageRow = Annotated[list[RGB], Field(min_length=IMAGE_SIZE, max_length=IMAGE_SIZE)]
RGBImage = Annotated[list[ImageRow], Field(min_length=IMAGE_SIZE, max_length=IMAGE_SIZE)]
VisualToken = Annotated[int, Field(ge=0, lt=len(PALETTE), strict=True)]
TokenGrid = Annotated[list[VisualToken], Field(min_length=VISUAL_TOKENS, max_length=VISUAL_TOKENS)]


class VisualRequest(BaseModel):
    """Bound batch and image shapes before optional numerical dependencies allocate arrays."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    images: list[RGBImage] = Field(min_length=1, max_length=8)


class VisualOutput(BaseModel):
    """Carry palette IDs and model identity; the output makes no semantic image-quality claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    token_ids: list[TokenGrid] = Field(min_length=1, max_length=8)
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int


class JAXVisualReference:
    """A seeded recurrent visual decoder; Flax owns parameters and JAX executes pure transforms.

    This is an actual image-conditioned autoregressive model, but its random parameters
    have never been trained. Palette pixels illustrate mechanics rather than useful images.
    """

    def __init__(self, seed: int = 17) -> None:
        """Load optional CPU dependencies lazily and retain one JIT cache per model instance."""
        if not 0 <= seed < 2**32:
            raise ValueError("seed must fit an unsigned 32-bit PRNG key")
        try:
            self._jax = importlib.import_module("jax")
            self._jnp = importlib.import_module("jax.numpy")
            nn = importlib.import_module("flax.linen")
        except ImportError:
            raise RuntimeError(
                "Install finserve[multimodal] for the JAX visual reference"
            ) from None
        self.seed = seed
        self._device = self._jax.devices("cpu")[0]
        self._layers: dict[str, Any] = {
            "image": nn.Dense(HIDDEN_SIZE),
            "initial": nn.Dense(HIDDEN_SIZE),
            "recurrent": nn.Dense(HIDDEN_SIZE, use_bias=False),
            "head": nn.Dense(len(PALETTE)),
            "token": nn.Embed(len(PALETTE) + 1, HIDDEN_SIZE),
            "position": nn.Embed(VISUAL_TOKENS, HIDDEN_SIZE),
        }
        with self._jax.default_device(self._device):
            self._parameters = self._initialize_parameters()
        self.model_sha256 = self._fingerprint()
        self._compiled = self._jax.jit(self._decode_scan)

    def _initialize_parameters(self) -> dict[str, Any]:
        """Split the explicit PRNG key once per layer; inference itself performs no sampling."""
        jnp = self._jnp
        samples = {
            "image": jnp.zeros((1, 3)),
            "initial": jnp.zeros((1, HIDDEN_SIZE)),
            "recurrent": jnp.zeros((1, HIDDEN_SIZE)),
            "head": jnp.zeros((1, HIDDEN_SIZE)),
            "token": jnp.zeros((1,), dtype=jnp.int32),
            "position": jnp.zeros((1,), dtype=jnp.int32),
        }
        keys = self._jax.random.split(self._jax.random.key(self.seed), len(samples))
        return {
            name: self._layers[name].init(keys[index], sample)
            for index, (name, sample) in enumerate(samples.items())
        }

    def _fingerprint(self) -> str:
        """Hash parameter bytes so evidence identifies actual weights as well as the seed."""
        digest = hashlib.sha256()
        for parameter in self._jax.tree_util.tree_leaves(self._parameters):
            parameter.block_until_ready()
            digest.update(str(parameter.shape).encode())
            digest.update(str(parameter.dtype).encode())
            digest.update(parameter.tobytes())
        return digest.hexdigest()

    def _apply(self, name: str, inputs: Any) -> Any:
        """Share the functional Flax init/apply parameter boundary across both decode paths."""
        return self._layers[name].apply(self._parameters[name], inputs)

    def _encode(self, images: Any) -> tuple[Any, tuple[Any, Any]]:
        """Condition output positions on input pixels and initialize hidden state from the image."""
        jnp = self._jnp
        normalized = images.reshape((images.shape[0], VISUAL_TOKENS, 3)) / 127.5 - 1
        features = self._apply("image", normalized)
        positions = self._apply("position", jnp.arange(VISUAL_TOKENS, dtype=jnp.int32))
        hidden = jnp.tanh(self._apply("initial", features.mean(axis=1)))
        previous = jnp.full((images.shape[0],), len(PALETTE), dtype=jnp.int32)
        return features + positions[None, :, :], (hidden, previous)

    def _transition(self, features: Any, carry: tuple[Any, Any], position: Any) -> tuple[Any, Any]:
        """Previous token and recurrent state make pixel generation genuinely autoregressive."""
        hidden, previous = carry
        hidden = self._jnp.tanh(
            self._apply("recurrent", hidden)
            + self._apply("token", previous)
            + features[:, position, :]
        )
        return hidden, self._apply("head", hidden)

    def _decode_scan(self, images: Any) -> Any:
        """Use a fixed-size carry and 64 iterations so compilation never grows prefix arrays."""
        features, initial = self._encode(images)

        def step(carry: tuple[Any, Any], position: Any) -> tuple[tuple[Any, Any], Any]:
            """Feed selected palette IDs into subsequent transitions without Python dispatch."""
            hidden, logits = self._transition(features, carry, position)
            token = logits.argmax(axis=-1).astype(self._jnp.int32)
            return (hidden, token), token

        _, tokens = self._jax.lax.scan(step, initial, self._jnp.arange(VISUAL_TOKENS))
        return tokens.T

    def _decode_python(self, images: Any) -> Any:
        """Retain a Python-loop correctness oracle with the same layers and greedy decisions."""
        features, carry = self._encode(images)
        tokens: list[Any] = []
        for position in range(VISUAL_TOKENS):
            hidden, logits = self._transition(features, carry, position)
            token = logits.argmax(axis=-1).astype(self._jnp.int32)
            tokens.append(token)
            carry = (hidden, token)
        return self._jnp.stack(tokens, axis=1)

    def generate(self, request: VisualRequest, *, compiled: bool = True) -> VisualOutput:
        """Run on CPU and synchronize before returning, making caller timing include actual work."""
        with self._jax.default_device(self._device):
            images = self._jnp.asarray(request.images, dtype=self._jnp.float32)
            tokens = self._compiled(images) if compiled else self._decode_python(images)
            tokens.block_until_ready()
        return VisualOutput(
            token_ids=tokens.tolist(), model_sha256=self.model_sha256, seed=self.seed
        )

    def prefix_logits(self, request: VisualRequest, prefix: list[int]) -> list[list[float]]:
        """Expose conditional logits for tests of input conditioning and prior-token influence."""
        if len(prefix) >= VISUAL_TOKENS or any(
            token < 0 or token >= len(PALETTE) for token in prefix
        ):
            raise ValueError("visual prefix must fit the grid and palette")
        with self._jax.default_device(self._device):
            images = self._jnp.asarray(request.images, dtype=self._jnp.float32)
            features, carry = self._encode(images)
            for position, supplied_token in enumerate(prefix):
                hidden, _ = self._transition(features, carry, position)
                carry = (
                    hidden,
                    self._jnp.full((images.shape[0],), supplied_token, dtype=self._jnp.int32),
                )
            _, logits = self._transition(features, carry, len(prefix))
            logits.block_until_ready()
        return logits.tolist()


def output_png(output: VisualOutput, index: int = 0, scale: int = 16) -> bytes:
    """Render palette IDs with nearest-neighbor scaling, preserving the actual model output."""
    if not 1 <= scale <= 64 or not 0 <= index < len(output.token_ids):
        raise ValueError("visual render index or scale is outside supported bounds")
    image_module = importlib.import_module("PIL.Image")
    image = image_module.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    image.putdata([PALETTE[token] for token in output.token_ids[index]])
    image = image.resize((IMAGE_SIZE * scale, IMAGE_SIZE * scale), image_module.Resampling.NEAREST)
    target = io.BytesIO()
    image.save(target, format="PNG")
    return target.getvalue()
