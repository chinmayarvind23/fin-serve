"""Bound image decoding before Pillow and produce metadata-free, deterministic RGB PNGs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import io
import struct
import zlib
from contextlib import closing
from dataclasses import dataclass
from typing import Protocol, cast

from finserve.contracts.vision import MAX_PNG_BYTES

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_IMAGE_SIDE = 512


class ImageHandle(Protocol):
    """Keep the optional Pillow dependency behind a small statically checked boundary."""

    size: tuple[int, int]

    def convert(self, mode: str) -> ImageHandle:
        """Materialize bounded pixels in a known color space."""
        ...

    def save(self, fp: io.BytesIO, *, format: str, compress_level: int) -> None:
        """Encode the same pixels with an explicit stable compression setting."""
        ...

    def close(self) -> None:
        """Release decoder resources before returning from the preprocessing stage."""
        ...


class ImageModule(Protocol):
    """Describe only the image operations used; importing this module never imports JAX."""

    def open(self, fp: io.BytesIO, *, formats: list[str]) -> ImageHandle:
        """Restrict format dispatch to PNG after bounded structural validation."""
        ...

    def new(self, mode: str, size: tuple[int, int], color: tuple[int, ...]) -> ImageHandle:
        """Use an explicit white alpha background rather than backend-dependent defaults."""
        ...

    def alpha_composite(self, first: ImageHandle, second: ImageHandle) -> ImageHandle:
        """Flatten transparent pixels consistently across local and remote preprocessing."""
        ...


@dataclass(frozen=True)
class PreparedImage:
    """Canonical bytes and identities are evidence; raw image content is never telemetry."""

    png: bytes
    width: int
    height: int
    source_sha256: str
    sha256: str

    def data_url(self) -> str:
        """Build the only permitted upstream media address from verified local bytes."""
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


def decode_inline_png(value: str) -> bytes:
    """Reject noncanonical base64 and destination syntax before any media decoder runs."""
    if len(value) > 4 * ((MAX_PNG_BYTES + 2) // 3):
        raise ValueError("PNG exceeds byte limit")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Image must be strict inline PNG base64") from None
    if len(raw) > MAX_PNG_BYTES or base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("PNG exceeds byte limit or base64 is noncanonical")
    return raw


def bounded_png(raw: bytes) -> bytes:
    """Check dimensions and chunk CRCs, stripping metadata before Pillow can decompress it.

    This slice supports noninterlaced 8-bit RGB/RGBA PNG, including ordinary browser
    screenshots. Palette, animation and other bit depths fail explicitly. Ancillary
    chunks are omitted so compressed text/ICC metadata never reaches Pillow.
    """
    if len(raw) > MAX_PNG_BYTES or not raw.startswith(PNG_SIGNATURE):
        raise ValueError("Invalid or oversized PNG")
    chunks: list[bytes] = []
    offset, saw_data, ended = 8, False, False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise ValueError("Truncated PNG chunk")
        length = int.from_bytes(raw[offset : offset + 4], "big")
        end = offset + 12 + length
        if end > len(raw):
            raise ValueError("Truncated PNG payload")
        kind = raw[offset + 4 : offset + 8]
        data = raw[offset + 8 : end - 4]
        if zlib.crc32(kind + data) != int.from_bytes(raw[end - 4 : end], "big"):
            raise ValueError("Invalid PNG checksum")
        if not chunks:
            _check_header(kind, data)
        elif kind == b"IHDR" or kind in (b"acTL", b"fcTL", b"fdAT", b"tRNS"):
            raise ValueError("Repeated header, animation or RGB color-key transparency unsupported")
        if kind in (b"IHDR", b"IDAT", b"IEND"):
            chunks.append(raw[offset:end])
        elif not kind[0] & 32:
            raise ValueError("Unsupported critical PNG chunk")
        saw_data = saw_data or kind == b"IDAT"
        if kind == b"IEND":
            if data or not saw_data or end != len(raw):
                raise ValueError("Invalid PNG end")
            ended = True
            break
        offset = end
    if not ended:
        raise ValueError("PNG omitted end")
    return PNG_SIGNATURE + b"".join(chunks)


def _check_header(kind: bytes, data: bytes) -> None:
    """Enforce the pixel allocation budget before any native image decoder sees input."""
    if kind != b"IHDR" or len(data) != 13:
        raise ValueError("PNG must start with IHDR")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
    if not 1 <= width <= MAX_IMAGE_SIDE or not 1 <= height <= MAX_IMAGE_SIDE:
        raise ValueError("PNG dimensions exceed limit")
    if depth != 8 or color not in (2, 6) or (compression, filtering, interlace) != (0, 0, 0):
        raise ValueError("PNG must be noninterlaced 8-bit RGB or RGBA")


def prepare_png(raw: bytes) -> PreparedImage:
    """Decode under a bounded owner, flatten alpha to white, and remove all metadata.

    Callers offload this synchronous CPU work under admission and drain the thread
    before returning capacity on cancellation. Model resizing remains inside vLLM.
    """
    sanitized = bounded_png(raw)
    image_module = cast(ImageModule, importlib.import_module("PIL.Image"))
    try:
        with closing(image_module.open(io.BytesIO(sanitized), formats=["PNG"])) as source:
            with closing(source.convert("RGBA")) as rgba:
                with closing(image_module.new("RGBA", source.size, (255, 255, 255, 255))) as bg:
                    with closing(image_module.alpha_composite(bg, rgba)) as flattened:
                        with closing(flattened.convert("RGB")) as rgb:
                            target = io.BytesIO()
                            rgb.save(target, format="PNG", compress_level=6)
                            result = target.getvalue()
                            width, height = rgb.size
    except (OSError, ValueError, SyntaxError):
        raise ValueError("PNG pixel decoding failed") from None
    if len(result) > MAX_PNG_BYTES:
        raise ValueError("Canonical PNG exceeds byte limit")
    return PreparedImage(
        result, width, height, hashlib.sha256(raw).hexdigest(), hashlib.sha256(result).hexdigest()
    )
