"""Optional CPU preprocessing hop, measured separately from coupled vLLM vision/decoding."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from finserve.contracts.vision import MAX_PNG_BYTES
from finserve.gateway.admission import Admission
from finserve.gateway.vision import drain, vision_error
from finserve.multimodal.images import PreparedImage, bounded_png, prepare_png


def create_preprocessor_app(api_key: str) -> FastAPI:
    """Create an authenticated, one-slot binary PNG stage; bind only to a trusted local network."""
    if len(api_key) < 24:
        raise ValueError("Preprocessor key must contain at least 24 characters")
    app, admission = FastAPI(), Admission(1)

    @app.post("/internal/prepare", response_model=None)
    async def prepare(request: Request) -> Response:
        """Retain capacity until native decoding drains, even when the HTTP caller disappears."""
        headers = request.headers.getlist("authorization")
        if len(headers) != 1 or not hmac.compare_digest(
            headers[0].encode(), ("Bearer " + api_key).encode()
        ):
            return vision_error("UNAUTHORIZED", 401)
        if not admission.acquire():
            return vision_error("OVERLOADED", 429)
        try:
            raw = bytearray()
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > MAX_PNG_BYTES:
                        return vision_error("BODY_TOO_LARGE", 413)
                    raw.extend(chunk)
            started = time.perf_counter()
            task = asyncio.create_task(asyncio.to_thread(prepare_png, bytes(raw)))
            try:
                async with asyncio.timeout(10):
                    image = await asyncio.shield(task)
            finally:
                if not task.done():
                    try:
                        await drain(task)
                    except Exception:
                        pass
            elapsed = time.perf_counter() - started
            return Response(
                image.png,
                media_type="image/png",
                headers={
                    "X-PNG-SHA256": image.sha256,
                    "X-Source-SHA256": image.source_sha256,
                    "X-Preprocess-Seconds": repr(elapsed),
                },
            )
        except ValueError:
            return vision_error("INVALID_PNG", 422)
        except TimeoutError:
            return vision_error("DEADLINE_EXCEEDED", 504)
        finally:
            admission.release()

    return app


class PreprocessorClient:
    """Binary HTTP is an explicit stage boundary; timings never imply one-way network latency."""

    def __init__(
        self, base_url: str, api_key: str, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        """Only trusted startup configuration can select the CPU worker destination."""
        url = httpx.URL(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise ValueError("Invalid preprocessing endpoint")
        self._url = str(url).rstrip("/") + "/internal/prepare"
        self._client = httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            headers={"authorization": "Bearer " + api_key, "accept-encoding": "identity"},
        )

    async def prepare(self, raw: bytes) -> tuple[PreparedImage, float, float]:
        """Return verified bytes, client roundtrip and worker duration from separate clocks."""
        if len(raw) > MAX_PNG_BYTES:
            raise ValueError("PNG exceeds byte limit")
        started = time.perf_counter()
        async with asyncio.timeout(15):
            async with self._client.stream("POST", self._url, content=raw, timeout=15) as response:
                if response.status_code != 200:
                    raise RuntimeError("Preprocessor request failed")
                if response.headers.get("content-type", "").split(";", 1)[0].strip() != "image/png":
                    raise RuntimeError("Preprocessor response is not PNG")
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise RuntimeError("Preprocessor compression unsupported")
                result = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(result) + len(chunk) > MAX_PNG_BYTES:
                        raise RuntimeError("Preprocessor response exceeds byte limit")
                    result.extend(chunk)
                source_sha, sha = (
                    hashlib.sha256(raw).hexdigest(),
                    hashlib.sha256(result).hexdigest(),
                )
                if (
                    response.headers.get("x-png-sha256") != sha
                    or response.headers.get("x-source-sha256") != source_sha
                ):
                    raise RuntimeError("Preprocessor image identity mismatch")
                worker_seconds = float(response.headers["x-preprocess-seconds"])
                if not math.isfinite(worker_seconds) or not 0 <= worker_seconds <= 15:
                    raise RuntimeError("Invalid preprocessing timing")
                if worker_seconds > time.perf_counter() - started + 0.001:
                    raise RuntimeError("Worker timing exceeds measured HTTP roundtrip")
        sanitized = bounded_png(bytes(result))
        if sanitized != bytes(result) or sanitized[25] != 2:
            raise RuntimeError("Preprocessor response must be metadata-free RGB PNG")
        width, height = (
            int.from_bytes(sanitized[16:20], "big"),
            int.from_bytes(sanitized[20:24], "big"),
        )
        return (
            PreparedImage(bytes(result), width, height, source_sha, sha),
            time.perf_counter() - started,
            worker_seconds,
        )

    async def close(self) -> None:
        """Release the independent CPU-worker connection pool after stage work drains."""
        await self._client.aclose()


def from_env() -> FastAPI:
    """Uvicorn factory keeps the internal credential out of command lines and evidence logs."""
    return create_preprocessor_app(os.environ["FINSERVE_PREPROCESSOR_KEY"])
