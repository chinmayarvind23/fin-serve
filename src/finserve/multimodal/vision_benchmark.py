"""Paired local-versus-HTTP preprocessing experiment with identical pretrained model inputs."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import struct
import subprocess
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from finserve.benchmark.runner import prepare_output, write_json
from finserve.contracts.vision import VISION_MODEL, VISION_REVISION, VisionRequest
from finserve.engines.vision_openai import OpenAIVisionEngine, VisionEngine
from finserve.multimodal.images import PNG_SIGNATURE, PreparedImage, prepare_png
from finserve.multimodal.preprocess_http import PreprocessorClient

COLORS = {"red": (255, 0, 0), "green": (0, 192, 0), "blue": (0, 0, 255)}
PROMPT = "What is the solid background color of this image? Answer with one color word."
REPOSITORY = Path(__file__).resolve().parents[3]


def endpoint_identity(value: str) -> str:
    """Record endpoint provenance without permitting embedded credentials or redirect parameters."""
    url = httpx.URL(value)
    if (
        url.scheme not in ("http", "https")
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise ValueError("Invalid benchmark endpoint")
    return str(url).rstrip("/")


def archive_sources(directory: Path) -> dict[str, str]:
    """Retain actual source bytes, so later edits cannot make an evidence hash unrecoverable."""
    hashes: dict[str, str] = {}
    for relative in (
        "src/finserve/multimodal/vision_benchmark.py",
        "src/finserve/multimodal/images.py",
        "src/finserve/multimodal/preprocess_http.py",
        "src/finserve/engines/vision_openai.py",
        "src/finserve/engines/openai_adapter.py",
        "src/finserve/contracts/vision.py",
        "src/finserve/contracts/inference.py",
        "src/finserve/gateway/vision.py",
    ):
        raw = (REPOSITORY / relative).read_bytes()
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        (directory / ("source__" + relative.replace("/", "__"))).write_bytes(raw)
    return hashes


def repository_state() -> dict[str, object]:
    """Record actual checkout state; archived source remains authoritative for a dirty workspace."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    return {
        "git_revision": head,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def color_png(color: tuple[int, int, int], side: int) -> bytes:
    """Generate public lossless fixtures with fixed answers and no external media rights."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        """Use genuine PNG encoding with explicit CRCs so fixture bytes are reproducible."""
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    raw = (b"\x00" + bytes(color) * side) * side
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


async def measured_inference(
    engine: VisionEngine, image: PreparedImage, origin: float
) -> dict[str, Any]:
    """Measure first text and terminal time without inferring encoder/decoder durations."""
    request = VisionRequest(
        prompt=PROMPT,
        image_png_base64=base64.b64encode(image.png).decode(),
        max_tokens=16,
        timeout_seconds=60.0,
    )
    started, first, output, count, reason = time.perf_counter(), None, "", 0, None
    end_to_end_first: float | None = None
    iterator = engine.stream(request, image)
    try:
        async for token in iterator:
            if token.text and first is None:
                observed = time.perf_counter()
                first = observed - started
                end_to_end_first = observed - origin
            output += token.text
            count += token.generated_tokens
            reason = token.finish_reason or reason
    finally:
        await iterator.aclose()
    if reason is None or first is None:
        raise RuntimeError("Vision result omitted text or terminal usage")
    return {
        "output": output,
        "completion_tokens": count,
        "finish_reason": reason,
        "coupled_vllm_client_ttft_seconds": first,
        "end_to_end_ttft_seconds": end_to_end_first,
        "coupled_vllm_client_duration_seconds": time.perf_counter() - started,
    }


async def sample(
    engine: VisionEngine,
    preprocessor: PreprocessorClient,
    raw: bytes,
    mode: str,
    expected_hash: str,
    expected_color: str,
) -> dict[str, Any]:
    """Include every stage in end-to-end time and preserve errors as run evidence upstream."""
    started = time.perf_counter()
    if mode == "local":
        stage_started = time.perf_counter()
        image = await asyncio.to_thread(prepare_png, raw)
        stage_seconds = time.perf_counter() - stage_started
        worker_seconds, transfer_bytes = None, 0
    else:
        image, stage_seconds, worker_seconds = await preprocessor.prepare(raw)
        transfer_bytes = len(raw) + len(image.png)
    if image.sha256 != expected_hash:
        raise RuntimeError("Stage changed canonical image bytes")
    result = await measured_inference(engine, image, started)
    output = str(result["output"]).strip().lower().rstrip(".! ")
    return {
        "mode": mode,
        "canonical_sha256": image.sha256,
        "input_bytes": len(raw),
        "canonical_bytes": len(image.png),
        "stage_body_transfer_bytes": transfer_bytes,
        "stage_client_seconds": stage_seconds,
        "worker_local_seconds": worker_seconds,
        "stage_nonworker_residual_seconds": None
        if worker_seconds is None
        else stage_seconds - worker_seconds,
        "end_to_end_seconds": time.perf_counter() - started,
        "expected_color": expected_color,
        "exact_color_correct": output == expected_color,
        **result,
    }


async def run(
    output: Path,
    base_url: str,
    preprocessor_url: str,
    repetitions: int,
    profile: Path = REPOSITORY / "benchmarks/configs/vllm-qwen2-vl-local-8gb.json",
) -> Path:
    """Freeze workload and identities before warmup, retaining partial/interrupted evidence."""
    if not 1 <= repetitions <= 20:
        raise ValueError("Repetitions must be between 1 and 20")
    engine_url, stage_url = endpoint_identity(base_url), endpoint_identity(preprocessor_url)
    directory = prepare_output(output)
    manifest: dict[str, Any] = {
        "run_id": str(uuid4()),
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "model": VISION_MODEL,
        "model_revision": VISION_REVISION,
        "engine_url": engine_url,
        "preprocessor_url": stage_url,
        "profile_attestation": "declared configuration; weights require separate startup evidence",
        "prompt": PROMPT,
        "repetitions": repetitions,
        "sizes": [256, 448],
        "colors": list(COLORS),
        "generation": {"max_tokens": 16, "temperature": 0, "stream": True},
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("httpx", "pillow")},
        "stage_scope": "CPU PNG HTTP hop; vision encoder and decoder remain coupled in vLLM",
        "timing_scope": (
            "client monotonic durations; remote residual includes transport/queue/serialization, "
            "not one-way network"
        ),
        "workload_scope": "six synthetic solid-color images, not a general VLM quality benchmark",
        "source_sha256": {},
    }
    write_json(directory / "manifest.json", manifest)
    engine: OpenAIVisionEngine | None = None
    preprocessor: PreprocessorClient | None = None
    records: list[dict[str, Any]] = []
    try:
        # Bounded local provenance writes precede measurement and do not enter timed samples.
        manifest["source_sha256"] = archive_sources(directory)
        manifest["repository"] = repository_state()
        write_json(directory / "manifest.json", manifest)
        profile_value = json.loads(await asyncio.to_thread(profile.read_bytes))
        write_json(directory / "engine-profile.json", profile_value)
        if (
            profile_value.get("model") != VISION_MODEL
            or profile_value.get("model_revision") != VISION_REVISION
        ):
            raise ValueError("Engine profile model/revision mismatch")
        engine = OpenAIVisionEngine(base_url, api_key=os.environ.get("FINSERVE_ENGINE_API_KEY"))
        preprocessor = PreprocessorClient(preprocessor_url, os.environ["FINSERVE_PREPROCESSOR_KEY"])
        for side in (256, 448):
            for color, channels in COLORS.items():
                raw = color_png(channels, side)
                image = prepare_png(raw)
                stem = f"{color}-{side}"
                (directory / f"{stem}-input.png").write_bytes(raw)
                (directory / f"{stem}-canonical.png").write_bytes(image.png)
                # Save every warmup separately; its cold execution is excluded from measured pairs.
                warmup: dict[str, Any] = {
                    "phase": "warmup",
                    "fixture": stem,
                    "mode": "local",
                    "canonical_sha256": image.sha256,
                }
                try:
                    warmup.update(
                        await sample(engine, preprocessor, raw, "local", image.sha256, color)
                    )
                    warmup["status"] = "succeeded"
                except BaseException as exc:
                    warmup.update(status="failed", error_type=type(exc).__name__)
                    raise
                finally:
                    write_json(directory / f"{stem}-warmup.json", warmup)
                for repetition in range(repetitions):
                    modes = ("local", "http") if repetition % 2 == 0 else ("http", "local")
                    for mode in modes:
                        record = {"fixture": stem, "repetition": repetition, "mode": mode}
                        try:
                            record.update(
                                await sample(engine, preprocessor, raw, mode, image.sha256, color)
                            )
                            record["status"] = "succeeded"
                        except BaseException as exc:
                            record.update(status="failed", error_type=type(exc).__name__)
                            raise
                        finally:
                            records.append(record)
                            with (directory / "samples.jsonl").open(
                                "a", encoding="utf-8"
                            ) as target:
                                target.write(json.dumps(record, allow_nan=False) + "\n")
        pairs = [
            (left, right)
            for index, left in enumerate(records)
            for right in records[index + 1 :]
            if left["fixture"] == right["fixture"] and left["repetition"] == right["repetition"]
        ]
        summary = {
            "measured_requests": len(records),
            "pairs": len(pairs),
            "exact_output_pair_parity": sum(a["output"] == b["output"] for a, b in pairs)
            / len(pairs),
            "exact_color_accuracy": sum(row["exact_color_correct"] for row in records)
            / len(records),
            "quality_gate_passed": all(row["exact_color_correct"] for row in records)
            and all(a["output"] == b["output"] for a, b in pairs),
            "modes": {
                mode: {
                    field: statistics.median(row[field] for row in records if row["mode"] == mode)
                    for field in (
                        "stage_client_seconds",
                        "end_to_end_seconds",
                        "end_to_end_ttft_seconds",
                    )
                }
                for mode in ("local", "http")
            },
        }
        write_json(directory / "summary.json", summary)
        manifest["status"] = "succeeded"
    except BaseException as exc:
        manifest["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        manifest["error_type"] = type(exc).__name__
        raise
    finally:
        try:
            if engine is not None:
                await engine.close()
        finally:
            try:
                if preprocessor is not None:
                    await preprocessor.close()
            finally:
                manifest["finished_at"] = datetime.now(UTC).isoformat()
                manifest["artifact_sha256"] = {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in directory.iterdir()
                    if path.is_file() and path.name != "manifest.json"
                }
                write_json(directory / "manifest.json", manifest)
    return directory


def main() -> None:
    """Require an explicit evidence directory and real endpoints; no hidden GPU provisioning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8020/v1")
    parser.add_argument("--preprocessor-url", default="http://127.0.0.1:8021")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPOSITORY / "benchmarks/configs/vllm-qwen2-vl-local-8gb.json",
    )
    args = parser.parse_args()
    asyncio.run(
        run(args.output, args.base_url, args.preprocessor_url, args.repetitions, args.profile)
    )


if __name__ == "__main__":
    main()
