"""Reproducible CPU visual-reference experiment with explicit compilation and synchronization."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from finserve.multimodal.jax_generator import (
    IMAGE_SIZE,
    VISUAL_TOKENS,
    JAXVisualReference,
    RGBImage,
    VisualOutput,
    VisualRequest,
    output_png,
)


def conditioning_image(index: int = 0) -> RGBImage:
    """Generate a public deterministic color gradient so no private media enters the experiment."""
    return [
        [
            ((x * 32 + index * 19) % 256, (y * 32 + index * 37) % 256, ((x + y) * 16) % 256)
            for x in range(IMAGE_SIZE)
        ]
        for y in range(IMAGE_SIZE)
    ]


def input_png(image: RGBImage) -> bytes:
    """Render the exact conditioning pixels at the same nearest-neighbor scale as model output."""
    import io

    image_module = importlib.import_module("PIL.Image")
    raster = image_module.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    raster.putdata([pixel for row in image for pixel in row])
    raster = raster.resize((128, 128), image_module.Resampling.NEAREST)
    target = io.BytesIO()
    raster.save(target, format="PNG")
    return target.getvalue()


def measured_generation(
    model: JAXVisualReference, request: VisualRequest, compiled: bool
) -> tuple[VisualOutput, int]:
    """Time through synchronized output conversion; dispatch-only JAX timing is invalid."""
    started = time.perf_counter_ns()
    output = model.generate(request, compiled=compiled)
    return output, time.perf_counter_ns() - started


def _write_json(path: Path, value: object) -> None:
    """Write complete records; the enclosing new run directory preserves earlier run history."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _batch_experiment(
    model: JAXVisualReference, batch_size: int, repetitions: int, directory: Path
) -> dict[str, Any]:
    """Separate compile-first execution and warmed paths while retaining every measured sample."""
    request = VisualRequest(images=[conditioning_image(index) for index in range(batch_size)])
    request_json = request.model_dump_json()
    (directory / f"batch-{batch_size}-request.json").write_text(request_json, encoding="utf-8")
    baseline, _ = measured_generation(model, request, compiled=False)
    first_compiled, first_ns = measured_generation(model, request, compiled=True)
    _write_json(directory / f"batch-{batch_size}-baseline.json", baseline.model_dump())
    _write_json(directory / f"batch-{batch_size}-compiled-first.json", first_compiled.model_dump())
    _write_json(
        directory / f"batch-{batch_size}-compilation.json",
        {"elapsed_ns": first_ns, "exact_token_parity": first_compiled == baseline},
    )
    if first_compiled != baseline:
        raise RuntimeError("Visual JIT output failed exact token parity")
    samples: dict[str, list[int]] = {"python_loop": [], "compiled": []}
    for mode, durations in samples.items():
        for repetition in range(repetitions):
            output, elapsed = measured_generation(model, request, compiled=mode == "compiled")
            record = {
                "batch_size": batch_size,
                "mode": mode,
                "repetition": repetition,
                "elapsed_ns": elapsed,
                "visual_tokens": batch_size * VISUAL_TOKENS,
                "exact_parity": output == baseline,
            }
            with (directory / "raw-samples.jsonl").open("a", encoding="utf-8") as target:
                target.write(json.dumps(record, sort_keys=True) + "\n")
            if output != baseline:
                _write_json(
                    directory / f"batch-{batch_size}-{mode}-{repetition}-rejected.json",
                    output.model_dump(),
                )
                raise RuntimeError("Warmed visual output failed exact token parity")
            durations.append(elapsed)
    for index, image in enumerate(request.images):
        (directory / f"batch-{batch_size}-input-{index}.png").write_bytes(input_png(image))
        (directory / f"batch-{batch_size}-output-{index}.png").write_bytes(
            output_png(baseline, index)
        )
    _write_json(directory / f"batch-{batch_size}-output.json", baseline.model_dump())
    return {
        "batch_size": batch_size,
        "input_sha256": hashlib.sha256(request_json.encode()).hexdigest(),
        "compile_and_first_execution_ns": first_ns,
        "warmup_python_loop_calls": 1,
        "warmed_samples_ns": samples,
        "warmed_mean_ns": {mode: statistics.mean(values) for mode, values in samples.items()},
        "exact_token_parity": True,
    }


def run_visual_benchmark(
    directory: Path, repetitions: int = 5, batch_sizes: tuple[int, ...] = (1, 4)
) -> dict[str, Any]:
    """Create fresh evidence and retain running/failed manifests alongside successful runs."""
    if (
        not 1 <= repetitions <= 100
        or not batch_sizes
        or any(not 1 <= size <= 8 for size in batch_sizes)
    ):
        raise ValueError("visual benchmark repetitions and batch sizes are outside limits")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("visual benchmark batch sizes must be unique")
    directory = directory.resolve()
    repository = Path(__file__).resolve().parents[3]
    if directory == repository or repository in directory.parents:
        raise ValueError("benchmark evidence must be outside the source repository")
    directory.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": str(uuid4()),
        "started_at": datetime.now(UTC).isoformat(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "running",
        "scope": "untrained_cpu_visual_reference",
        "timing": "perf_counter_ns through block_until_ready and output conversion",
        "device": "cpu",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repetitions": repetitions,
        "batch_sizes": batch_sizes,
        "results": [],
    }
    _write_json(directory / "manifest.json", manifest)
    try:
        manifest["versions"] = {
            name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax")
        }
        started = time.perf_counter_ns()
        model = JAXVisualReference(seed=17)
        manifest["model_initialization_ns"] = time.perf_counter_ns() - started
        manifest["model_sha256"] = model.model_sha256
        source = Path(__file__).with_name("jax_generator.py").read_bytes()
        manifest["implementation_sha256"] = hashlib.sha256(source).hexdigest()
        for batch_size in batch_sizes:
            manifest["results"].append(_batch_experiment(model, batch_size, repetitions, directory))
            _write_json(directory / "manifest.json", manifest)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        manifest["failure_type"] = type(exc).__name__
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest["artifact_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        _write_json(directory / "manifest.json", manifest)
    return manifest


def main() -> None:
    """Expose a CPU-only reproducible experiment command whose output directory must be new."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    manifest = run_visual_benchmark(args.output_dir, repetitions=args.repetitions)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
