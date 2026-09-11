"""Fixed-shape contracts run everywhere; numerical tests require the optional CPU ML stack."""

import asyncio
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from finserve.multimodal.benchmark import run_visual_benchmark
from finserve.multimodal.jax_generator import (
    IMAGE_SIZE,
    PALETTE,
    VISUAL_TOKENS,
    JAXVisualReference,
    VisualOutput,
    VisualRequest,
    output_png,
)


def solid(value: int) -> VisualRequest:
    """Make deterministic conditioning images without any image-file or numerical dependency."""
    return VisualRequest(
        images=[[[(value, value, value) for _ in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)]]
    )


@pytest.fixture(scope="module")
def model() -> JAXVisualReference:
    """Reuse parameters/JIT caches and keep optional JAX out of normal contract-only CI."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    return JAXVisualReference(seed=17)


def test_fixed_shape_and_channel_contracts() -> None:
    """Reject variable shapes, oversized batches, and lossy channel coercion before allocation."""
    request = solid(127)
    assert len(request.images[0]) == IMAGE_SIZE
    for images in ([], request.images * 9, [request.images[0][:-1]]):
        with pytest.raises(ValidationError):
            VisualRequest(images=images)
    for channel in (-1, 256, 1.5, True):
        data = request.model_dump()
        data["images"][0][0][0] = (channel, 0, 0)
        with pytest.raises(ValidationError):
            VisualRequest.model_validate(data)


def test_eager_jit_exact_token_parity(model: JAXVisualReference) -> None:
    """Compilation must preserve every generated palette ID on the same conditioned workload."""
    request = VisualRequest(images=[solid(0).images[0], solid(255).images[0]])
    eager = model.generate(request, compiled=False)
    compiled = model.generate(request)
    assert compiled == eager
    assert len(compiled.token_ids) == 2
    assert all(len(tokens) == VISUAL_TOKENS for tokens in compiled.token_ids)
    assert all(0 <= token < len(PALETTE) for tokens in compiled.token_ids for token in tokens)


def test_image_conditioning_and_previous_tokens_change_logits(model: JAXVisualReference) -> None:
    """Prove image pixels and prior output tokens actually influence the conditional model."""
    black, white = solid(0), solid(255)
    assert model.prefix_logits(black, []) != model.prefix_logits(white, [])
    assert model.prefix_logits(black, [0]) != model.prefix_logits(black, [1])
    assert model.generate(black).token_ids != model.generate(white).token_ids


def test_batch_isolation_and_deterministic_model_identity(model: JAXVisualReference) -> None:
    """Batch neighbors must not alter an image's sequence; seeded model identity must be stable."""
    dark, light = solid(20), solid(220)
    combined = model.generate(VisualRequest(images=[dark.images[0], light.images[0]]))
    assert combined.token_ids[0] == model.generate(dark).token_ids[0]
    assert combined.token_ids[1] == model.generate(light).token_ids[0]
    second = JAXVisualReference(seed=17)
    assert second.model_sha256 == model.model_sha256
    assert second.generate(dark) == model.generate(dark)
    assert JAXVisualReference(seed=18).model_sha256 != model.model_sha256


@pytest.mark.parametrize("prefix", [[-1], [len(PALETTE)], [0] * VISUAL_TOKENS])
def test_invalid_prefix_is_rejected(model: JAXVisualReference, prefix: list[int]) -> None:
    """Internal inspection helpers still honor palette and sequence bounds."""
    with pytest.raises(ValueError, match="prefix"):
        model.prefix_logits(solid(0), prefix)


def test_rendered_png_matches_generated_palette_tokens(model: JAXVisualReference) -> None:
    """Verify artifact pixels instead of treating a successfully written PNG as correctness."""
    image_module = pytest.importorskip("PIL.Image")
    output = model.generate(solid(64))
    png = output_png(output, scale=2)
    image = image_module.open(io.BytesIO(png))
    assert image.size == (16, 16)
    for index, token in enumerate(output.token_ids[0]):
        x, y = index % IMAGE_SIZE, index // IMAGE_SIZE
        assert image.getpixel((x * 2, y * 2)) == PALETTE[token]
    assert output_png(output, scale=2) == png


def test_invalid_seed_and_render_bounds() -> None:
    """Reject invalid configuration without requiring optional numerical dependencies."""
    with pytest.raises(ValueError, match="seed"):
        JAXVisualReference(seed=-1)
    output = VisualOutput(token_ids=[[0] * VISUAL_TOKENS], seed=17, model_sha256="a" * 64)
    for index, scale in ((1, 1), (0, 0), (0, 65)):
        with pytest.raises(ValueError, match="bounds"):
            output_png(output, index=index, scale=scale)


def test_benchmark_preserves_raw_samples_and_artifacts(
    model: JAXVisualReference, tmp_path: Path
) -> None:
    """Check compile separation, authoritative sample counts, and refusal to overwrite a run."""
    directory = tmp_path / "visual-run"
    manifest = run_visual_benchmark(directory, repetitions=1, batch_sizes=(1,))
    assert manifest["status"] == "completed"
    assert manifest["model_sha256"] == model.model_sha256
    records = [
        json.loads(line) for line in (directory / "raw-samples.jsonl").read_text().splitlines()
    ]
    assert {record["mode"] for record in records} == {"python_loop", "compiled"}
    assert len(records) == 2 and all(record["exact_parity"] for record in records)
    assert manifest["results"][0]["compile_and_first_execution_ns"] > 0
    assert (directory / "batch-1-input-0.png").is_file()
    assert (directory / "batch-1-output-0.png").is_file()
    with pytest.raises(FileExistsError):
        run_visual_benchmark(directory, repetitions=1)


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError])
def test_failed_benchmark_retains_failure_manifest(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    """Missing optional runtime metadata is still a recorded failed experiment, not an empty run."""
    directory = tmp_path / "failed-run"
    with patch("finserve.multimodal.benchmark.importlib.metadata.version", side_effect=failure):
        with pytest.raises(failure):
            run_visual_benchmark(directory, repetitions=1)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == ("failed" if issubclass(failure, Exception) else "interrupted")
    assert manifest["failure_type"] == failure.__name__
    assert manifest["run_id"] and manifest["started_at"] <= manifest["finished_at"]


@pytest.mark.parametrize("warmed", [False, True])
def test_parity_failure_retains_actual_tokens(tmp_path: Path, warmed: bool) -> None:
    """Keep baseline and rejected IDs for both initial and warmed numerical disagreements."""
    directory = tmp_path / "parity-run"
    baseline = VisualOutput(token_ids=[[0] * VISUAL_TOKENS], seed=17, model_sha256="a" * 64)
    rejected = baseline.model_copy(update={"token_ids": [[1] * VISUAL_TOKENS]})
    sequence = [(baseline, 1), (baseline if warmed else rejected, 1), (rejected, 1)]
    with (
        patch("finserve.multimodal.benchmark.importlib.metadata.version", return_value="test"),
        patch("finserve.multimodal.benchmark.JAXVisualReference") as constructor,
        patch("finserve.multimodal.benchmark.measured_generation", side_effect=sequence),
    ):
        constructor.return_value.model_sha256 = "a" * 64
        with pytest.raises(RuntimeError, match="parity"):
            run_visual_benchmark(directory, repetitions=1, batch_sizes=(1,))
    candidate = "python_loop-0-rejected" if warmed else "compiled-first"
    assert json.loads((directory / "batch-1-baseline.json").read_text())["token_ids"][0][0] == 0
    assert json.loads((directory / f"batch-1-{candidate}.json").read_text())["token_ids"][0][0] == 1
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert f"batch-1-{candidate}.json" in manifest["artifact_sha256"]


def test_invalid_benchmark_bounds(tmp_path: Path) -> None:
    """Invalid work budgets fail before creating any evidence directory."""
    with pytest.raises(ValueError):
        run_visual_benchmark(tmp_path / "none", repetitions=0)
    with pytest.raises(ValueError):
        run_visual_benchmark(tmp_path / "none", batch_sizes=(1, 1))
    with pytest.raises(ValueError, match="outside"):
        run_visual_benchmark(Path(__file__).resolve().parents[2] / "forbidden-visual-evidence")
    assert not (tmp_path / "none").exists()
