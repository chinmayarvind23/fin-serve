"""Recomputed catalog observations reject mismatched clocks, models, answers and mutations."""

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from finserve.benchmark.gpu import TelemetrySample, aggregate
from finserve.benchmark.runner import RunConfig, run_benchmark
from finserve.benchmark.workload import default_workload
from finserve.evaluation.quality import default_suite, evaluate_quality
from finserve.registry.annotations import AnnotationStore, evaluator_hashes, main
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.metadata import Registry, RegistryConflict


def write(path: Path, value: object) -> bytes:
    """The fixture retains exact serialized bytes so hash relationships are exercised."""
    data = json.dumps(value).encode()
    path.write_bytes(data)
    return data


async def prepare(tmp_path: Path) -> tuple[Registry, AnnotationStore, str, Path, Path]:
    """Produce actual benchmark records and an independent deterministic quality evaluation."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"text":"answer"}]}\n\n'
                    'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        )
    ) as client:
        await run_benchmark(
            client,
            "http://fixture/completions",
            default_workload(),
            RunConfig(
                requests=2,
                warmup=0,
                concurrency=1,
                hardware="CPU",
                revision="source",
                model_revision="model",
                tokenizer_revision="tokenizer",
                engine="fixture",
                engine_config="transport-only",
            ),
            tmp_path / "run",
        )
    registry = Registry(f"sqlite:///{tmp_path / 'catalog.db'}")
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    bundle = registry.register_run(tmp_path / "run", artifacts)
    catalog = AnnotationStore(registry, artifacts)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_bytes())
    write(
        tmp_path / "environment.json",
        {
            "configuration": manifest["configuration"],
            "clock_epoch_anchor_s": 1800000000,
            "git_sha": "source",
            "git_status": [],
            "clock_monotonic_anchor_s": manifest["measured_started_s"],
        },
    )
    write(tmp_path / "experiment-status.json", {"status": "completed", "clock_drift_seconds": 0})
    sample = TelemetrySample(
        epoch_s=1800000000, collection_seconds=0, devices=[], error="fixture-no-gpu"
    )
    (tmp_path / "gpu.jsonl").write_text(sample.model_dump_json() + "\n")
    write(
        tmp_path / "gpu-summary.json",
        aggregate([sample], 1800000000, 1800000000 + manifest["measured_seconds"]),
    )
    quality = tmp_path / "quality"
    quality.mkdir()
    suite = default_suite()
    answers = {case.case_id: "incorrect" for case in suite.cases}
    digest = hashlib.sha256(write(quality / "answers.json", answers)).hexdigest()
    (quality / "requests.jsonl").write_text(
        "\n".join(json.dumps({"case_id": key, "output": value}) for key, value in answers.items())
    )
    write(
        quality / "manifest.json",
        {
            **manifest["configuration"],
            "git_sha": "source",
            "git_status": [],
            "suite": suite.model_dump(),
            "suite_hash": suite.digest(),
            "status": "completed",
            "answers_sha256": digest,
            "evaluator_sha256": sorted(evaluator_hashes())[0],
            "run_id": "quality-fixture",
        },
    )
    write(quality / "quality.json", evaluate_quality(suite, answers, answers))
    return registry, catalog, bundle.run_id, tmp_path, quality


async def test_missing_gpu_is_null_and_failed_quality_is_retained(tmp_path: Path) -> None:
    """No device sample means unknown utilization; a rejected quality gate stays visible."""
    registry, catalog, run_id, experiment, quality = await prepare(tmp_path)
    try:
        assert catalog.read(run_id, "gpu") is None
        gpu = catalog.register_gpu(run_id, experiment)
        assert (
            json.loads(catalog.artifacts.get(gpu.report))["average_gpu_utilization_percent"] is None
        )
        assert catalog.register_gpu(run_id, experiment) == gpu
        result = catalog.register_quality(run_id, quality, quality)
        report = json.loads(catalog.artifacts.get(result.report))
        assert report["passed"] is False and report["candidate_accuracy"] == 0
        assert len(result.inputs) == 8 and catalog.read(run_id, "quality") == result
        with pytest.raises(RegistryConflict):
            catalog.save(run_id, "gpu", {"changed": True}, [])
    finally:
        registry.close()


@pytest.mark.parametrize(
    "target,key,value",
    [
        ("gpu-summary.json", "coverage", 1),
        ("gpu-summary.json", "window_start_epoch_s", 1800000000.05),
        ("experiment-status.json", "clock_drift_seconds", 1),
        ("environment.json", "configuration", {}),
        ("environment.json", "git_sha", "wrong-source"),
        ("environment.json", "git_status", ["dirty"]),
        ("quality/manifest.json", "model_revision", "wrong-model"),
        ("quality/manifest.json", "git_status", ["dirty"]),
        ("quality/manifest.json", "suite_hash", "wrong-suite"),
        ("quality/manifest.json", "answers_sha256", "wrong-answers"),
        ("quality/manifest.json", "evaluator_sha256", "wrong-evaluator"),
        ("quality/quality.json", "candidate_accuracy", 1),
    ],
)
async def test_mismatched_inputs_are_rejected(
    tmp_path: Path, target: str, key: str, value: Any
) -> None:
    """Reject reports whose identity, raw measurement or recomputed score differs."""
    registry, catalog, run_id, experiment, quality = await prepare(tmp_path)
    path = tmp_path / target
    data = json.loads(path.read_bytes())
    data[key] = value
    write(path, data)
    try:
        with pytest.raises(ValueError):
            if target.startswith("quality/"):
                catalog.register_quality(run_id, quality, quality)
            else:
                catalog.register_gpu(run_id, experiment)
        assert catalog.read(run_id, "gpu") is None and catalog.read(run_id, "quality") is None
    finally:
        registry.close()


async def test_reference_embedded_suite_cannot_hide_behind_declared_hash(tmp_path: Path) -> None:
    """A correctly linked reference manifest still must contain the frozen suite it names."""
    registry, catalog, run_id, _, quality = await prepare(tmp_path)
    reference = tmp_path / "baseline"
    reference.mkdir()
    for path in quality.iterdir():
        (reference / path.name).write_bytes(path.read_bytes())
    ref = json.loads((reference / "manifest.json").read_bytes())
    ref["suite"]["cases"][0]["expected"] = "changed"
    digest = hashlib.sha256(write(reference / "manifest.json", ref)).hexdigest()
    candidate = json.loads((quality / "manifest.json").read_bytes())
    candidate["reference_manifest_sha256"] = digest
    write(quality / "manifest.json", candidate)
    try:
        with pytest.raises(ValueError, match="embedded suite"):
            catalog.register_quality(run_id, quality, reference)
    finally:
        registry.close()


async def test_cli_imports_and_retries_without_rewriting_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise the operator command with completed recorder outputs and rejected quality."""
    registry, catalog, run_id, experiment, quality = await prepare(tmp_path)
    arguments = [
        "annotations",
        "--database-url",
        f"sqlite:///{tmp_path / 'catalog.db'}",
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--experiment",
        str(experiment),
        "--quality",
        str(quality),
        "--reference-quality",
        str(quality),
    ]
    try:
        monkeypatch.setattr("sys.argv", arguments)
        main()
        first = catalog.read(run_id, "quality")
        assert first is not None
        main()
        assert catalog.read(run_id, "quality") == first
        assert len(capsys.readouterr().out.splitlines()) == 2
        monkeypatch.setattr("sys.argv", arguments[:-2])
        with pytest.raises(SystemExit):
            main()
    finally:
        registry.close()
