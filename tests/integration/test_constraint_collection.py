"""Shared constraints bind native wire requests, collector evidence and release comparisons."""

import json
import time
from pathlib import Path

import httpx
import pytest

from finserve.benchmark.constraint_mapping import (
    RequestConstraintBinding,
    RequestConstraintMap,
    prompt_digest,
)
from finserve.benchmark.gpu import TelemetrySample
from finserve.benchmark.runner import (
    RunConfig,
    run_benchmark,
    validate_comparison,
    validate_evidence,
)
from finserve.benchmark.workload import WorkItem, Workload
from finserve.contracts.deployment import Revision
from finserve.contracts.output_constraint import OutputConstraint
from finserve.contracts.performance import PerformanceCollectionSpec
from finserve.contracts.producer import QualityCollectionSpec
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.evaluation.quality import GoldenCase, GoldenSuite, evaluate_quality
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.performance_collection import collect_performance
from finserve.registry.produced_release import ReleaseCohort
from finserve.registry.quality_collection import collect_quality, load_quality_collection
from finserve.reliability.promotion import QualityEvidence, check_quality_identity


def specifications(
    api: str = "completions", transport: str = "native_vllm"
) -> tuple[QualityCollectionSpec, PerformanceCollectionSpec]:
    """Synthetic runtime identities test contracts; no real model, image or GPU is represented."""
    prompts = ("First: return an integer.", "Second: return a decimal.", "Write freely.")
    constraints = (OutputConstraint(kind="integer"), OutputConstraint(kind="decimal"), None)
    mapping = RequestConstraintMap(
        entries=tuple(
            RequestConstraintBinding(prompt_sha256=prompt_digest(prompt), constraint=constraint)
            for prompt, constraint in zip(prompts, constraints, strict=True)
        )
    )
    profile = ServingProfileV1(
        engine="vllm",
        engine_version="0.29.0",
        engine_parameters_json=VLLMParameters(
            structured_output_backend="xgrammar"
        ).model_dump_json(),
        model_revision="a" * 40,
        tokenizer_revision="a" * 40,
        model_manifest_sha256="b" * 64,
        tokenizer_manifest_sha256="b" * 64,
        base_url="http://fixture/v1",
        served_model="fixture",
    )
    revision = Revision(
        revision_id="fixture",
        model_revision=profile.model_revision,
        tokenizer_revision=profile.tokenizer_revision,
        source_revision="c" * 40,
        image_digest="sha256:" + "d" * 64,
        config_digest=profile.digest(),
        engine=profile.engine,
        engine_config=profile.engine_parameters_json,
    )
    config = RunConfig.model_validate(
        dict(
            requests=3,
            warmup=0,
            concurrency=1,
            hardware="fixture-cpu",
            model="fixture",
            revision=revision.source_revision,
            model_revision=revision.model_revision,
            tokenizer_revision=revision.tokenizer_revision,
            engine=revision.engine,
            engine_config=revision.engine_config,
            image_digest=revision.image_digest,
            config_digest=revision.config_digest,
            request_api=api,
            chat_template_sha256="e" * 64 if api == "chat" else None,
            output_constraints=mapping,
            constraint_transport=transport,
        )
    )
    suite = GoldenSuite(
        cases=(
            GoldenCase(
                case_id="quality_integer", family="unrelated", prompt=prompts[0], expected="5"
            ),
            GoldenCase(
                case_id="quality_decimal", family="other", prompt=prompts[1], expected="0.2"
            ),
        )
    )
    quality = QualityCollectionSpec(
        collection_id="quality",
        profile=profile,
        revision=revision,
        suite=suite,
        configuration=config,
        max_tokens=4,
    )
    workload = Workload(
        suite_id="fixture",
        version=1,
        items=tuple(
            WorkItem(
                case_id=f"performance_{index}", family="performance", prompt=prompt, max_tokens=4
            )
            for index, prompt in enumerate(prompts)
        ),
    )
    performance = PerformanceCollectionSpec(
        collection_id="performance",
        collector_revision="c" * 40,
        profile=profile,
        revision=revision,
        workload=workload,
        configuration=config,
        timeout_seconds=15,
    )
    return quality, performance


def response(request: httpx.Request, text: str = "7") -> httpx.Response:
    """Return an intentionally wrong answer with valid shape; shape checks must not invent truth."""
    content = (
        {"delta": {"content": text}}
        if "messages" in json.loads(request.content)
        else {"text": text}
    )
    return httpx.Response(
        200,
        content="data: "
        + json.dumps(
            {
                "choices": [{**content, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1},
            }
        )
        + "\n\ndata: [DONE]\n\n",
    )


@pytest.mark.parametrize(
    ("api", "transport"),
    [("completions", "native_vllm"), ("chat", "native_vllm"), ("chat", "finserve")],
)
async def test_both_collectors_send_identical_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api: str, transport: str
) -> None:
    """Both collector loops share wire requests despite different reporting case IDs."""
    quality, performance = specifications(api, transport)
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture serialized HTTP requests, not a mocked mapper result."""
        seen.append(json.loads(request.content))
        return response(request)

    def client(_: RunConfig) -> httpx.AsyncClient:
        """The performance worker owns a separate pool using the same local fixture transport."""
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def clean_git(arguments: list[str], **kwargs: object) -> str:
        """Declare synthetic source provenance explicitly; no real clean-image claim is made."""
        return "" if arguments[1] == "status" else "c" * 40 + "\n"

    monkeypatch.setattr("finserve.benchmark.experiment.benchmark_client", client)
    monkeypatch.setattr("finserve.benchmark.experiment.subprocess.check_output", clean_git)
    monkeypatch.setattr(
        "finserve.benchmark.experiment.collect",
        lambda: TelemetrySample(
            epoch_s=time.time(), collection_seconds=0, devices=[], error="FixtureNoGPU"
        ),
    )
    frozen_suite = quality.suite.model_dump_json()
    async with client(quality.configuration) as connection:
        result = await collect_quality(connection, quality, tmp_path / "quality")
    await collect_performance(performance, tmp_path / "performance")
    assert seen[:2] == seen[2:4]
    wire_key = "structured_outputs" if transport == "native_vllm" else "output_constraint"
    assert wire_key in seen[0] and wire_key in seen[1] and wire_key not in seen[4]
    assert result.successful == 2
    grade = evaluate_quality(
        quality.suite, {case.case_id: case.expected for case in quality.suite.cases}, result.outputs
    )
    assert grade["candidate_accuracy"] == 0
    assert quality.suite.model_dump_json() == frozen_suite
    assert load_quality_collection(tmp_path / "quality") == (quality, result)
    evidence = validate_evidence(tmp_path / "performance" / "run")
    assert result.request_mapping_sha256 == evidence.configuration.request_mapping_digest()
    ReleaseCohort(
        performance_stage="performance",
        quality_stage="quality",
        performance=performance,
        quality=quality,
    )


async def test_missing_binding_rejects_before_any_request(tmp_path: Path) -> None:
    """A partially mapped population cannot collect only mapped cases or use literal fallback."""
    quality, performance = specifications()
    mapping = quality.configuration.output_constraints
    assert mapping is not None
    partial = RequestConstraintMap(entries=(mapping.entries[0],))
    config = quality.configuration.model_copy(update={"output_constraints": partial})
    for specification in (quality, performance):
        with pytest.raises(ValueError, match="missing"):
            type(specification).model_validate_json(
                specification.model_copy(update={"configuration": config}).model_dump_json()
            )
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ValueError, match="missing"):
            await run_benchmark(
                client, performance.endpoint(), performance.workload, config, tmp_path / "run"
            )
    assert not (tmp_path / "run").exists()


async def test_invalid_raw_shape_remains_failure_and_cannot_be_relabelled(tmp_path: Path) -> None:
    """Native output bypassing the gateway receives the same final shape check in collection."""
    quality, performance = specifications()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: response(request, "0.20"))
    ) as client:
        result = await collect_quality(client, quality, tmp_path / "quality")
        await run_benchmark(
            client,
            performance.endpoint(),
            performance.workload,
            performance.configuration,
            tmp_path / "run",
        )
    assert result.successful == 0 and result.recorded == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "quality" / "requests.jsonl").read_text().splitlines()
    ]
    assert all(row["response"]["output"] == "0.20" for row in rows)
    assert all(not row["response"]["success"] for row in rows)
    validate_evidence(tmp_path / "run")
    path = tmp_path / "run" / "requests.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0].update(success=True, error=None)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="frozen constraint"):
        validate_evidence(tmp_path / "run")


async def test_comparison_and_completion_quality_require_constraint_identity(
    tmp_path: Path,
) -> None:
    """Completion cohorts require explicit quality identity and unchanged constraint mappings."""
    quality, performance = specifications()
    left, right = tmp_path / "left", tmp_path / "right"
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        for path in (left, right):
            await run_benchmark(
                client,
                performance.endpoint(),
                performance.workload,
                performance.configuration,
                path,
            )
    validate_comparison(left, right)
    first, second = validate_evidence(left), validate_evidence(right)
    evidence = QualityEvidence(
        suite_hash=quality.suite.digest(),
        evaluator_version=quality.suite.evaluator_version,
        baseline_run_id=json.loads((left / "manifest.json").read_text())["run_id"],
        candidate_run_id=json.loads((right / "manifest.json").read_text())["run_id"],
        reference_model_revision=first.configuration.model_revision,
        candidate_model_revision=second.configuration.model_revision,
        reference={},
        candidate={},
    )
    with pytest.raises(ValueError, match="mapping"):
        check_quality_identity(evidence, quality.suite, left, right, first, second)
    check_quality_identity(
        evidence.model_copy(
            update={"request_mapping_sha256": performance.configuration.request_mapping_digest()}
        ),
        quality.suite,
        left,
        right,
        first,
        second,
    )
    document = json.loads((right / "manifest.json").read_text())
    document["configuration"]["constraint_transport"] = "finserve"
    (right / "manifest.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="constraint_transport"):
        validate_comparison(left, right)
    changed = performance.model_copy(
        update={
            "configuration": performance.configuration.model_copy(
                update={"constraint_transport": "finserve"}
            )
        }
    )
    with pytest.raises(ValueError, match="mapping"):
        ReleaseCohort(
            performance_stage="p", quality_stage="q", performance=changed, quality=quality
        )
