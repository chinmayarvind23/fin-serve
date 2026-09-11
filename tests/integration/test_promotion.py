"""Artifact-driven promotion tests use clearly synthetic raw timing fixtures."""

import json
from pathlib import Path

import pytest

from finserve.benchmark.metrics import RequestRecord
from finserve.benchmark.runner import RunConfig, build_summary, write_json
from finserve.benchmark.workload import default_workload
from finserve.contracts.deployment import Revision
from finserve.evaluation.quality import default_suite
from finserve.reliability.promotion import (
    PerformanceSample,
    PromotionPolicy,
    QualityEvidence,
    evaluate_promotion,
    performance_failures,
)


def candidate_revision() -> Revision:
    """Image/config digests are synthetic test identities, not claims of a deployed image."""
    return Revision(
        revision_id="candidate",
        model_revision="weights-v1",
        tokenizer_revision="tokenizer-v1",
        source_revision="source-v1",
        image_digest="sha256:" + "a" * 64,
        config_digest="b" * 64,
        engine="fixture",
        engine_config="fixture-config",
    )


def artifact(path: Path, name: str, duration: float, first_content: float) -> None:
    """Create internally consistent raw/summary fixtures so gates exercise loader verification."""
    path.mkdir()
    revision = candidate_revision()
    workload = default_workload(32)
    config = RunConfig(
        requests=4,
        concurrency=4,
        warmup=0,
        hardware="fixture CPU",
        revision=revision.source_revision,
        model_revision=revision.model_revision,
        tokenizer_revision=revision.tokenizer_revision,
        engine=revision.engine,
        engine_config=revision.engine_config,
        image_digest=revision.image_digest,
        config_digest=revision.config_digest,
    )
    records = [
        RequestRecord(
            logical_id=index,
            case_id=item.case_id,
            family=item.family,
            scheduled_s=100,
            send_s=100,
            first_content_s=100 + first_content,
            complete_s=100 + duration * 0.9,
            success=True,
            status_code=200,
            generated_tokens=8,
            output="fixture output",
        )
        for index, item in enumerate(workload.items)
    ]
    write_json(
        path / "manifest.json",
        {
            "run_id": name,
            "status": "completed",
            "workload": workload.model_dump(),
            "workload_hash": workload.digest(),
            "configuration": config.model_dump(),
            "measured_started_s": 100,
            "measured_finished_s": 100 + duration,
            "measured_seconds": duration,
            "scope": "synthetic unit fixture",
        },
    )
    (path / "requests.jsonl").write_text("\n".join(row.model_dump_json() for row in records) + "\n")
    write_json(path / "summary.json", build_summary(records, duration, workload))


def quality() -> QualityEvidence:
    """Bind independently known answers to the exact synthetic baseline/candidate run identities."""
    suite = default_suite()
    answers = {case.case_id: case.expected for case in suite.cases}
    return QualityEvidence(
        suite_hash=suite.digest(),
        evaluator_version=suite.evaluator_version,
        baseline_run_id="baseline-fixture",
        candidate_run_id="candidate-fixture",
        reference_model_revision="weights-v1",
        candidate_model_revision="weights-v1",
        reference=answers,
        candidate=answers.copy(),
    )


def test_faster_correct_candidate_passes_both_gates(tmp_path: Path) -> None:
    """A speed improvement can pass only with raw evidence and recomputed known-answer quality."""
    artifact(tmp_path / "baseline", "baseline-fixture", 4, 0.5)
    artifact(tmp_path / "candidate", "candidate-fixture", 2, 0.2)
    decision = evaluate_promotion(
        tmp_path / "baseline",
        tmp_path / "candidate",
        quality(),
        default_suite(),
        PromotionPolicy(),
        candidate_revision(),
    )
    assert decision.approved and decision.evidence_digest
    assert decision.candidate_accuracy == 1 and decision.quality_parity == 1


def test_faster_but_wrong_candidate_cannot_promote(tmp_path: Path) -> None:
    """Even twice the throughput cannot override one wrong deterministic financial answer."""
    artifact(tmp_path / "baseline", "baseline-fixture", 4, 0.5)
    artifact(tmp_path / "candidate", "candidate-fixture", 2, 0.2)
    evidence = quality()
    evidence.candidate["margin"] = "0.20"
    decision = evaluate_promotion(
        tmp_path / "baseline",
        tmp_path / "candidate",
        evidence,
        default_suite(),
        PromotionPolicy(),
        candidate_revision(),
    )
    assert not decision.approved and "quality_gate_failed" in decision.rejection_reasons


@pytest.mark.parametrize(
    "failure", ["missing", "wrong_suite", "wrong_model", "wrong_image", "edited_summary"]
)
def test_missing_or_mismatched_evidence_fails_closed(tmp_path: Path, failure: str) -> None:
    """Absent evidence and convenient relabeling cannot be transformed into an approval."""
    artifact(tmp_path / "baseline", "baseline-fixture", 4, 0.5)
    artifact(tmp_path / "candidate", "candidate-fixture", 2, 0.2)
    evidence, revision = quality(), candidate_revision()
    if failure == "missing":
        (tmp_path / "candidate/requests.jsonl").unlink()
    elif failure == "wrong_suite":
        evidence = evidence.model_copy(update={"suite_hash": "changed"})
    elif failure == "wrong_model":
        evidence = evidence.model_copy(update={"candidate_model_revision": "different"})
    elif failure == "wrong_image":
        revision = revision.model_copy(update={"image_digest": "sha256:" + "d" * 64})
    else:
        path = tmp_path / "candidate/summary.json"
        values = json.loads(path.read_text())
        values["requests_per_second"] = 9999
        write_json(path, values)
    decision = evaluate_promotion(
        tmp_path / "baseline",
        tmp_path / "candidate",
        evidence,
        default_suite(),
        PromotionPolicy(),
        revision,
    )
    assert not decision.approved and decision.evidence_digest is None


def test_each_performance_regression_is_independent() -> None:
    """Higher request throughput does not excuse lower token throughput, latency or success."""
    baseline = PerformanceSample(
        requests_per_second=10,
        tokens_per_second=100,
        client_ttft_median_s=0.1,
        e2e_p95_s=1,
        success_rate=1,
    )
    candidate = PerformanceSample(
        requests_per_second=9,
        tokens_per_second=90,
        client_ttft_median_s=0.2,
        e2e_p95_s=2,
        success_rate=0.9,
    )
    assert len(performance_failures(baseline, candidate, PromotionPolicy())) == 5
    with pytest.raises(ValueError):
        performance_failures(
            baseline.model_copy(update={"e2e_p95_s": 1e308}),
            candidate,
            PromotionPolicy(maximum_e2e_p95_ratio=1e308),
        )
