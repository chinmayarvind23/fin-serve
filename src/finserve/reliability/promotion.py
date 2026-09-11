"""Recompute performance and quality evidence before allowing candidate activation."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from finserve.benchmark.runner import ComparisonInput, validate_comparison, validate_evidence
from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.evaluation.quality import GoldenSuite, QualityConfig, evaluate_quality


class PromotionPolicy(ImmutableModel):
    """Version thresholds before benchmarking; quality hard failures override every speed gain."""

    version: str = "performance-quality-v1"
    minimum_requests_per_second_ratio: float = Field(default=1, gt=0)
    minimum_tokens_per_second_ratio: float = Field(default=1, gt=0)
    maximum_client_ttft_ratio: float = Field(default=1.05, gt=0)
    maximum_e2e_p95_ratio: float = Field(default=1.05, gt=0)
    minimum_success_rate: float = Field(default=0.9995, ge=0, le=1)
    quality: QualityConfig = Field(default_factory=QualityConfig)


class QualityEvidence(ImmutableModel):
    """Raw outputs are bound to exact run and model identities, not a caller's passed boolean."""

    suite_hash: str
    evaluator_version: str
    baseline_run_id: str
    candidate_run_id: str
    reference_model_revision: str
    candidate_model_revision: str
    reference: dict[str, str]
    candidate: dict[str, str]
    request_mapping_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def preserve_legacy_encoding(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep historical quality bytes stable; new chat evidence explicitly binds its mapping."""
        result: dict[str, Any] = handler(self)
        if self.request_mapping_sha256 is None:
            result.pop("request_mapping_sha256", None)
        return result


class PerformanceSample(BaseModel):
    """Unknown or zero baseline denominators fail validation rather than producing infinity."""

    model_config = ConfigDict(allow_inf_nan=False)
    requests_per_second: float = Field(gt=0)
    tokens_per_second: float = Field(gt=0)
    client_ttft_median_s: float = Field(gt=0)
    e2e_p95_s: float = Field(gt=0)
    success_rate: float = Field(ge=0, le=1)


class PromotionDecision(ImmutableModel):
    """An evidence fingerprint and reason list make every rejection reviewable and persistent."""

    candidate_revision: str
    candidate_digest: str
    policy_version: str
    evidence_digest: str | None
    rejection_reasons: tuple[str, ...]
    quality_parity: float | None = Field(default=None, ge=0, le=1)
    candidate_accuracy: float | None = Field(default=None, ge=0, le=1)

    @property
    def approved(self) -> bool:
        """An approval cannot exist without verified evidence identity and zero failed gates."""
        return self.evidence_digest is not None and not self.rejection_reasons


def performance_failures(
    baseline: PerformanceSample,
    candidate: PerformanceSample,
    policy: PromotionPolicy,
) -> list[str]:
    """Use multiplication against baseline thresholds, avoiding unstable division ratios."""
    limits = (
        baseline.requests_per_second * policy.minimum_requests_per_second_ratio,
        baseline.tokens_per_second * policy.minimum_tokens_per_second_ratio,
        baseline.client_ttft_median_s * policy.maximum_client_ttft_ratio,
        baseline.e2e_p95_s * policy.maximum_e2e_p95_ratio,
    )
    if not all(math.isfinite(limit) for limit in limits):
        raise ValueError("performance threshold overflow")
    checks = (
        (
            candidate.requests_per_second
            >= baseline.requests_per_second * policy.minimum_requests_per_second_ratio,
            "request_throughput_regression",
        ),
        (
            candidate.tokens_per_second
            >= baseline.tokens_per_second * policy.minimum_tokens_per_second_ratio,
            "token_throughput_regression",
        ),
        (
            candidate.client_ttft_median_s
            <= baseline.client_ttft_median_s * policy.maximum_client_ttft_ratio,
            "client_ttft_regression",
        ),
        (
            candidate.e2e_p95_s <= baseline.e2e_p95_s * policy.maximum_e2e_p95_ratio,
            "e2e_latency_regression",
        ),
        (candidate.success_rate >= policy.minimum_success_rate, "success_rate_below_floor"),
    )
    return [reason for passed, reason in checks if not passed]


def check_quality_identity(
    evidence: QualityEvidence,
    suite: GoldenSuite,
    baseline: Path,
    candidate: Path,
    left: ComparisonInput,
    right: ComparisonInput,
) -> None:
    """Reject an easier substituted suite or outputs labeled for a different model/run."""
    baseline_manifest: dict[str, object] = json.loads((baseline / "manifest.json").read_text())
    candidate_manifest: dict[str, object] = json.loads((candidate / "manifest.json").read_text())
    checks = (
        evidence.suite_hash == suite.digest(),
        evidence.evaluator_version == suite.evaluator_version,
        evidence.baseline_run_id == baseline_manifest.get("run_id"),
        evidence.candidate_run_id == candidate_manifest.get("run_id"),
        evidence.reference_model_revision == left.configuration.model_revision,
        evidence.candidate_model_revision == right.configuration.model_revision,
    )
    if not all(checks):
        raise ValueError("quality identity mismatch")
    if (
        left.configuration.request_api == "chat"
        or right.configuration.request_api == "chat"
        or left.configuration.output_constraints is not None
        or right.configuration.output_constraints is not None
        or evidence.request_mapping_sha256 is not None
    ):
        if not (
            evidence.request_mapping_sha256
            == left.configuration.request_mapping_digest()
            == right.configuration.request_mapping_digest()
        ):
            raise ValueError("quality request mapping differs from measured cohort")
        for configuration in (left.configuration, right.configuration):
            if configuration.output_constraints is not None:
                configuration.output_constraints.require_prompts(
                    case.prompt for case in suite.cases
                )


def verify_candidate_identity(candidate: ComparisonInput, revision: Revision) -> None:
    """The revision being deployed must be the model/tokenizer/source/engine actually measured."""
    for field, attribute in (
        ("model_revision", "model_revision"),
        ("tokenizer_revision", "tokenizer_revision"),
        ("revision", "source_revision"),
        ("engine", "engine"),
        ("engine_config", "engine_config"),
        ("image_digest", "image_digest"),
        ("config_digest", "config_digest"),
    ):
        if getattr(candidate.configuration, field) != getattr(revision, attribute):
            raise ValueError("candidate revision differs from benchmark evidence")


def evidence_digest(
    baseline: Path,
    candidate: Path,
    quality: QualityEvidence,
    suite: GoldenSuite,
    policy: PromotionPolicy,
    revision: Revision,
) -> str:
    """Bind the decision to exact raw files, quality outputs, policy and immutable target."""
    digest = hashlib.sha256()
    for directory in (baseline, candidate):
        for filename in ("manifest.json", "requests.jsonl", "summary.json"):
            digest.update(hashlib.sha256((directory / filename).read_bytes()).digest())
    for payload in (
        quality.model_dump_json(),
        suite.model_dump_json(),
        policy.model_dump_json(),
        revision.model_dump_json(),
    ):
        digest.update(hashlib.sha256(payload.encode()).digest())
    return digest.hexdigest()


def evaluate_promotion(
    baseline: Path,
    candidate: Path,
    quality: QualityEvidence,
    suite: GoldenSuite,
    policy: PromotionPolicy,
    revision: Revision,
) -> PromotionDecision:
    """Fail closed on missing/invalid artifacts; recompute both gates from their raw evidence."""
    common = {
        "candidate_revision": revision.revision_id,
        "candidate_digest": revision.digest(),
        "policy_version": policy.version,
    }
    try:
        before_digest = evidence_digest(baseline, candidate, quality, suite, policy, revision)
        validate_comparison(baseline, candidate)
        left, right = validate_evidence(baseline), validate_evidence(candidate)
        verify_candidate_identity(right, revision)
        check_quality_identity(quality, suite, baseline, candidate, left, right)
        before = PerformanceSample.model_validate_json((baseline / "summary.json").read_text())
        after = PerformanceSample.model_validate_json((candidate / "summary.json").read_text())
        failures = performance_failures(before, after, policy)
        result = evaluate_quality(suite, quality.reference, quality.candidate, policy.quality)
        if result["passed"] is not True:
            failures.append("quality_gate_failed")
        after_digest = evidence_digest(baseline, candidate, quality, suite, policy, revision)
        if before_digest != after_digest:
            raise ValueError("evidence changed during evaluation")
        return PromotionDecision.model_validate(
            {
                **common,
                "evidence_digest": after_digest,
                "rejection_reasons": tuple(failures),
                "quality_parity": result["parity"],
                "candidate_accuracy": result["candidate_accuracy"],
            }
        )
    except (OSError, ValueError, TypeError, KeyError):
        return PromotionDecision.model_validate(
            {
                **common,
                "evidence_digest": None,
                "rejection_reasons": ("missing_or_invalid_evidence",),
            }
        )
