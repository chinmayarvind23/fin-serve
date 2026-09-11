"""Freeze release inputs before collection and register only reconstructed managed receipts."""

import json
from typing import Self

from pydantic import Field, model_validator

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.performance import PerformanceCollectionSpec
from finserve.contracts.producer import QualityCollectionSpec
from finserve.registry.artifacts import ArtifactRef
from finserve.registry.lifecycle import LifecycleSpec
from finserve.registry.managed_quality import load_managed_quality
from finserve.registry.performance_stages import PerformanceReceipt, load_performance_receipt
from finserve.registry.producer_stages import ProducerStages, StageState
from finserve.registry.producer_tasks import declare_input, load_quality_receipt
from finserve.registry.release_gate import GateRequest, freeze_gate_request
from finserve.reliability.promotion import PromotionPolicy, QualityEvidence


class ReleaseCohort(ImmutableModel):
    """A cohort names both collector inputs before either result is observed."""

    performance_stage: str = Field(min_length=1, max_length=128)
    quality_stage: str = Field(min_length=1, max_length=128)
    performance: PerformanceCollectionSpec
    quality: QualityCollectionSpec

    @model_validator(mode="after")
    def same_requests(self) -> Self:
        """Quality must exercise the same runtime and wire mapping as its performance cohort."""
        if (
            self.performance.profile != self.quality.profile
            or self.performance.revision != self.quality.revision
            or self.performance.configuration.request_mapping_digest()
            != self.quality.configuration.request_mapping_digest()
        ):
            raise ValueError("release cohort runtime or request mapping differs")
        return self


class ProducedReleasePlan(ImmutableModel):
    """Policy, quality suite and collectors are immutable before any collection attempt."""

    job_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    deployment_id: str = Field(min_length=1, max_length=128)
    expected_generation: int = Field(ge=0, strict=True)
    baseline: ReleaseCohort
    candidate: ReleaseCohort
    policy: PromotionPolicy

    @model_validator(mode="after")
    def comparable_pair(self) -> Self:
        """Reject asymmetric quality experiments and aliasing of stage identities."""
        left, right = self.baseline, self.candidate
        ids = [
            left.performance_stage,
            left.quality_stage,
            right.performance_stage,
            right.quality_stage,
            self.job_id + ":release-plan",
        ]
        if len(set(ids)) != len(ids):
            raise ValueError("release stage identities must be distinct")
        if (
            left.quality.suite.digest() != right.quality.suite.digest()
            or left.quality.max_tokens != right.quality.max_tokens
            or left.quality.configuration.request_mapping_digest()
            != right.quality.configuration.request_mapping_digest()
            or left.performance.workload.digest() != right.performance.workload.digest()
            or left.performance.revision.revision_id == right.performance.revision.revision_id
        ):
            raise ValueError("release cohorts must name comparable distinct revisions")
        return self


def freeze_produced_release(journal: ProducerStages, plan: ProducedReleasePlan) -> str:
    """Freeze metadata only; retries cannot alter policy or backdate existing collection work."""
    plan = ProducedReleasePlan.model_validate_json(plan.model_dump_json())
    stage_id = plan.job_id + ":release-plan"
    state = declare_input(journal, stage_id, plan.model_dump(mode="json"))
    if state.status == "completed":
        if state.output != state.input:
            raise ValueError("release plan receipt differs from frozen input")
        return stage_id
    if state.status == "planned":
        state = journal.start(stage_id)
    # This stage owns no external action: a lost metadata acknowledgment can finish the same CAS.
    assert state.attempt_id is not None
    journal.finish(stage_id, state.attempt_id, state.input)
    return stage_id


def collection_state(journal: ProducerStages, stage_id: str, frozen_at: float) -> StageState:
    """All attempts, including failed predecessors, must follow the frozen release plan."""
    state = journal.state(stage_id)
    if state.status != "completed" or state.output is None:
        raise ValueError("release requires completed collection stages")
    if any(
        item.started_at is not None and item.started_at < frozen_at
        for item in journal.history(stage_id)
    ):
        raise ValueError("collection began before release plan was frozen")
    return state


def cohort_evidence(
    journal: ProducerStages, cohort: ReleaseCohort, frozen_at: float
) -> tuple[PerformanceReceipt, dict[str, str]]:
    """Verify stage inputs and raw receipts, then require one shared runtime launch."""
    performance_state = collection_state(journal, cohort.performance_stage, frozen_at)
    quality_state = collection_state(journal, cohort.quality_stage, frozen_at)
    assert performance_state.output is not None and quality_state.output is not None
    performance = load_performance_receipt(journal, performance_state.output)
    quality = load_managed_quality(journal, quality_state.output)
    if min(performance.before.observed_at, quality.before.observed_at) < frozen_at:
        raise ValueError("collection observations precede frozen release plan")
    recorded, result = load_quality_receipt(journal, quality.quality)
    if (
        performance.specification.sha256 != cohort.performance.digest()
        or recorded.digest() != cohort.quality.digest()
        or performance.launch != quality.launch
        or performance.runtime != quality.runtime
    ):
        raise ValueError("release collection receipts differ from frozen cohort")
    for state, kind, specification in (
        (
            performance_state,
            "performance-collection-v1",
            json.loads(cohort.performance.canonical()),
        ),
        (quality_state, "managed-quality-v1", json.loads(cohort.quality.canonical())),
    ):
        frozen = json.loads(journal.artifacts.get(state.input))
        if (
            frozen.get("kind") != kind
            or frozen.get("specification") != specification
            or frozen.get("launch") != performance.launch.model_dump()
        ):
            raise ValueError("release receipt differs from collection stage input")
    return performance, result.outputs


def register_produced_release(journal: ProducerStages, plan_stage_id: str) -> str:
    """Register canonical gate inputs from four verified producer stages, never caller outputs."""
    state = journal.state(plan_stage_id)
    if state.status != "completed" or state.output != state.input or state.finished_at is None:
        raise ValueError("release plan is not frozen")
    plan = ProducedReleasePlan.model_validate_json(journal.artifacts.get(state.input))
    if plan_stage_id != plan.job_id + ":release-plan":
        raise ValueError("release plan identity differs from stage")
    baseline, reference = cohort_evidence(journal, plan.baseline, state.finished_at)
    candidate, outputs = cohort_evidence(journal, plan.candidate, state.finished_at)
    suite = plan.baseline.quality.suite
    quality = QualityEvidence(
        suite_hash=suite.digest(),
        evaluator_version=suite.evaluator_version,
        baseline_run_id=baseline.run.run_id,
        candidate_run_id=candidate.run.run_id,
        reference_model_revision=baseline.runtime.revision.model_revision,
        candidate_model_revision=candidate.runtime.revision.model_revision,
        reference=reference,
        candidate=outputs,
        request_mapping_sha256=plan.baseline.quality.configuration.request_mapping_digest(),
    )

    def put(value: str) -> ArtifactRef:
        """CAS stores the exact reconstructed bytes used by the shared release gate."""
        return journal.artifacts.put(value.encode())

    specification = LifecycleSpec(
        job_id=plan.job_id,
        deployment_id=plan.deployment_id,
        expected_generation=plan.expected_generation,
        expected_revision=baseline.runtime.revision.revision_id,
        baseline_run_id=baseline.run.run_id,
        candidate_run_id=candidate.run.run_id,
        quality=put(quality.model_dump_json()),
        suite=put(suite.model_dump_json()),
        policy=plan.policy,
        target=candidate.runtime.revision,
        gate_mode="canonical-profile-v1",
    )
    journal.registry.create_job(plan.job_id, specification.canonical())
    freeze_gate_request(
        journal.registry,
        GateRequest(
            job_id=plan.job_id,
            baseline_profile=put(plan.baseline.performance.profile.canonical()),
            candidate_profile=put(plan.candidate.performance.profile.canonical()),
        ),
    )
    return plan.job_id
