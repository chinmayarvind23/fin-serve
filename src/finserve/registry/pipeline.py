"""Server-configured Airflow stage entry points, with immutable job IDs as the only task handoff."""

import asyncio
import importlib
import os
from pathlib import Path
from typing import Self, cast

from pydantic import model_validator

from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.lifecycle import LifecycleService, LifecycleSpec
from finserve.registry.metadata import Registry
from finserve.registry.produced_release import register_produced_release
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.release_gate import GateRequest, evaluate_gate, freeze_gate_request
from finserve.reliability.promotion import PromotionPolicy
from finserve.reliability.rollback import DeploymentAdapter


class PipelineRequest(ImmutableModel):
    """This server-owned input identifies existing verified build/run artifacts, not client
    assertions.
    """

    job_id: str
    deployment_id: str
    expected_generation: int
    baseline_directory: Path
    candidate_directory: Path
    quality_file: Path
    suite_file: Path
    baseline_revision: Revision
    target_revision: Revision
    policy: PromotionPolicy
    baseline_profile_file: Path | None = None
    candidate_profile_file: Path | None = None

    @model_validator(mode="after")
    def complete_profile_pair(self) -> Self:
        """Reject partial canonical requests before any durable run or job registration."""
        if (self.baseline_profile_file is None) != (self.candidate_profile_file is None):
            raise ValueError("both canonical profile files are required")
        return self


def runtime() -> tuple[Registry, LocalArtifactStore]:
    """Resolve database and artifact endpoints only at task runtime, never during DAG parsing."""
    return Registry(os.environ["FINSERVE_REGISTRY_URL"]), LocalArtifactStore(
        Path(os.environ["FINSERVE_ARTIFACT_ROOT"])
    )


def register_produced_stage(plan_stage_id: str) -> str:
    """Pass a frozen plan ID from producer tasks into the existing canonical release stages."""
    registry, artifacts = runtime()
    try:
        return register_produced_release(ProducerStages(registry, artifacts), plan_stage_id)
    finally:
        registry.close()


def register_stage() -> str:
    """Register frozen runs and quality inputs, then persist the full immutable lifecycle
    specification.
    """
    request = PipelineRequest.model_validate_json(
        Path(os.environ["FINSERVE_PIPELINE_REQUEST"]).read_text()
    )
    registry, artifacts = runtime()
    try:
        baseline = registry.register_run(
            request.baseline_directory, artifacts, request.baseline_revision
        )
        candidate = registry.register_run(
            request.candidate_directory, artifacts, request.target_revision
        )
        spec = LifecycleSpec(
            job_id=request.job_id,
            deployment_id=request.deployment_id,
            expected_revision=request.baseline_revision.revision_id,
            expected_generation=request.expected_generation,
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            quality=artifacts.put(request.quality_file.read_bytes()),
            suite=artifacts.put(request.suite_file.read_bytes()),
            policy=request.policy,
            target=request.target_revision,
            gate_mode=(
                "canonical-profile-v1"
                if request.baseline_profile_file is not None
                else "legacy-drill"
            ),
        )
        registry.create_job(spec.job_id, spec.canonical())
        if request.baseline_profile_file is not None and request.candidate_profile_file is not None:
            freeze_gate_request(
                registry,
                GateRequest(
                    job_id=spec.job_id,
                    baseline_profile=artifacts.put(request.baseline_profile_file.read_bytes()),
                    candidate_profile=artifacts.put(request.candidate_profile_file.read_bytes()),
                ),
            )
        return spec.job_id
    finally:
        registry.close()


def evaluate_stage(job_id: str) -> str:
    """Legacy evidence-drill entry point; the release DAG uses evaluate_release_stage instead."""
    registry, artifacts = runtime()
    try:
        specification = LifecycleSpec.model_validate_json(registry.specification(job_id))
        state = asyncio.run(
            LifecycleService(registry, artifacts).run(specification, evaluate_only=True)
        )
        if state.status not in {"evaluated", "promoted"} or state.last_error is not None:
            raise RuntimeError("lifecycle evaluation did not pass")
        return job_id
    finally:
        registry.close()


def evaluate_release_stage(job_id: str) -> str:
    """Use the same persisted profile-aware outcome and rejection behavior as the CI command."""
    registry, artifacts = runtime()
    try:
        outcome = evaluate_gate(registry, artifacts, job_id)
        if outcome.status != "approved":
            raise RuntimeError("canonical release evidence did not pass")
        return job_id
    finally:
        registry.close()


def deploy_release_stage(job_id: str) -> str:
    """Revalidate release evidence immediately before invoking any deployment callback."""
    return deploy_stage(job_id, require_profile=True)


def deploy_stage(job_id: str, *, require_profile: bool = False) -> str:
    """Load a server-configured adapter and rely on durable lifecycle reconciliation across
    retries.
    """
    registry, artifacts = runtime()
    try:
        specification = LifecycleSpec.model_validate_json(registry.specification(job_id))
        has_profile = specification.gate_mode == "canonical-profile-v1"
        # Existing drill entry points cannot bypass a canonical job's release gate.
        if require_profile or has_profile:
            outcome = evaluate_gate(registry, artifacts, job_id)
            if outcome.status != "approved":
                raise RuntimeError("canonical release evidence did not pass")
        module_name, factory_name = os.environ["FINSERVE_DEPLOYMENT_ADAPTER"].split(":", maxsplit=1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        adapter = cast(DeploymentAdapter, factory())
        state = asyncio.run(LifecycleService(registry, artifacts).run(specification, adapter))
        if state.status != "promoted":
            raise RuntimeError("lifecycle deployment is not verified healthy")
        return job_id
    finally:
        registry.close()
