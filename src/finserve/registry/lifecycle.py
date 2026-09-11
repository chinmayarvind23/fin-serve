"""Offline evidence registration, dual-gate evaluation and bounded deployment callback
orchestration.
"""

import asyncio
import tempfile
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.evaluation.quality import GoldenSuite
from finserve.registry.artifacts import ArtifactRef, ArtifactStore
from finserve.registry.metadata import (
    LifecycleState,
    Registry,
    RegistryConflict,
    RunBundle,
    canonical_json,
)
from finserve.reliability.promotion import PromotionPolicy, QualityEvidence, evaluate_promotion
from finserve.reliability.rollback import ApplyRequest, DeploymentAdapter


class LifecycleSpec(ImmutableModel):
    """A persisted specification binds immutable runs, quality bytes, target and activation
    generation.
    """

    job_id: str = Field(min_length=1, max_length=128)
    deployment_id: str = Field(min_length=1, max_length=128)
    expected_revision: str
    expected_generation: int = Field(ge=0)
    baseline_run_id: str
    candidate_run_id: str
    quality: ArtifactRef
    suite: ArtifactRef
    policy: PromotionPolicy = Field(default_factory=PromotionPolicy)
    target: Revision

    def canonical(self) -> str:
        """Normalize typed defaults so task JSON round-trips preserve specification identity."""
        normalized = LifecycleSpec.model_validate_json(self.model_dump_json())
        return canonical_json(normalized.model_dump())


def materialize(bundle: RunBundle, store: ArtifactStore, directory: Path) -> None:
    """Only verified immutable references become temporary files consumed by the existing loader."""
    directory.mkdir()
    for filename, reference in (
        ("manifest.json", bundle.manifest),
        ("requests.jsonl", bundle.requests),
        ("summary.json", bundle.summary),
    ):
        (directory / filename).write_bytes(store.get(reference))


class LifecycleService:
    """Lifecycle state outlives Airflow task attempts; a passed client flag is never an input."""

    def __init__(
        self, registry: Registry, artifacts: ArtifactStore, timeout_seconds: float = 30
    ) -> None:
        """The deployment lease bounds apply plus reconciliation probes without looping
        indefinitely.
        """
        if not 0 < timeout_seconds <= 3600:
            raise ValueError("timeout must be finite and positive")
        self.registry, self.artifacts, self.timeout_seconds = registry, artifacts, timeout_seconds

    def _decision(self, specification: LifecycleSpec) -> str:
        """Materialize verified bytes and recompute all gates before any new deployment attempt."""
        baseline = self.registry.run(specification.baseline_run_id)
        candidate = self.registry.run(specification.candidate_run_id)
        if baseline.revision_id != specification.expected_revision:
            raise ValueError("baseline bundle does not describe the expected active revision")
        if candidate.revision_id != specification.target.revision_id:
            raise ValueError("candidate bundle is not bound to target deployment revision")
        if self.registry.revision(specification.target.revision_id) != specification.target:
            raise ValueError("target revision differs from registered identity")
        quality = QualityEvidence.model_validate_json(self.artifacts.get(specification.quality))
        suite = GoldenSuite.model_validate_json(self.artifacts.get(specification.suite))
        with tempfile.TemporaryDirectory(prefix="finserve-gate-") as temporary:
            directory = Path(temporary)
            materialize(baseline, self.artifacts, directory / "baseline")
            materialize(candidate, self.artifacts, directory / "candidate")
            decision = evaluate_promotion(
                directory / "baseline",
                directory / "candidate",
                quality,
                suite,
                specification.policy,
                specification.target,
            )
        return self.registry.record_decision(decision, specification.candidate_run_id)

    async def run(
        self,
        specification: LifecycleSpec,
        adapter: DeploymentAdapter | None = None,
        evaluate_only: bool = False,
    ) -> LifecycleState:
        """Resume durable state, preserving ambiguous actions and rejected decisions."""
        specification = LifecycleSpec.model_validate_json(specification.model_dump_json())
        state = self.registry.create_job(specification.job_id, specification.canonical())
        if state.status in {"promoted", "rejected"}:
            return state
        owner = str(uuid4())
        state = self.registry.claim(
            state.job_id, specification.deployment_id, owner, 3 * self.timeout_seconds + 30
        )
        try:
            decision_id = self._decision(specification)
            decision = self.registry.decision(decision_id)
            if not decision.approved:
                if state.status not in {"registered", "evaluated"}:
                    raise RegistryConflict("previously approved deployment evidence changed")
                return self.registry.advance(
                    state, owner, status="rejected", decision_digest=decision_id, last_error=None
                )
            if state.decision_digest is not None and state.decision_digest != decision_id:
                raise RegistryConflict("recorded decision changed")
            if state.status == "registered":
                state = self.registry.advance(
                    state, owner, status="evaluated", decision_digest=decision_id, last_error=None
                )
            elif state.last_error is not None:
                state = self.registry.advance(state, owner, last_error=None)
            if evaluate_only:
                return state
            if adapter is None:
                raise ValueError("deployment adapter is required for activation")
            return await self._deploy(specification, state, owner, adapter)
        except asyncio.CancelledError:
            current = self.registry.job(state.job_id)
            self.registry.advance(current, owner, last_error="CancelledError")
            raise
        except Exception as error:
            current = self.registry.job(state.job_id)
            return self.registry.advance(current, owner, last_error=type(error).__name__)
        finally:
            self.registry.release(state.job_id, owner)

    async def _probe(self, specification: LifecycleSpec, adapter: DeploymentAdapter) -> bool:
        """Require exact traffic-route health after any apply acknowledgement."""
        async with asyncio.timeout(self.timeout_seconds):
            health = await adapter.health(specification.deployment_id)
        return health.verifies(specification.target)

    async def _deploy(
        self,
        specification: LifecycleSpec,
        state: LifecycleState,
        owner: str,
        adapter: DeploymentAdapter,
    ) -> LifecycleState:
        """Persist action intent first and only replay uncertainty with a same-key idempotency
        guarantee.
        """
        if state.status in {"deploying", "verifying", "needs_reconciliation"}:
            if await self._probe(specification, adapter):
                return self.registry.advance(state, owner, status="promoted", last_error=None)
            if state.status == "verifying" or adapter.supports_idempotency is not True:
                status = "verifying" if state.status == "verifying" else "needs_reconciliation"
                return self.registry.advance(
                    state, owner, status=status, last_error="health_not_verified"
                )
        state = self.registry.advance(
            state, owner, status="deploying", apply_attempts=state.apply_attempts + 1
        )
        request = ApplyRequest(
            deployment_id=specification.deployment_id,
            expected_revision=specification.expected_revision,
            expected_generation=specification.expected_generation,
            target=specification.target,
            idempotency_key="lifecycle-" + specification.job_id,
        )
        async with asyncio.timeout(self.timeout_seconds):
            await adapter.apply(request)
        state = self.registry.advance(state, owner, status="verifying")
        if await self._probe(specification, adapter):
            return self.registry.advance(state, owner, status="promoted", last_error=None)
        return self.registry.advance(state, owner, last_error="health_not_verified")
