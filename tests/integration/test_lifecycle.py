"""Local SQLAlchemy/CAS/gate/deployment-callback integration, using declared synthetic evidence."""

import asyncio
import sys
import types
from pathlib import Path

import pytest
from test_promotion import artifact, candidate_revision, quality

from finserve.contracts.deployment import HealthObservation, Revision
from finserve.evaluation.quality import default_suite
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.lifecycle import LifecycleService, LifecycleSpec
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.pipeline import PipelineRequest, deploy_stage, evaluate_stage, register_stage
from finserve.reliability.promotion import PromotionDecision
from finserve.reliability.rollback import ApplyRequest


class CandidateAdapter:
    """Injected CPU fixture has real state transitions but is not a cloud deployment adapter."""

    supports_idempotency = True

    def __init__(self, active: Revision) -> None:
        """Track external state separately from registry acknowledgement and simulate uncertain
        writes.
        """
        self.active = active
        self.calls: list[ApplyRequest] = []
        self.fail = False
        self.ready = True
        self.block = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def apply(self, request: ApplyRequest) -> None:
        """A failed transport may leave the action outcome unknown to the lifecycle controller."""
        self.calls.append(request)
        self.started.set()
        if self.block:
            await self.release.wait()
        if self.fail:
            raise OSError("fixture deployment uncertainty")
        self.active = request.target

    async def health(self, deployment_id: str) -> HealthObservation:
        """Return the revision actually active in this injected fixture."""
        return HealthObservation(
            revision_id=self.active.revision_id,
            revision_digest=self.active.digest(),
            ready=self.ready,
            smoke_passed=self.ready,
        )


def registered(
    tmp_path: Path, wrong: bool = False
) -> tuple[Registry, LocalArtifactStore, LifecycleSpec, CandidateAdapter]:
    """Register two complete runs and raw quality outputs before invoking any lifecycle gate."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.db"))
    store = LocalArtifactStore(tmp_path / "objects")
    artifact(tmp_path / "baseline", "baseline-fixture", 4, 0.5)
    artifact(tmp_path / "candidate", "candidate-fixture", 2, 0.2)
    candidate = candidate_revision()
    baseline = candidate.model_copy(update={"revision_id": "baseline"})
    registry.register_run(tmp_path / "baseline", store, baseline)
    bundle = registry.register_run(tmp_path / "candidate", store, candidate)
    assert registry.register_run(tmp_path / "candidate", store, candidate) == bundle
    evidence = quality()
    if wrong:
        evidence.candidate["margin"] = "0.20"
    specification = LifecycleSpec(
        job_id="job-1",
        deployment_id="service",
        expected_revision=baseline.revision_id,
        expected_generation=0,
        baseline_run_id="baseline-fixture",
        candidate_run_id=bundle.run_id,
        quality=store.put(evidence.model_dump_json().encode()),
        suite=store.put(default_suite().model_dump_json().encode()),
        target=candidate,
    )
    return registry, store, specification, CandidateAdapter(baseline)


async def test_registration_evaluation_and_callback_flow(tmp_path: Path) -> None:
    """The local end-to-end flow persists a real decision and waits for exact callback health."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        service = LifecycleService(registry, store)
        evaluated = await service.run(specification, evaluate_only=True)
        assert evaluated.status == "evaluated" and not adapter.calls
        result = await service.run(specification, adapter)
        assert result.status == "promoted" and result.decision_digest
        assert registry.decision(result.decision_digest).approved
        assert len(adapter.calls) == 1
        assert await service.run(specification, adapter) == result
        assert len(adapter.calls) == 1
        assert registry.history(specification.job_id)[-1].status == "promoted"
    finally:
        registry.close()


async def test_wrong_quality_cannot_reach_deployment_callback(tmp_path: Path) -> None:
    """Twice the request throughput cannot compensate for the candidate's wrong financial answer."""
    registry, store, specification, adapter = registered(tmp_path, wrong=True)
    try:
        result = await LifecycleService(registry, store).run(specification, adapter)
        assert result.status == "rejected" and result.decision_digest
        assert not registry.decision(result.decision_digest).approved
        assert adapter.calls == []
    finally:
        registry.close()


async def test_ambiguous_callback_requires_idempotency_or_reconciliation(tmp_path: Path) -> None:
    """Task retry cannot blindly repeat an uncertain non-idempotent deployment action."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        adapter.supports_idempotency, adapter.fail = False, True
        result = await LifecycleService(registry, store).run(specification, adapter)
        assert result.status == "deploying" and result.last_error == "OSError"
        result = await LifecycleService(registry, store).run(specification, adapter)
        assert result.status == "needs_reconciliation" and len(adapter.calls) == 1
        with pytest.raises(RegistryConflict):
            await LifecycleService(registry, store).run(
                specification.model_copy(update={"job_id": "another"}), adapter
            )
        adapter.active = specification.target
        assert (
            await LifecycleService(registry, store).run(specification, adapter)
        ).status == "promoted"
    finally:
        registry.close()


async def test_corrupt_registered_artifact_blocks_callback(tmp_path: Path) -> None:
    """Registration is not permission to trust later corrupted storage bytes."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        reference = specification.quality
        path = store.root / "sha256" / reference.sha256[:2] / reference.sha256
        path.write_bytes(b"changed")
        result = await LifecycleService(registry, store).run(specification, adapter)
        assert result.status == "registered" and result.last_error == "ValueError"
        assert not adapter.calls
    finally:
        registry.close()


@pytest.mark.parametrize("already_evaluated", [False, True])
def test_evaluation_retry_clears_only_resolved_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, already_evaluated: bool
) -> None:
    """A transient artifact failure can recover without retaining an obsolete failed gate flag."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.db"))
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(store.root))
        registry.create_job(specification.job_id, specification.canonical())
        if already_evaluated:
            assert evaluate_stage(specification.job_id) == specification.job_id
        reference = specification.quality
        path = store.root / "sha256" / reference.sha256[:2] / reference.sha256
        original = path.read_bytes()
        path.write_bytes(b"transient corrupted storage fixture")
        with pytest.raises(RuntimeError):
            evaluate_stage(specification.job_id)
        assert registry.job(specification.job_id).last_error == "ValueError"
        path.write_bytes(original)
        assert evaluate_stage(specification.job_id) == specification.job_id
        assert registry.job(specification.job_id).last_error is None
        assert not adapter.calls
    finally:
        registry.close()


async def test_cancellation_and_worker_collision_are_durable(tmp_path: Path) -> None:
    """Cancellation keeps action ambiguity while a concurrent task is denied the same lease."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        adapter.block, adapter.supports_idempotency = True, False
        service = LifecycleService(registry, store)
        task = asyncio.create_task(service.run(specification, adapter))
        await adapter.started.wait()
        with pytest.raises(RegistryConflict):
            await service.run(specification, adapter)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert registry.job(specification.job_id).last_error == "CancelledError"
        result = await service.run(specification, adapter)
        assert result.status == "needs_reconciliation" and len(adapter.calls) == 1
    finally:
        registry.close()


async def test_idempotent_retry_and_health_failure_do_not_duplicate_effects(tmp_path: Path) -> None:
    """Allow same-key replay after uncertainty; acknowledged apply gets only health probes."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        service = LifecycleService(registry, store)
        adapter.fail = True
        await service.run(specification, adapter)
        adapter.fail, adapter.ready = False, False
        result = await service.run(specification, adapter)
        assert result.status == "verifying" and len(adapter.calls) == 2
        assert adapter.calls[0] == adapter.calls[1]
        await service.run(specification, adapter)
        assert len(adapter.calls) == 2
        adapter.ready = True
        assert (await service.run(specification, adapter)).status == "promoted"
    finally:
        registry.close()


async def test_changed_specification_cannot_reuse_completed_job(tmp_path: Path) -> None:
    """An Airflow retry cannot replace a job's previously frozen input or target revision."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        service = LifecycleService(registry, store)
        await service.run(specification, adapter)
        with pytest.raises(RegistryConflict):
            await service.run(specification.model_copy(update={"expected_generation": 2}), adapter)
        assert len(adapter.calls) == 1
    finally:
        registry.close()


@pytest.mark.parametrize("mismatch", ["baseline", "candidate", "config"])
async def test_wrong_registered_identity_blocks_before_callback(
    tmp_path: Path, mismatch: str
) -> None:
    """A valid run cannot authorize a different baseline, candidate or deployment configuration."""
    registry, store, specification, adapter = registered(tmp_path)
    try:
        if mismatch == "baseline":
            specification = specification.model_copy(update={"expected_revision": "unmeasured"})
        else:
            changes = (
                {"revision_id": "unmeasured"}
                if mismatch == "candidate"
                else {"config_digest": "f" * 64}
            )
            specification = specification.model_copy(
                update={"target": specification.target.model_copy(update=changes)}
            )
        result = await LifecycleService(registry, store).run(specification, adapter)
        assert result.status == "registered" and result.last_error == "ValueError"
        assert not adapter.calls
    finally:
        registry.close()


async def test_missing_adapter_and_invalid_timeout_fail_closed(tmp_path: Path) -> None:
    """Evaluation can complete without an adapter, but activation requires an explicit callback."""
    registry, store, specification, _ = registered(tmp_path)
    try:
        for timeout in (0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                LifecycleService(registry, store, timeout)
        state = await LifecycleService(registry, store).run(specification)
        assert state.status == "evaluated" and state.last_error == "ValueError"
    finally:
        registry.close()


def test_identical_rejections_for_distinct_runs_have_separate_identity(tmp_path: Path) -> None:
    """Missing-evidence rejections retain each evaluated run even when their reason lists match."""
    registry, store, specification, _ = registered(tmp_path)
    try:
        artifact(tmp_path / "another", "another-run", 2, 0.2)
        registry.register_run(tmp_path / "another", store, specification.target)
        rejected = PromotionDecision(
            candidate_revision=specification.target.revision_id,
            candidate_digest=specification.target.digest(),
            policy_version="fixture",
            evidence_digest=None,
            rejection_reasons=("invalid_evidence:ValueError",),
        )
        first = registry.record_decision(rejected, specification.candidate_run_id)
        second = registry.record_decision(rejected, "another-run")
        assert first != second
        assert registry.decision(first) == registry.decision(second) == rejected
        registry.verify_decision_run(rejected, "another-run")
    finally:
        registry.close()


@pytest.mark.parametrize("wrong", [False, True])
def test_pipeline_stage_handoff_uses_durable_server_owned_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wrong: bool
) -> None:
    """Actual stage functions exchange job IDs and never pass a client-supplied gate approval."""
    registry, store, specification, adapter = registered(tmp_path, wrong=wrong)
    try:
        quality_path, suite_path = tmp_path / "quality.json", tmp_path / "suite.json"
        quality_path.write_bytes(store.get(specification.quality))
        suite_path.write_bytes(store.get(specification.suite))
        request = PipelineRequest(
            job_id=specification.job_id,
            deployment_id=specification.deployment_id,
            expected_generation=0,
            baseline_directory=tmp_path / "baseline",
            candidate_directory=tmp_path / "candidate",
            quality_file=quality_path,
            suite_file=suite_path,
            baseline_revision=registry.revision("baseline"),
            target_revision=specification.target,
            policy=specification.policy,
        )
        request_path = tmp_path / "request.json"
        request_path.write_text(request.model_dump_json())
        monkeypatch.setenv("FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.db"))
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(store.root))
        monkeypatch.setenv("FINSERVE_PIPELINE_REQUEST", str(request_path))
        module = types.ModuleType("finserve_fixture_adapter")

        def factory() -> CandidateAdapter:
            """Only a server-configured fixture module can supply the test deployment callback."""
            return adapter

        monkeypatch.setattr(module, "create", factory, raising=False)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setenv("FINSERVE_DEPLOYMENT_ADAPTER", "finserve_fixture_adapter:create")
        job_id = register_stage()
        if wrong:
            with pytest.raises(RuntimeError):
                evaluate_stage(job_id)
            assert not adapter.calls
        else:
            assert evaluate_stage(job_id) == job_id
            assert deploy_stage(job_id) == job_id
            assert len(adapter.calls) == 1
    finally:
        registry.close()
