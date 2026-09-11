"""Canonical release registration consumes collector rows with explicit fixture provenance."""

import json
import time
from pathlib import Path

import httpx
import pytest
from test_managed_runtime import Daemon, handler, performance_spec, specification, upstream

from finserve.benchmark.gpu import TelemetrySample
from finserve.benchmark.runner import RunConfig
from finserve.contracts.producer import QualityCollectionSpec
from finserve.evaluation.quality import default_suite
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.lifecycle import LifecycleSpec
from finserve.registry.managed_quality import managed_quality_stage
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.metadata import Registry, RegistryConflict
from finserve.registry.performance_stages import performance_stage
from finserve.registry.pipeline import register_produced_stage
from finserve.registry.produced_release import (
    ProducedReleasePlan,
    ReleaseCohort,
    freeze_produced_release,
    register_produced_release,
)
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.release_gate import evaluate_gate
from finserve.registry.runtime_stages import launch_runtime_stage
from finserve.reliability.promotion import PromotionPolicy, QualityEvidence


async def test_collected_receipts_register_canonical_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise launch->quality/performance->registration->gate; fixture answers remain wrong."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.sqlite"))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(tmp_path / "artifacts"))
        base = upstream(journal, specification(tmp_path), tmp_path)
        runtimes = [
            base.model_copy(
                update={"revision": base.revision.model_copy(update={"revision_id": name})}
            )
            for name in ("baseline", "candidate")
        ]
        cohorts = [
            ReleaseCohort(
                performance_stage=name + ":performance",
                quality_stage=name + ":quality",
                performance=performance_spec(runtime),
                quality=QualityCollectionSpec(
                    collection_id=name + "-quality",
                    profile=runtime.profile,
                    revision=runtime.revision,
                    suite=default_suite(),
                    configuration=performance_spec(runtime).configuration,
                ),
            )
            for name, runtime in zip(("baseline", "candidate"), runtimes, strict=True)
        ]
        plan = ProducedReleasePlan(
            job_id="release",
            deployment_id="fixture",
            expected_generation=0,
            baseline=cohorts[0],
            candidate=cohorts[1],
            policy=PromotionPolicy(),
        )
        plan_id = freeze_produced_release(journal, plan)
        assert freeze_produced_release(journal, plan) == plan_id
        with pytest.raises(RegistryConflict):
            freeze_produced_release(journal, plan.model_copy(update={"expected_generation": 1}))
        with pytest.raises(ValueError):
            ProducedReleasePlan.model_validate(
                {**plan.model_dump(), "candidate": plan.baseline.model_dump()}
            )

        def fixture_git(arguments: list[str], **kwargs: object) -> str:
            """Synthetic clean-source declarations never serve as real benchmark evidence."""
            return "" if arguments[1] == "status" else "c" * 40 + "\n"

        calls = 0

        def response(request: httpx.Request) -> httpx.Response:
            """Count offered work across runtime observations and both collectors."""
            nonlocal calls
            calls += 1
            return handler(request)

        def client(config: RunConfig) -> httpx.AsyncClient:
            """The isolated performance loop uses the same explicit protocol fixture."""
            return httpx.AsyncClient(transport=httpx.MockTransport(response))

        monkeypatch.setattr("finserve.benchmark.experiment.benchmark_client", client)
        monkeypatch.setattr("finserve.benchmark.experiment.subprocess.check_output", fixture_git)
        monkeypatch.setattr(
            "finserve.benchmark.experiment.collect",
            lambda: TelemetrySample(
                epoch_s=time.time(), collection_seconds=0, devices=[], error="FixtureNoGPU"
            ),
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as http:
            for cohort, runtime_spec in zip(cohorts, runtimes, strict=True):
                name = runtime_spec.revision.revision_id
                runtime = DockerRuntime(Daemon(runtime_spec))
                await launch_runtime_stage(
                    journal,
                    name + ":launch",
                    "job:model",
                    "job:build",
                    runtime_spec,
                    tmp_path / name / "runtime",
                    http,
                    runtime,
                )
                await managed_quality_stage(
                    journal,
                    cohort.quality_stage,
                    name + ":launch",
                    cohort.quality,
                    tmp_path / name / "quality",
                    http,
                    runtime,
                )
                await performance_stage(
                    journal,
                    cohort.performance_stage,
                    name + ":launch",
                    cohort.performance,
                    tmp_path / name / "performance",
                    http,
                    runtime,
                )
        before = calls
        assert register_produced_release(journal, plan_id) == plan.job_id
        assert register_produced_release(journal, plan_id) == plan.job_id
        monkeypatch.setenv(
            "FINSERVE_REGISTRY_URL", "sqlite:///" + str(tmp_path / "registry.sqlite")
        )
        monkeypatch.setenv("FINSERVE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
        assert register_produced_stage(plan_id) == plan.job_id
        assert calls == before
        outcome = evaluate_gate(registry, journal.artifacts, plan.job_id)
        assert outcome.status == "rejected"
        stored = LifecycleSpec.model_validate_json(registry.specification(plan.job_id))
        quality = QualityEvidence.model_validate_json(journal.artifacts.get(stored.quality))
        assert stored.gate_mode == "canonical-profile-v1"
        assert quality.candidate == quality.reference
        assert outcome.decision_digest is not None
        assert registry.decision(outcome.decision_digest).candidate_accuracy == 0
        late = plan.model_copy(update={"job_id": "late-release"})
        with pytest.raises(ValueError, match="before release plan"):
            register_produced_release(journal, freeze_produced_release(journal, late))
        with pytest.raises(KeyError):
            registry.specification(late.job_id)
        transplanted = plan.model_copy(
            update={
                "job_id": "transplanted",
                "baseline": plan.baseline.model_copy(
                    update={
                        "performance_stage": "copied:baseline-performance",
                        "quality_stage": "copied:baseline-quality",
                    }
                ),
                "candidate": plan.candidate.model_copy(
                    update={
                        "performance_stage": "copied:candidate-performance",
                        "quality_stage": "copied:candidate-quality",
                    }
                ),
            }
        )
        transplant_id = freeze_produced_release(journal, transplanted)
        for original, copied in zip(
            (plan.baseline, plan.candidate),
            (transplanted.baseline, transplanted.candidate),
            strict=True,
        ):
            for old_id, new_id in (
                (original.performance_stage, copied.performance_stage),
                (original.quality_stage, copied.quality_stage),
            ):
                old = journal.state(old_id)
                journal.declare(new_id, old.input)
                attempt = journal.start(new_id)
                assert attempt.attempt_id is not None and old.output is not None
                journal.finish(new_id, attempt.attempt_id, old.output)
        with pytest.raises(ValueError, match="observations precede"):
            register_produced_release(journal, transplant_id)
        (tmp_path / "gate.json").write_text(json.dumps(outcome.model_dump()))
    finally:
        registry.close()
