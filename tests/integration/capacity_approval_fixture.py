"""Fresh synthetic gate artifacts exercise canonical approval before capacity enrollment."""

import json
from pathlib import Path

from test_promotion import artifact, quality

from finserve.contracts.managed_runtime import RuntimeLaunchSpec
from finserve.evaluation.quality import default_suite
from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.lifecycle import LifecycleService, LifecycleSpec
from finserve.registry.metadata import Registry
from finserve.registry.producer_stages import ProducerStages
from finserve.registry.release_activation import acknowledge_release, complete_probation
from finserve.registry.release_gate import GateRequest, evaluate_gate, freeze_gate_request
from finserve.reliability.monitor import MonitorPolicy, ProbeMonitor
from finserve.reliability.rollback import DeploymentStore
from finserve.reliability.warm_routes import WarmRouteAdapter


def declare_approval(
    root: Path,
    registry: Registry,
    artifacts: LocalArtifactStore,
    baseline: RuntimeLaunchSpec,
    primary: RuntimeLaunchSpec,
) -> LifecycleSpec:
    """Create explicitly synthetic timing/answer fixtures bound to actual test launch profiles."""
    for name, duration, launch in (("baseline", 4, baseline), ("candidate", 2, primary)):
        directory = root / name
        artifact(directory, name + "-fixture", duration, duration / 10)
        manifest = directory / "manifest.json"
        value = json.loads(manifest.read_text())
        value["url"] = launch.profile.base_url + "/completions"
        value["configuration"]["model"] = launch.profile.served_model
        for field, source in (
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
            ("revision", "source_revision"),
            ("image_digest", "image_digest"),
            ("config_digest", "config_digest"),
            ("engine", "engine"),
            ("engine_config", "engine_config"),
        ):
            value["configuration"][field] = getattr(launch.revision, source)
        manifest.write_text(json.dumps(value))
        registry.register_run(directory, artifacts, launch.revision)
    evidence = quality().model_copy(
        update={
            "reference_model_revision": baseline.profile.model_revision,
            "candidate_model_revision": primary.profile.model_revision,
        }
    )
    spec = LifecycleSpec(
        job_id="capacity-release",
        deployment_id="capacity-service",
        expected_revision=baseline.revision.revision_id,
        expected_generation=0,
        baseline_run_id="baseline-fixture",
        candidate_run_id="candidate-fixture",
        quality=artifacts.put(evidence.model_dump_json().encode()),
        suite=artifacts.put(default_suite().model_dump_json().encode()),
        target=primary.revision,
        gate_mode="canonical-profile-v1",
    )
    registry.create_job(spec.job_id, spec.canonical())
    freeze_gate_request(
        registry,
        GateRequest(
            job_id=spec.job_id,
            baseline_profile=artifacts.put(baseline.profile.model_dump_json().encode()),
            candidate_profile=artifacts.put(primary.profile.model_dump_json().encode()),
        ),
    )
    return spec


async def approve_primary(
    registry: Registry,
    artifacts: LocalArtifactStore,
    spec: LifecycleSpec,
    control: DeploymentStore,
    adapter: WarmRouteAdapter,
) -> None:
    """Execute real health, promotion, acknowledgment and spaced probation without fake receipts."""
    control.bootstrap(
        spec.deployment_id, spec.expected_revision, await adapter.health(spec.deployment_id)
    )
    gate = evaluate_gate(registry, artifacts, spec.job_id)
    assert gate.status == "approved", gate
    promoted = await LifecycleService(registry, artifacts).run(spec, adapter)
    assert promoted.status == "promoted", promoted
    await acknowledge_release(registry, artifacts, spec.job_id, control, adapter)
    monitor = ProbeMonitor(
        MonitorPolicy(
            monitor_id="capacity-probation",
            deployment_id=spec.deployment_id,
            revision_id=spec.target.revision_id,
            revision_digest=spec.target.digest(),
            generation=1,
            maximum_probes=3,
            consecutive_regressions=2,
            interval_seconds=0.1,
            slow_probe_seconds=4,
            probe_timeout_seconds=5,
        ),
        ProducerStages(registry, artifacts),
        control,
        adapter,
    )
    await monitor.run()
    stable = await complete_probation(spec.job_id, monitor)
    assert stable.active_revision == stable.known_good_revision == spec.target.revision_id
