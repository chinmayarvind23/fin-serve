"""Acknowledge a canonical, health-verified lifecycle cutover in the rollback controller."""

from functools import partial

from finserve.contracts.deployment import DeploymentState
from finserve.registry.artifacts import ArtifactStore
from finserve.registry.lifecycle import LifecycleSpec
from finserve.registry.metadata import Registry
from finserve.registry.model_assets import owned_disk
from finserve.registry.release_gate import evaluate_gate
from finserve.reliability.promotion import PromotionDecision
from finserve.reliability.rollback import ApplyRequest, DeploymentStore
from finserve.reliability.warm_routes import WarmRouteAdapter


def approved_activation(
    registry: Registry,
    artifacts: ArtifactStore,
    job_id: str,
) -> tuple[ApplyRequest, PromotionDecision]:
    """Require a completed lifecycle action matching the recomputed canonical gate decision."""
    spec = LifecycleSpec.model_validate_json(registry.specification(job_id))
    outcome = evaluate_gate(registry, artifacts, job_id)
    state = registry.job(job_id)
    if (
        outcome.status != "approved"
        or outcome.decision_digest is None
        or state.status != "promoted"
        or state.last_error is not None
        or state.decision_digest != outcome.decision_digest
    ):
        raise ValueError("activation acknowledgment requires a verified canonical deployment")
    return ApplyRequest(
        deployment_id=spec.deployment_id,
        expected_revision=spec.expected_revision,
        expected_generation=spec.expected_generation,
        target=spec.target,
        idempotency_key="lifecycle-" + spec.job_id,
    ), registry.decision(outcome.decision_digest)


async def acknowledge_release(
    registry: Registry,
    artifacts: ArtifactStore,
    job_id: str,
    control: DeploymentStore,
    adapter: WarmRouteAdapter,
) -> DeploymentState:
    """Probe traffic, then fence its route while acknowledging activation in the controller."""
    request, decision = await owned_disk(partial(approved_activation, registry, artifacts, job_id))
    health = await adapter.health(request.deployment_id)

    def acknowledge() -> DeploymentState:
        """Drain native writes on cancellation; retry the same activation identity."""
        control.register_revision(request.target)
        return adapter.store.acknowledge_candidate(control, request, decision, health)

    return await owned_disk(acknowledge)
