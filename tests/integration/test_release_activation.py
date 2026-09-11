"""Canonical lifecycle and route receipts govern rollback-controller acknowledgment."""

from pathlib import Path

import httpx
import pytest
from test_release_gate import prepare_gate

from finserve.contracts.serving_profile import ServingProfileV1
from finserve.registry.lifecycle import LifecycleService, LifecycleSpec
from finserve.registry.release_activation import acknowledge_release
from finserve.reliability.rollback import ApplyRequest, ControlConflict, DeploymentStore
from finserve.reliability.warm_routes import (
    BackendConfiguration,
    WarmBackend,
    WarmRouteAdapter,
    WarmRouteStore,
)


@pytest.mark.parametrize("fault", ["none", "quality", "unhealthy", "superseded"])
async def test_canonical_activation_acknowledges_exact_route(tmp_path: Path, fault: str) -> None:
    """Exercise the real gate and SSE health parser with explicit synthetic deployment traffic."""
    registry, artifacts, gate = prepare_gate(tmp_path, wrong=fault == "quality")
    try:
        spec = LifecycleSpec.model_validate_json(registry.specification(gate.job_id))
        routes = WarmRouteStore(tmp_path / "routes.sqlite")
        control = DeploymentStore(tmp_path / "control.sqlite")
        for revision_id, profile_ref in (
            (spec.expected_revision, gate.baseline_profile),
            (spec.target.revision_id, gate.candidate_profile),
        ):
            revision = registry.revision(revision_id)
            profile = ServingProfileV1.model_validate_json(artifacts.get(profile_ref))
            routes.register(
                WarmBackend(
                    revision=revision,
                    serving_profile=profile,
                    configuration=BackendConfiguration(
                        base_url=profile.base_url, model=profile.served_model
                    ),
                )
            )
            control.register_revision(revision)
        routes.bootstrap(spec.deployment_id, spec.expected_revision)
        unhealthy = False
        calls = 0

        def response(request: httpx.Request) -> httpx.Response:
            """Return current route identity with one visible token and authoritative usage."""
            nonlocal calls
            calls += 1
            route = routes.snapshot(spec.deployment_id)
            return httpx.Response(
                503 if unhealthy else 200,
                headers={
                    "content-type": "text/event-stream",
                    "x-finserve-revision": route.revision_id,
                    "x-finserve-revision-digest": route.revision_digest,
                    "x-finserve-route-generation": str(route.generation),
                },
                content=(
                    b'data: {"choices":[{"index":0,"text":"ok","finish_reason":"stop"}],'
                    b'"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
                ),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            adapter = WarmRouteAdapter(routes, "http://fixture", client)
            control.bootstrap(
                spec.deployment_id, spec.expected_revision, await adapter.health(spec.deployment_id)
            )
            before = calls
            with pytest.raises(ValueError, match="verified canonical deployment"):
                await acknowledge_release(registry, artifacts, gate.job_id, control, adapter)
            assert calls == before
            state = await LifecycleService(registry, artifacts).run(spec, adapter)
            if fault == "quality":
                assert state.status == "rejected"
                with pytest.raises(ValueError):
                    await acknowledge_release(registry, artifacts, gate.job_id, control, adapter)
                assert control.deployment(spec.deployment_id).generation == 0
                return
            assert state.status == "promoted"
            assert control.deployment(spec.deployment_id).generation == 0
            unhealthy = fault == "unhealthy"
            if fault == "superseded":
                routes.apply(
                    ApplyRequest(
                        deployment_id=spec.deployment_id,
                        expected_revision=spec.target.revision_id,
                        expected_generation=1,
                        target=registry.revision(spec.expected_revision),
                        idempotency_key="fixture-later-action",
                    )
                )
            if fault in {"unhealthy", "superseded"}:
                with pytest.raises(ControlConflict):
                    await acknowledge_release(registry, artifacts, gate.job_id, control, adapter)
                assert control.deployment(spec.deployment_id).generation == 0
                return
            acknowledged = await acknowledge_release(
                registry, artifacts, gate.job_id, control, adapter
            )
            assert acknowledged.generation == 1
            assert acknowledged.active_revision == spec.target.revision_id
            assert acknowledged.known_good_revision == spec.expected_revision
            assert (
                await acknowledge_release(registry, artifacts, gate.job_id, control, adapter)
                == acknowledged
            )
            assert len(routes.history(spec.deployment_id)) == 2
    finally:
        registry.close()
