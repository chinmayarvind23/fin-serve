"""Retirement fences protect public traffic and reserve endpoints until exact runtime stop."""

from pathlib import Path

import pytest
from test_warm_rollback import bound_backend

from finserve.contracts.deployment import HealthObservation
from finserve.reliability.rollback import ApplyRequest, ControlConflict, DeploymentStore
from finserve.reliability.warm_routes import WarmBackend, WarmRouteStore


def test_retirement_reserves_endpoint_and_fences_racing_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-retirement backend read cannot authorize a later route write or endpoint alias."""
    routes, control = WarmRouteStore(tmp_path / "routes"), DeploymentStore(tmp_path / "control")
    baseline = bound_backend("baseline", "http://127.0.0.1:9000")
    unused = bound_backend("unused", "http://127.0.0.1:9001")
    routes.register(baseline)
    routes.register(unused)
    routes.bootstrap("service", "baseline")
    assert not routes.retire_unserved(control, baseline)
    assert routes.retire_unserved(control, unused)
    assert routes.retire_unserved(control, unused)
    alias = unused.model_copy(
        update={"revision": unused.revision.model_copy(update={"revision_id": "alias"})}
    )
    with pytest.raises(ControlConflict, match="distinct endpoint"):
        routes.register(alias)
    with pytest.raises(ControlConflict, match="retired"):
        routes.register(unused)

    def stale_backend(revision_id: str) -> WarmBackend:
        """Simulate a backend read completed before the retirement transaction."""
        assert revision_id == "unused"
        return unused

    monkeypatch.setattr(routes, "backend", stale_backend)
    with pytest.raises(ControlConflict, match="retired"):
        routes.bootstrap("late", "unused")
    with pytest.raises(ControlConflict, match="retired"):
        routes.apply(
            ApplyRequest(
                deployment_id="service",
                expected_revision="baseline",
                expected_generation=0,
                target=unused.revision,
                idempotency_key="late",
            )
        )
    routes.release_retired_endpoint(unused)
    routes.release_retired_endpoint(unused)
    routes.register(alias)


def test_retirement_preserves_controller_targets_and_historical_routes(tmp_path: Path) -> None:
    """Known-good and previously served revisions lack a proven stream-drain boundary."""
    routes, control = WarmRouteStore(tmp_path / "routes"), DeploymentStore(tmp_path / "control")
    old = bound_backend("old", "http://127.0.0.1:9000")
    new = bound_backend("new", "http://127.0.0.1:9001")
    routes.register(old)
    routes.register(new)
    routes.bootstrap("service", "old")
    routes.apply(
        ApplyRequest(
            deployment_id="service",
            expected_revision="old",
            expected_generation=0,
            target=new.revision,
            idempotency_key="switch",
        )
    )
    assert not routes.retire_unserved(control, old)
    assert not routes.retire_unserved(control, new)
    reserve = bound_backend("reserve", "http://127.0.0.1:9002")
    control.register_revision(reserve.revision)
    control.bootstrap(
        "other",
        "reserve",
        HealthObservation(
            revision_id="reserve",
            revision_digest=reserve.revision.digest(),
            ready=True,
            smoke_passed=True,
        ),
    )
    assert not routes.retire_unserved(control, reserve)
