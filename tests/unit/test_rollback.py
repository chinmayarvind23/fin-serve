"""Failure-oriented rollback tests use injected adapters, not fabricated deployment claims."""

import asyncio
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from finserve.contracts.deployment import (
    HealthObservation,
    RegressionSignal,
    Revision,
    RollbackRecord,
)
from finserve.reliability.promotion import PromotionDecision
from finserve.reliability.rollback import (
    ApplyRequest,
    ControlConflict,
    DeploymentStore,
    RollbackController,
    validate_transition,
)


def revision(name: str) -> Revision:
    """Use explicit immutable synthetic identities; no test digest describes a real image."""
    return Revision(
        revision_id=name,
        model_revision="weights-v1",
        tokenizer_revision="tokenizer-v1",
        source_revision="source-v1",
        image_digest="sha256:" + "a" * 64,
        config_digest="b" * 64,
        engine="fixture",
        engine_config=name,
    )


def healthy(value: Revision, ready: bool = True) -> HealthObservation:
    """Health verifies both readable revision ID and exact content identity."""
    return HealthObservation(
        revision_id=value.revision_id,
        revision_digest=value.digest(),
        ready=ready,
        smoke_passed=ready,
    )


def approval(value: Revision, approved: bool = True) -> PromotionDecision:
    """A unit fixture isolates orchestration; integration tests recompute actual gate decisions."""
    return PromotionDecision(
        candidate_revision=value.revision_id,
        candidate_digest=value.digest(),
        policy_version="fixture",
        evidence_digest="c" * 64 if approved else None,
        rejection_reasons=() if approved else ("quality_gate_failed",),
    )


def initialized(path: Path) -> tuple[DeploymentStore, Revision, Revision, RegressionSignal]:
    """Start with a verified known-good revision followed by a verified candidate activation."""
    store = DeploymentStore(path)
    good, bad = revision("good"), revision("candidate")
    for value in (good, bad):
        store.register_revision(value)
    store.bootstrap("service", good.revision_id, healthy(good))
    store.activate_candidate("service", 0, approval(bad), healthy(bad))
    signal = RegressionSignal(
        signal_id="regression-1",
        deployment_id="service",
        observed_revision=bad.revision_id,
        observed_generation=1,
        detected_at=time.time(),
        detector="quality-v1",
        reason="fixed-suite accuracy regression",
    )
    return store, good, bad, signal


class Adapter:
    """A deterministic fake adapter makes external ambiguity and exact-health checks observable."""

    supports_idempotency = True

    def __init__(self, active: Revision, target: Revision) -> None:
        """Track adapter calls separately from database state to detect duplicate actions."""
        self.active, self.target = active, target
        self.calls: list[ApplyRequest] = []
        self.fail_apply = False
        self.change_before_failure = False
        self.ready = True
        self.block = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def apply(self, request: ApplyRequest) -> None:
        """Optionally fail after changing traffic, the ambiguous external-effect case."""
        self.calls.append(request)
        self.started.set()
        if self.block:
            await self.release.wait()
        if self.change_before_failure:
            self.active = request.target
        if self.fail_apply:
            raise OSError("external API may have applied")
        self.active = request.target

    async def health(self, deployment_id: str) -> HealthObservation:
        """Return what traffic actually serves, independently of what apply intended."""
        return healthy(self.active, self.ready)


async def test_real_clock_restoration_and_idempotent_resume(tmp_path: Path) -> None:
    """Detector-to-health timing is observed in order and a completed retry makes no new action."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    before = time.time()
    detected = store.detect(signal)
    assert store.detect(signal) == detected
    adapter = Adapter(bad, good)
    result = await RollbackController(store).resume(detected.operation_id, adapter)
    assert result.status == "restored"
    assert result.started_at is not None and result.restored_at is not None
    assert result.detected_at == signal.detected_at
    assert (
        signal.detected_at
        <= before
        <= result.received_at
        <= result.started_at
        <= result.restored_at
        <= time.time()
    )
    assert result.duration_seconds() == result.restored_at - result.detected_at
    assert store.deployment("service").active_revision == "good"
    assert store.deployment("service").generation == 2
    assert [event.status for event in store.history(signal.signal_id)] == [
        "detected",
        "applying",
        "verifying",
        "restored",
    ]
    assert (
        await RollbackController(DeploymentStore(store.path)).resume(signal.signal_id, adapter)
        == result
    )
    assert len(adapter.calls) == 1


async def test_failed_health_never_claims_recovery(tmp_path: Path) -> None:
    """An acknowledged apply is insufficient; readiness and smoke must succeed afterward."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    store.detect(signal)
    adapter = Adapter(bad, good)
    adapter.ready = False
    result = await RollbackController(store).resume(signal.signal_id, adapter)
    assert result.status == "verifying"
    assert result.restored_at is None and result.duration_seconds() is None
    assert store.deployment("service").active_revision == "candidate"
    await RollbackController(store).resume(signal.signal_id, adapter)
    assert len(adapter.calls) == 1
    adapter.ready = True
    assert (await RollbackController(store).resume(signal.signal_id, adapter)).status == "restored"


async def test_nonidempotent_ambiguous_failure_is_not_replayed(tmp_path: Path) -> None:
    """After an uncertain external response, unsupported retries require reconciliation."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    store.detect(signal)
    adapter = Adapter(bad, good)
    adapter.supports_idempotency, adapter.fail_apply = False, True
    failed = await RollbackController(store).resume(signal.signal_id, adapter)
    assert failed.status == "applying" and failed.last_error == "OSError"
    resumed = await RollbackController(DeploymentStore(store.path)).resume(
        signal.signal_id, adapter
    )
    assert resumed.status == "needs_reconciliation"
    assert len(adapter.calls) == 1
    adapter.active = good
    assert (await RollbackController(store).resume(signal.signal_id, adapter)).status == "restored"
    assert len(adapter.calls) == 1


async def test_idempotent_retry_preserves_same_key(tmp_path: Path) -> None:
    """A supported retry reuses the durable operation key and immutable target."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    store.detect(signal)
    adapter = Adapter(bad, good)
    adapter.fail_apply = True
    await RollbackController(store).resume(signal.signal_id, adapter)
    adapter.fail_apply = False
    result = await RollbackController(DeploymentStore(store.path)).resume(signal.signal_id, adapter)
    assert result.status == "restored" and result.apply_attempts == 2
    assert adapter.calls[0] == adapter.calls[1]


async def test_cancellation_persists_and_does_not_replay_unsupported_action(tmp_path: Path) -> None:
    """Cancellation during apply leaves durable ambiguity after the initiating task disappears."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    store.detect(signal)
    adapter = Adapter(bad, good)
    adapter.block, adapter.supports_idempotency = True, False
    task = asyncio.create_task(RollbackController(store).resume(signal.signal_id, adapter))
    await adapter.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.operation(signal.signal_id).last_error == "CancelledError"
    result = await RollbackController(DeploymentStore(store.path)).resume(signal.signal_id, adapter)
    assert result.status == "needs_reconciliation" and len(adapter.calls) == 1


async def test_competing_worker_and_activation_are_fenced(tmp_path: Path) -> None:
    """A second controller cannot duplicate apply or activate during an owned rollback."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    store.detect(signal)
    adapter = Adapter(bad, good)
    adapter.block = True
    task = asyncio.create_task(RollbackController(store).resume(signal.signal_id, adapter))
    await adapter.started.wait()
    with pytest.raises(ControlConflict):
        await RollbackController(DeploymentStore(store.path)).resume(signal.signal_id, adapter)
    with pytest.raises(ControlConflict):
        store.activate_candidate("service", 1, approval(bad), healthy(bad))
    adapter.release.set()
    assert (await task).status == "restored"


def test_revision_immutability_and_stale_signals(tmp_path: Path) -> None:
    """The same revision ID cannot change bits and old detectors cannot target newer generations."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    store.register_revision(good)
    with pytest.raises(ControlConflict):
        store.register_revision(good.model_copy(update={"engine_config": "changed"}))
    with pytest.raises(ControlConflict):
        store.detect(signal.model_copy(update={"observed_generation": 0}))
    store.detect(signal)
    with pytest.raises(ControlConflict):
        store.detect(signal.model_copy(update={"reason": "changed intent"}))
    with pytest.raises(ValueError):
        store.activate_candidate("service", 1, approval(bad, False), healthy(bad))


def test_health_identity_and_rollback_timestamp_invariants(tmp_path: Path) -> None:
    """Healthy wrong revisions and impossible clock sequences fail closed."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    operation = store.detect(signal)
    assert operation.duration_seconds() is None
    assert not healthy(bad).verifies(good)
    assert not healthy(good).model_copy(update={"revision_digest": "e" * 64}).verifies(good)
    store.claim(signal.signal_id, "owner", 30)
    store.update(signal.signal_id, "owner", "applying", start_attempt=True)
    with pytest.raises(ValueError):
        store.update(signal.signal_id, "owner", "restored", health=healthy(bad))
    with pytest.raises(ValidationError):
        RollbackRecord.model_validate({**operation.model_dump(), "status": "restored"})
    with pytest.raises(ControlConflict):
        validate_transition(operation, "verifying", False)


@pytest.mark.parametrize(
    "changes",
    [
        {"started_at": 1},
        {"restored_at": 200},
        {"started_at": 101, "restored_at": 100, "status": "restored"},
    ],
)
def test_impossible_rollback_clocks_fail(changes: dict[str, object]) -> None:
    """A wall-clock regression must invalidate the duration rather than report fast recovery."""
    signal = RegressionSignal(
        signal_id="clock",
        deployment_id="service",
        observed_revision="bad",
        observed_generation=1,
        detected_at=100,
        detector="fixture",
        reason="fixture",
    )
    with pytest.raises(ValidationError):
        RollbackRecord.model_validate(
            {
                "operation_id": "clock",
                "signal": signal.model_dump(),
                "target_revision": "good",
                "target_digest": "b" * 64,
                "detected_at": 100,
                "received_at": 100,
                **changes,
            }
        )


def test_bootstrap_validation_and_mutable_identity_rejection(tmp_path: Path) -> None:
    """Initialization cannot overwrite live truth or call a mismatched health probe known-good."""
    store = DeploymentStore(tmp_path / "truth.db")
    good, bad = revision("good"), revision("bad")
    store.register_revision(good)
    with pytest.raises(ValueError):
        store.bootstrap("service", good.revision_id, healthy(bad))
    first = store.bootstrap("service", good.revision_id, healthy(good))
    assert store.bootstrap("service", good.revision_id, healthy(good)) == first
    store.register_revision(bad)
    with pytest.raises(ControlConflict):
        store.bootstrap("service", bad.revision_id, healthy(bad))
    with pytest.raises(ValidationError):
        Revision.model_validate({**good.model_dump(), "model_revision": "main"})


def test_expired_lease_owner_cannot_commit_or_release_new_owner(tmp_path: Path) -> None:
    """Separate processes share the same fencing checks; stale holders have no write authority."""
    store, _, _, signal = initialized(tmp_path / "truth.db")
    operation = store.detect(signal)
    with pytest.raises(ValueError):
        store.claim(signal.signal_id, "", 30)
    with pytest.raises(ValueError):
        store.claim(signal.signal_id, "first", float("nan"))
    store.claim(signal.signal_id, "first", 30)
    with store.transaction() as connection:
        connection.execute("UPDATE leases SET expires=0 WHERE id=?", (signal.signal_id,))
    store.claim(signal.signal_id, "second", 30)
    store.release(signal.signal_id, "first")
    with pytest.raises(ControlConflict):
        store.update(signal.signal_id, "first", "applying", start_attempt=True)
    assert (
        store.update(signal.signal_id, "second", "applying", start_attempt=True).apply_attempts == 1
    )
    with pytest.raises(ControlConflict):
        validate_transition(operation, "detected", True)
    with pytest.raises(ControlConflict):
        validate_transition(operation, "applying", False)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeouts_fail(tmp_path: Path, timeout: float) -> None:
    """A nonfinite deadline would invalidate the lease and external-action bounds."""
    with pytest.raises(ValueError):
        RollbackController(DeploymentStore(tmp_path / "truth.db"), timeout)


async def test_lost_apply_response_reconciles_without_another_action(tmp_path: Path) -> None:
    """If apply changed traffic before failing, exact health resolves ambiguity without replay."""
    store, good, bad, signal = initialized(tmp_path / "truth.db")
    store.detect(signal)
    adapter = Adapter(bad, good)
    adapter.change_before_failure = adapter.fail_apply = True
    adapter.supports_idempotency = False
    await RollbackController(store).resume(signal.signal_id, adapter)
    result = await RollbackController(store).resume(signal.signal_id, adapter)
    assert result.status == "restored" and len(adapter.calls) == 1
    with pytest.raises(ControlConflict):
        store.claim(signal.signal_id, "late", 30)


def test_activation_and_gate_recording_are_idempotent(tmp_path: Path) -> None:
    """Replayed acknowledgements cannot advance generations or erase rejected gates."""
    store, _, bad, _ = initialized(tmp_path / "truth.db")
    before = store.deployment("service")
    assert store.activate_candidate("service", 0, approval(bad), healthy(bad)) == before
    assert store.deployment("service").generation == 1
    rejected = approval(bad, False)
    digest = store.record_decision(rejected)
    assert store.record_decision(rejected) == digest
    assert store.decision(digest) == rejected


def test_detector_queue_delay_is_preserved_and_future_signal_rejected(tmp_path: Path) -> None:
    """Detection and controller receipt are distinct timestamps, preventing hidden queue delay."""
    store, _, _, signal = initialized(tmp_path / "truth.db")
    with pytest.raises(ValidationError):
        store.detect(signal.model_copy(update={"detected_at": time.time() + 60}))
    earlier = signal.model_copy(update={"detected_at": time.time() - 30})
    result = store.detect(earlier)
    assert result.received_at - result.detected_at >= 30


def test_known_good_advances_only_after_explicit_probation(tmp_path: Path) -> None:
    """A warmed candidate becomes rollback truth only after recorded probation evidence."""
    store, _, candidate, signal = initialized(tmp_path / "truth.db")
    with pytest.raises(ValueError):
        store.mark_stable("service", 1, approval(candidate), healthy(candidate), "missing")
    with pytest.raises(ValueError):
        store.mark_stable("service", 1, approval(candidate, False), healthy(candidate), "f" * 64)
    with pytest.raises(ControlConflict):
        store.mark_stable("service", 0, approval(candidate), healthy(candidate), "f" * 64)
    stable = store.mark_stable("service", 1, approval(candidate), healthy(candidate), "f" * 64)
    assert stable.known_good_revision == candidate.revision_id
    assert (
        store.mark_stable("service", 1, approval(candidate), healthy(candidate), "f" * 64) == stable
    )
    with pytest.raises(ControlConflict):
        store.mark_stable("service", 1, approval(candidate), healthy(candidate), "e" * 64)
    with pytest.raises(ControlConflict):
        store.detect(signal)


def test_truth_database_cannot_be_written_in_source_repository() -> None:
    """Control-plane state is private execution evidence, kept outside the source tree."""
    with pytest.raises(ValueError):
        DeploymentStore(Path(__file__).resolve().parents[2] / "should-not-exist.db")
