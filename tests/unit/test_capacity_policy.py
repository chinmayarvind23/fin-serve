"""Temporal evidence must earn a scaling proposal without weakening runtime drain fences."""

import pytest
from pydantic import ValidationError

from finserve.reliability.capacity_policy import (
    CapacityMemory,
    CapacityObservation,
    CapacityPolicy,
    decide_capacity,
)


def sample(
    sequence: int, *, members: int = 1, active: int = 4, **changes: object
) -> CapacityObservation:
    """Supply a real ordered clock population with explicit test interventions."""
    return CapacityObservation.model_validate(
        {
            "sequence": sequence,
            "epoch": "process-a",
            "observed_seconds": float(sequence),
            "serving_active": active,
            "serving_capacity": 4 * members,
            "members": members,
            "unresolved": False,
            "rejected_total": 0,
            **changes,
        }
    )


def test_sustained_pressure_then_longer_quiet_window() -> None:
    """A complete 1-to-2-to-1 policy cycle needs distinct high and low dwell populations."""
    policy = CapacityPolicy(high_samples=2, low_samples=3, cooldown_seconds=2)
    memory = CapacityMemory()
    actions: list[str] = []
    for sequence in range(3):
        memory, action = decide_capacity(policy, sample(sequence), memory)
        actions.append(action)
    assert actions == ["hold", "hold", "up"]
    actions = []
    for sequence in range(3, 7):
        memory, action = decide_capacity(policy, sample(sequence, members=2, active=0), memory)
        actions.append(action)
    assert actions == ["hold", "hold", "hold", "down"]


def test_replays_and_conflicting_identity_do_not_create_dwell() -> None:
    """Reading the same durable sample repeatedly cannot trigger another scale action."""
    policy = CapacityPolicy(cooldown_seconds=0)
    memory, _ = decide_capacity(policy, sample(0), CapacityMemory())
    memory, _ = decide_capacity(policy, sample(1), memory)
    for observation in (sample(0), sample(1), sample(1)):
        assert decide_capacity(policy, observation, memory) == (memory, "hold")
    with pytest.raises(ValueError, match="conflicting"):
        decide_capacity(policy, sample(1, active=0), memory)


def test_restart_and_missing_samples_reset_evidence() -> None:
    """Clock restart and sample gaps cannot bridge old demand into a new controller epoch."""
    policy = CapacityPolicy(high_samples=2, low_samples=2, cooldown_seconds=0)
    memory = CapacityMemory()
    for sequence in (0, 1, 4):
        memory, action = decide_capacity(policy, sample(sequence), memory)
        assert action == "hold"
    memory, action = decide_capacity(policy, sample(0, epoch="process-b"), memory)
    assert action == "hold" and memory.high_streak == 0


def test_fast_samples_and_unknown_work_cannot_establish_idle_drain() -> None:
    """Pre-dispatch uncertainty, however old, prevents the policy from proposing removal."""
    policy = CapacityPolicy(high_samples=1, low_samples=1, cooldown_seconds=0)
    memory, _ = decide_capacity(policy, sample(0, members=2, active=0), CapacityMemory())
    memory, action = decide_capacity(
        policy, sample(1, members=2, active=0, observed_seconds=0.5), memory
    )
    assert action == "hold"
    for sequence in range(2, 10):
        memory, action = decide_capacity(
            policy, sample(sequence, members=2, active=0, unresolved=True), memory
        )
        assert action == "hold"


def test_only_new_capacity_rejections_contribute_pressure() -> None:
    """A historical cumulative rejection does not become a permanently overloaded sample."""
    policy = CapacityPolicy(high_samples=2, low_samples=2, cooldown_seconds=0)
    memory, _ = decide_capacity(policy, sample(0, active=0, rejected_total=5), CapacityMemory())
    memory, action = decide_capacity(policy, sample(1, active=0, rejected_total=6), memory)
    assert action == "hold"
    memory, action = decide_capacity(policy, sample(2, active=0, rejected_total=6), memory)
    assert action == "hold" and memory.high_streak == 0
    with pytest.raises(ValueError, match="backward"):
        decide_capacity(policy, sample(3, active=0, rejected_total=0), memory)


def test_new_epoch_restarts_cooldown_even_under_continuous_load() -> None:
    """Restarting a process cannot bypass the frozen anti-oscillation interval."""
    policy = CapacityPolicy(high_samples=1, low_samples=1, cooldown_seconds=5)
    memory = CapacityMemory()
    for sequence in range(5):
        memory, action = decide_capacity(policy, sample(sequence), memory)
        assert action == "hold"
    _, action = decide_capacity(policy, sample(5), memory)
    assert action == "up"


@pytest.mark.parametrize("members,active,expected", [(1, 4, "up"), (2, 0, "down")])
def test_long_pause_breaks_dwell_even_with_consecutive_sequence(
    members: int, active: int, expected: str
) -> None:
    """An hour-old load sample cannot contribute to a fresh scaling decision."""
    policy = CapacityPolicy(high_samples=2, low_samples=2, cooldown_seconds=0)
    memory = CapacityMemory()
    for sequence, timestamp in ((0, 0), (1, 1), (2, 3600)):
        memory, action = decide_capacity(
            policy,
            sample(sequence, members=members, active=active, observed_seconds=timestamp),
            memory,
        )
        assert action == "hold"
    _, action = decide_capacity(
        policy, sample(3, members=members, active=active, observed_seconds=3601), memory
    )
    assert action == expected


@pytest.mark.parametrize(
    "changes",
    [{"low_load": 0.9}, {"low_samples": 1}, {"sample_seconds": 0}, {"high_samples": True}],
)
def test_invalid_policy_is_rejected_before_observation(changes: dict[str, object]) -> None:
    """Reject degenerate or coercible thresholds rather than interpret an unsafe policy."""
    with pytest.raises(ValidationError):
        CapacityPolicy.model_validate(changes)
