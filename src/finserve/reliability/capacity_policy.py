"""Demand hysteresis proposes capacity changes; runtime ownership remains with the controller."""

from typing import Literal, Self

from pydantic import Field, model_validator

from finserve.contracts.deployment import ImmutableModel

CapacityAction = Literal["hold", "up", "down"]


class CapacityPolicy(ImmutableModel):
    """Freeze dwell and occupancy thresholds before sampling rather than tuning a live run."""

    sample_seconds: float = Field(default=1, gt=0, le=3600)
    high_load: float = Field(default=0.8, gt=0, le=1)
    low_load: float = Field(default=0.2, ge=0, lt=1)
    high_samples: int = Field(default=3, ge=1, le=1000, strict=True)
    low_samples: int = Field(default=8, ge=1, le=1000, strict=True)
    cooldown_seconds: float = Field(default=30, ge=0, le=86400)

    @model_validator(mode="after")
    def hysteresis(self) -> Self:
        """Separate up/down thresholds and require at least as much evidence for removal."""
        if self.low_load >= self.high_load or self.low_samples < self.high_samples:
            raise ValueError("capacity policy requires separated thresholds and slower removal")
        return self


class CapacityObservation(ImmutableModel):
    """Only authenticated dispatched work is occupancy; uncertainty cannot establish idleness."""

    sequence: int = Field(ge=0, strict=True)
    epoch: str = Field(min_length=1, max_length=128)
    observed_seconds: float = Field(ge=0)
    serving_active: int = Field(ge=0, le=4096, strict=True)
    serving_capacity: int = Field(ge=1, le=4096, strict=True)
    members: Literal[1, 2]
    unresolved: bool = Field(strict=True)
    rejected_total: int = Field(ge=0, strict=True)


class CapacityMemory(ImmutableModel):
    """Persist the last observation so replay or a clock restart cannot manufacture dwell."""

    last: CapacityObservation | None = None
    high_streak: int = Field(default=0, ge=0, le=1000, strict=True)
    low_streak: int = Field(default=0, ge=0, le=1000, strict=True)
    last_action_seconds: float | None = Field(default=None, ge=0)


def decide_capacity(
    policy: CapacityPolicy, observation: CapacityObservation, memory: CapacityMemory
) -> tuple[CapacityMemory, CapacityAction]:
    """Count fresh spaced samples; a proposal never authorizes allocation or proves drain.

    A new epoch or observed membership change starts a fresh cooldown. Reordered samples
    are ignored, while conflicting replays and backward clocks/counters are invalid. A
    decision consumes its dwell even if its executor later refuses the change: durable
    slot reservation and reconciliation remain mandatory outside this pure function.
    """
    previous = memory.last
    if previous is None or observation.epoch != previous.epoch:
        return CapacityMemory(
            last=observation, last_action_seconds=observation.observed_seconds
        ), "hold"
    if observation.sequence < previous.sequence:
        return memory, "hold"
    if observation.sequence == previous.sequence:
        if observation != previous:
            raise ValueError("capacity observation sequence has conflicting content")
        return memory, "hold"
    if (
        observation.observed_seconds < previous.observed_seconds
        or observation.rejected_total < previous.rejected_total
    ):
        raise ValueError("capacity clock or rejection counter moved backward within one epoch")
    if observation.members != previous.members:
        return CapacityMemory(
            last=observation, last_action_seconds=observation.observed_seconds
        ), "hold"
    if observation.observed_seconds - previous.observed_seconds < policy.sample_seconds:
        return CapacityMemory(
            last=observation, last_action_seconds=memory.last_action_seconds
        ), "hold"
    return _fresh_decision(policy, observation, memory)


def _fresh_decision(
    policy: CapacityPolicy, observation: CapacityObservation, memory: CapacityMemory
) -> tuple[CapacityMemory, CapacityAction]:
    """A gap beyond twice the frozen period breaks dwell; overload blocks removal."""
    previous = memory.last
    assert previous is not None
    # A paused sampler cannot bridge stale occupancy merely by incrementing its sequence.
    # Permit at most one period of scheduling jitter, identically for scale-up and removal.
    consecutive = (
        observation.sequence == previous.sequence + 1
        and observation.observed_seconds - previous.observed_seconds <= 2 * policy.sample_seconds
    )
    rejected = observation.rejected_total > previous.rejected_total
    occupancy = observation.serving_active / observation.serving_capacity
    high = observation.members == 1 and (occupancy >= policy.high_load or rejected)
    low = (
        observation.members == 2
        and occupancy <= policy.low_load
        and not rejected
        and not observation.unresolved
    )
    high_streak = (
        min(policy.high_samples, (memory.high_streak if consecutive else 0) + 1) if high else 0
    )
    low_streak = (
        min(policy.low_samples, (memory.low_streak if consecutive else 0) + 1) if low else 0
    )
    last_action = memory.last_action_seconds
    cooled = (
        last_action is not None
        and observation.observed_seconds - last_action >= policy.cooldown_seconds
    )
    action: CapacityAction = "hold"
    if cooled and high_streak >= policy.high_samples:
        action = "up"
    elif cooled and low_streak >= policy.low_samples:
        action = "down"
    return CapacityMemory(
        last=observation,
        high_streak=high_streak if action == "hold" else 0,
        low_streak=low_streak if action == "hold" else 0,
        last_action_seconds=last_action if action == "hold" else observation.observed_seconds,
    ), action
