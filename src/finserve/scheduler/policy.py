"""Deterministic routing scores; hard eligibility always precedes optimization preferences."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finserve.contracts.routing import ReplicaSnapshot, RoutingDecision, RoutingRequest


class RoutingPolicy(BaseModel):
    """Conservative defaults are tunable hypotheses, not evidence of performance improvement."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    mode: Literal["least_load", "adaptive"] = "least_load"
    snapshot_ttl_seconds: float = Field(default=5, gt=0, le=300)
    gpu_memory_limit: float = Field(default=0.95, gt=0, le=1)
    memory_weight: float = Field(default=0.2, ge=0, le=1)
    cache_bonus: float = Field(default=0.1, ge=0, le=0.25)
    cache_load_gap: float = Field(default=0.15, ge=0, le=0.25)


@dataclass(frozen=True)
class Candidate:
    """Carry already-reconciled occupancy so all policies obey identical capacity limits."""

    snapshot: ReplicaSnapshot
    effective_requests: int

    @property
    def load(self) -> float:
        """Normalize heterogeneous capacities after filtering zero-capacity replicas."""
        return self.effective_requests / self.snapshot.capacity


def is_eligible(
    candidate: Candidate, request: RoutingRequest, policy: RoutingPolicy, now: float
) -> bool:
    """Fail closed on stale telemetry, hardware mismatch, saturation, or unhealthy replicas."""
    snapshot = candidate.snapshot
    age = now - snapshot.received_at
    available = snapshot.capacity > candidate.effective_requests and snapshot.capacity > 0
    matches_gpu = (not request.requires_gpu or snapshot.gpu_type is not None) and (
        request.gpu_type is None or request.gpu_type == snapshot.gpu_type
    )
    memory_safe = snapshot.gpu_type is None or (
        snapshot.gpu_memory_utilization is not None
        and snapshot.gpu_memory_utilization < policy.gpu_memory_limit
        and (
            snapshot.gpu_observed_at is None
            or 0 <= now - snapshot.gpu_observed_at <= policy.snapshot_ttl_seconds
        )
    )
    return (
        snapshot.healthy
        and snapshot.model == request.model
        and 0 <= age <= policy.snapshot_ttl_seconds
        and (
            snapshot.engine_observed_at is None
            or 0 <= now - snapshot.engine_observed_at <= policy.snapshot_ttl_seconds
        )
        and available
        and matches_gpu
        and memory_safe
    )


def score_candidate(
    candidate: Candidate,
    request: RoutingRequest,
    policy: RoutingPolicy,
    minimum_load: float,
    now: float,
) -> RoutingDecision:
    """Affinity may break a near-load tie but never bypass eligibility or create a hot replica."""
    affinity = (
        policy.mode == "adaptive"
        and request.prefix_digest is not None
        and request.prefix_digest in candidate.snapshot.cached_prefixes
        and candidate.load <= minimum_load + policy.cache_load_gap
    )
    score = candidate.load
    if policy.mode == "adaptive":
        memory = candidate.snapshot.gpu_memory_utilization or 0
        if candidate.snapshot.gpu_type is None:
            memory = 0
        score += policy.memory_weight * memory - (policy.cache_bonus if affinity else 0)
    return RoutingDecision(
        replica_id=candidate.snapshot.replica_id,
        policy=policy.mode,
        score=score,
        effective_requests=candidate.effective_requests,
        capacity=candidate.snapshot.capacity,
        cache_affinity_used=affinity,
        snapshot_age_seconds=now - candidate.snapshot.received_at,
    )
