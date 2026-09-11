"""Bounded routing telemetry with explicit reservation reconciliation semantics."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
PrefixDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ReplicaSnapshot(BaseModel):
    """Stamp received_at on the router clock; remote monotonic clocks are not comparable.

    ongoing_requests includes reflected_lease_ids. Locally reserved leases absent from
    that set are added conservatively until worker telemetry acknowledges them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    replica_id: Identifier
    model: Identifier
    received_at: float = Field(ge=0)
    healthy: bool = True
    capacity: int = Field(ge=0, le=4096, strict=True)
    ongoing_requests: int = Field(default=0, ge=0, strict=True)
    queued_requests: int = Field(default=0, ge=0, strict=True)
    gpu_type: Identifier | None = None
    gpu_device_id: Identifier | None = None
    gpu_memory_utilization: float | None = Field(default=None, ge=0, le=1)
    gpu_observed_at: float | None = Field(default=None, ge=0)
    engine_running_requests: int | None = Field(default=None, ge=0, le=65536, strict=True)
    engine_waiting_requests: int | None = Field(default=None, ge=0, le=65536, strict=True)
    kv_cache_utilization: float | None = Field(default=None, ge=0, le=1)
    engine_observed_at: float | None = Field(default=None, ge=0)
    cached_prefixes: frozenset[PrefixDigest] = Field(default_factory=frozenset, max_length=128)
    reflected_lease_ids: frozenset[Identifier] = Field(default_factory=frozenset, max_length=4096)

    @model_validator(mode="after")
    def validate_reflected_leases(self) -> Self:
        """A telemetry count smaller than its acknowledged leases cannot support safe admission."""
        if len(self.reflected_lease_ids) > self.ongoing_requests:
            raise ValueError("reflected leases exceed the reported ongoing request count")
        return self


class RoutingRequest(BaseModel):
    """Route on serving requirements and hashed affinity, never finance-domain content."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: Identifier
    requires_gpu: bool = False
    gpu_type: Identifier | None = None
    prefix_digest: PrefixDigest | None = None


class RoutingDecision(BaseModel):
    """Keep enough policy evidence to explain selection without retaining prompt text."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    replica_id: Identifier
    policy: Literal["least_load", "adaptive"]
    score: float
    effective_requests: int = Field(ge=0)
    capacity: int = Field(gt=0)
    cache_affinity_used: bool
    snapshot_age_seconds: float = Field(ge=0)
