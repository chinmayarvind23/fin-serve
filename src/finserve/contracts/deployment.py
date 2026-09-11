"""Immutable deployment identities and durable control-plane state contracts."""

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImmutableModel(BaseModel):
    """Control-plane inputs reject unknown fields and nonfinite numeric values."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Revision(ImmutableModel):
    """Rollback names an exact image/model/tokenizer/config, never a mutable deployment alias."""

    revision_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    model_revision: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    source_revision: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    engine: str = Field(min_length=1)
    engine_config: str = Field(min_length=1)

    @model_validator(mode="after")
    def pinned_identity(self) -> "Revision":
        """Reject common mutable labels; the registry separately forbids changing an existing ID."""
        for value in (self.model_revision, self.tokenizer_revision, self.source_revision):
            if value.lower() in {"latest", "main", "master", "undeclared"}:
                raise ValueError("revision identities must be pinned")
        return self

    def digest(self) -> str:
        """Fingerprint every serving identity field, including engine configuration."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class HealthObservation(ImmutableModel):
    """Readiness and a smoke check must identify the exact immutable revision they tested."""

    revision_id: str
    revision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    ready: bool = Field(strict=True)
    smoke_passed: bool = Field(strict=True)

    def verifies(self, revision: Revision) -> bool:
        """A healthy load balancer serving a different revision cannot prove restoration."""
        return (
            self.ready
            and self.smoke_passed
            and self.revision_id == revision.revision_id
            and self.revision_digest == revision.digest()
        )


class DeploymentState(ImmutableModel):
    """Generation fences old detector signals; rollback ownership freezes concurrent promotion."""

    deployment_id: str
    active_revision: str
    known_good_revision: str
    generation: int = Field(ge=0)
    rollback_id: str | None = None


class RegressionSignal(ImmutableModel):
    """A detector must include the deployment generation and revision actually observed."""

    signal_id: str = Field(min_length=1, max_length=128)
    deployment_id: str
    observed_revision: str
    observed_generation: int = Field(ge=0)
    detected_at: float = Field(gt=0)
    clock_domain: Literal["utc_unix_seconds"] = "utc_unix_seconds"
    detector: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=1000)


RollbackStatus = Literal["detected", "applying", "verifying", "needs_reconciliation", "restored"]


class RollbackRecord(ImmutableModel):
    """Keep trusted detector UTC time separate from controller receipt, start and recovery."""

    operation_id: str
    signal: RegressionSignal
    target_revision: str
    target_digest: str
    status: RollbackStatus = "detected"
    detected_at: float = Field(gt=0)
    received_at: float = Field(gt=0)
    started_at: float | None = Field(default=None, gt=0)
    restored_at: float | None = Field(default=None, gt=0)
    last_error: str | None = None
    apply_attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def state_timing(self) -> "RollbackRecord":
        """Never emit a rollback duration without a real start and verified restoration."""
        if self.detected_at != self.signal.detected_at or self.received_at < self.detected_at:
            raise ValueError("detector timestamp mismatch or future detection")
        if self.started_at is not None and self.started_at < self.received_at:
            raise ValueError("rollback clock moved backward before start")
        if self.status == "restored":
            if self.started_at is None or self.restored_at is None:
                raise ValueError("restoration requires start and verified-health timestamps")
            if self.restored_at < self.started_at:
                raise ValueError("rollback clock moved backward before recovery")
        elif self.restored_at is not None:
            raise ValueError("unverified state cannot contain recovery timestamp")
        return self

    def duration_seconds(self) -> float | None:
        """Detector-to-health is the reported interval, including controller and action delay."""
        return None if self.restored_at is None else self.restored_at - self.detected_at
