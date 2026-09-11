"""Bounded single-image contracts shared by durable jobs and the optional binary worker."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from finserve.multimodal.jax_generator import RGBImage

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
JobState = Literal["queued", "running", "succeeded", "failed", "cancel_requested", "cancelled"]


class VisualJobRequest(BaseModel):
    """Admit one bounded image; arbitrary media URLs and execution destinations are excluded."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)
    image: RGBImage
    model_revision: Identifier
    seed: int = Field(default=17, ge=0, lt=2**32, strict=True)
    timeout_seconds: float = Field(default=30, ge=0.05, le=120, allow_inf_nan=False)


class VisualAttempt(BaseModel):
    """Fence each job execution; worker identities must never be reused after process restart."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: Identifier
    generation: int = Field(ge=1, strict=True)
    request: VisualJobRequest


class VisualArtifact(BaseModel):
    """Keep a small PNG in-band with content identity and measured CPU stage durations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    png: bytes = Field(min_length=1, max_length=131072)
    sha256: Digest
    model_sha256: Digest
    initialization_ns: int = Field(ge=0)
    generation_ns: int = Field(ge=0)
    rendering_ns: int = Field(ge=0)


class VisualJob(BaseModel):
    """Expose tenant-scoped status without exposing stored image bytes or internal credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: Identifier
    state: JobState
    generation: int = Field(ge=0)
    model_revision: Identifier
    artifact_sha256: Digest | None = None
    failure_type: str | None = None
    created_at: float
    updated_at: float
