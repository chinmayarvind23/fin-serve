"""Frozen local engine launches bind actual image, model, endpoint and resource identities."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import httpx
from pydantic import Field, model_validator

from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.contracts.model_assets import ModelFetchSpec
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.registry.engine_entrypoint import VLLMParameters
from finserve.registry.runtime_build import RuntimeImage


class RuntimeLaunchSpec(ImmutableModel):
    """This trusted local executor exposes only a loopback port and owned read-only model files."""

    schema_version: Literal["managed-runtime-v1"] = "managed-runtime-v1"
    image: RuntimeImage
    profile: ServingProfileV1
    revision: Revision
    model: ModelFetchSpec
    model_directory: Path
    memory_mib: int = Field(default=8192, ge=4096, le=32768, strict=True)
    pids_limit: int = Field(default=512, ge=128, le=2048, strict=True)
    readiness_timeout_seconds: float = Field(default=360.0, gt=0, le=900)
    shutdown_timeout_seconds: int = Field(default=30, ge=1, le=60, strict=True)

    @model_validator(mode="after")
    def exact_runtime(self) -> Self:
        """Reject image/profile drift and destinations outside the local managed engine boundary."""
        self.profile.verify_revision(self.revision)
        address = httpx.URL(self.profile.base_url)
        if (
            address.scheme != "http"
            or address.host != "127.0.0.1"
            or address.port is None
            or not 1024 <= address.port <= 65535
            or self.profile.credential_env is not None
        ):
            raise ValueError("local managed runtime requires an unauthenticated loopback endpoint")
        if (
            self.profile.engine != "vllm"
            or self.profile.engine_version != "0.29.0"
            or self.revision.image_digest != self.image.image_manifest_digest
            or self.revision.source_revision != self.image.specification.source_revision
            or self.profile.model_manifest_sha256 != self.image.specification.model_manifest_sha256
            or self.model.revision != self.profile.model_revision
            or self.profile.tokenizer_revision != self.model.revision
            or self.profile.tokenizer_manifest_sha256 != self.profile.model_manifest_sha256
            or not self.model_directory.is_absolute()
            or any(character in str(self.model_directory) for character in (",", '"', "\n", "\r"))
        ):
            raise ValueError("managed runtime identity differs from verified producer inputs")
        VLLMParameters.model_validate_json(self.profile.engine_parameters_json)
        return self

    def canonical(self) -> str:
        """Normalize nested numeric defaults before hashing a durable launch identity."""
        value = RuntimeLaunchSpec.model_validate_json(self.model_dump_json())
        return json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Container labels bind the full launch, including local model and resource boundaries."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()


class RuntimeReceipt(ImmutableModel):
    """Readiness is a real endpoint observation tied to one immutable Docker container start."""

    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    container_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    container_started_at: str = Field(min_length=1)
    observed_at: float = Field(gt=0)
    elapsed_seconds: float = Field(ge=0)
    output_directory: Path
