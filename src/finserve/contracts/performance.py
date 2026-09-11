"""Canonical bounded performance inputs for a trusted managed-runtime producer."""

import hashlib
import json
from typing import Self

from pydantic import Field, model_validator

from finserve.benchmark.runner import RunConfig
from finserve.benchmark.workload import Workload
from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.contracts.serving_profile import ServingProfileV1


class PerformanceCollectionSpec(ImmutableModel):
    """One frozen workload/configuration produces evidence, not a caller-supplied pass flag."""

    collection_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    collector_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    profile: ServingProfileV1
    revision: Revision
    workload: Workload
    configuration: RunConfig
    timeout_seconds: float = Field(default=1800.0, gt=0, le=7200)
    maximum_raw_bytes: int = Field(default=512 * 1024**2, ge=1024, le=512 * 1024**2, strict=True)

    @model_validator(mode="after")
    def coherent_runtime(self) -> Self:
        """Bound retained workload/population and reject runtime identity drift before requests."""
        self.profile.verify_revision(self.revision)
        self.configuration.require_constraint_runtime()
        if self.configuration.model != self.profile.served_model:
            raise ValueError("performance model differs from serving profile")
        for field, attribute in (
            ("revision", "source_revision"),
            ("model_revision", "model_revision"),
            ("tokenizer_revision", "tokenizer_revision"),
            ("engine", "engine"),
            ("engine_config", "engine_config"),
            ("image_digest", "image_digest"),
            ("config_digest", "config_digest"),
        ):
            if getattr(self.configuration, field) != getattr(self.revision, attribute):
                raise ValueError("performance configuration differs from runtime revision")
        if (
            len(self.workload.items) > 1024
            or len(self.workload.model_dump_json().encode()) > 1024**2
            or self.configuration.requests + self.configuration.warmup > 65536
            or self.configuration.concurrency > 128
        ):
            raise ValueError("performance collection exceeds bounded local tier")
        if self.configuration.output_constraints is not None:
            self.configuration.output_constraints.require_prompts(
                item.prompt for item in self.workload.items
            )
        return self

    def canonical(self) -> str:
        """Normalize nested numeric defaults before freezing stage identity."""
        validated = PerformanceCollectionSpec.model_validate_json(self.model_dump_json())
        return json.dumps(validated.model_dump(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """Bind endpoint, runtime, workload, mapping and collection limits to immutable bytes."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def endpoint(self) -> str:
        """Only the server-owned serving profile selects the request destination."""
        suffix = "/chat/completions" if self.configuration.request_api == "chat" else "/completions"
        return self.profile.base_url + suffix
