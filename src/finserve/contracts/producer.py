"""Frozen inputs for trusted offline producers before any model output is observed."""

import hashlib
import json
from typing import Self

from pydantic import Field, model_validator

from finserve.benchmark.runner import RunConfig
from finserve.contracts.deployment import ImmutableModel, Revision
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.evaluation.quality import GoldenSuite


class QualityCollectionSpec(ImmutableModel):
    """A collector binds requests to an existing runtime; orchestration verifies that runtime."""

    collection_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    profile: ServingProfileV1
    revision: Revision
    suite: GoldenSuite
    configuration: RunConfig
    max_tokens: int = Field(default=128, ge=1, le=2048, strict=True)
    maximum_raw_bytes: int = Field(default=64 * 1024**2, ge=1024, le=64 * 1024**2, strict=True)

    @model_validator(mode="after")
    def coherent_runtime(self) -> Self:
        """Reject identity drift and unbounded suites before creating output or issuing requests."""
        self.profile.verify_revision(self.revision)
        if self.configuration.model != self.profile.served_model:
            raise ValueError("collector model differs from serving profile")
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
                raise ValueError("collector configuration differs from runtime revision")
        if len(self.suite.cases) > 1024 or len(self.suite.model_dump_json().encode()) > 1024**2:
            raise ValueError("quality suite exceeds collection budget")
        return self

    def canonical(self) -> str:
        """Stable bytes freeze suite, mapping and runtime identities for durable stage receipts."""
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """A repeated collection ID cannot silently acquire changed prompts or runtime settings."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def endpoint(self) -> str:
        """Destination comes from the trusted profile, not a separate task or client URL."""
        suffix = "/chat/completions" if self.configuration.request_api == "chat" else "/completions"
        return self.profile.base_url + suffix
