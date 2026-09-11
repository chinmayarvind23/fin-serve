"""Versioned configuration identity shared by measurement, gates and warm traffic routing."""

import hashlib
import json
from typing import Any, Literal, Self

import httpx
from pydantic import Field, field_validator, model_validator

from finserve.contracts.deployment import ImmutableModel, Revision


def unique_parameters(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate configuration keys and literal credentials instead of hiding overrides."""
    result: dict[str, Any] = {}
    secret_keys = {"api_key", "password", "secret", "token", "hf_token", "access_key"}
    for key, value in pairs:
        if key in result or key.lower().replace("-", "_") in secret_keys:
            raise ValueError("duplicate parameter or literal credential field")
        result[key] = value
    return result


def canonical_parameters(value: str) -> str:
    """Keep nested parameters immutable as canonical finite JSON, independent of key order."""
    try:
        parameters = json.loads(value, object_pairs_hook=unique_parameters)
        if not isinstance(parameters, dict):
            raise ValueError("engine parameters must be an object")
        encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded) > 65536:
            raise ValueError("canonical parameter object exceeds size limit")
        return encoded
    except (ValueError, TypeError, RecursionError):
        raise ValueError("invalid engine parameter object") from None


def validate_backend_url(value: str) -> str:
    """Immutable endpoints cannot embed credential bytes, query overrides or ambiguous paths."""
    address = httpx.URL(value)
    if (
        address.scheme not in {"http", "https"}
        or not address.host
        or address.username
        or address.password
        or address.query
        or address.fragment
        or address.path != "/v1"
        or str(address) != value
    ):
        raise ValueError("canonical HTTP(S) /v1 endpoint without credentials required")
    return value


class ServingProfileV1(ImmutableModel):
    """Image digest stays in Revision to avoid an image/profile identity hashing cycle."""

    schema_version: Literal["serving-profile-v1"] = "serving-profile-v1"
    engine: str = Field(min_length=1, max_length=128)
    engine_version: str = Field(pattern=r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?[a-zA-Z0-9.+-]*$")
    engine_parameters_json: str = Field(default="{}", max_length=65536)
    model_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    tokenizer_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    model_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_url: str
    served_model: str = Field(min_length=1, max_length=256)
    credential_env: str | None = Field(default=None, pattern=r"^[A-Z_][A-Z0-9_]{0,127}$")

    @field_validator("engine_parameters_json")
    @classmethod
    def freeze_parameters(cls, value: str) -> str:
        """Canonicalize before any digest is computed so task JSON round-trips are stable."""
        return canonical_parameters(value)

    @model_validator(mode="after")
    def endpoint_is_fixed(self) -> Self:
        """Share endpoint constraints with the warm router rather than accepting client URLs."""
        validate_backend_url(self.base_url)
        return self

    def canonical(self) -> str:
        """Include schema version in the stable byte representation to permit explicit evolution."""
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self) -> str:
        """Measurement and routing both use this complete configuration digest."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()

    def verify_revision(self, revision: Revision) -> None:
        """Check model, tokenizer and engine parameters alongside the unified config digest."""
        if (
            revision.config_digest != self.digest()
            or revision.engine != self.engine
            or revision.engine_config != self.engine_parameters_json
            or revision.model_revision != self.model_revision
            or revision.tokenizer_revision != self.tokenizer_revision
        ):
            raise ValueError("revision differs from canonical serving profile")
