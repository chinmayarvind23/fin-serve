"""Freeze caller-declared output shapes independently of suite answers and reporting labels."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from finserve.contracts.output_constraint import OutputConstraint, unique_fields

MAXIMUM_MAP_BYTES = 262144


def prompt_digest(prompt: str) -> str:
    """Bind original UTF-8 bytes before chat-role mapping, with no trimming or normalization."""
    encoded = prompt.encode("utf-8")
    if not 1 <= len(encoded) <= 131072:
        raise ValueError("constrained prompt exceeds byte budget")
    return hashlib.sha256(encoded).hexdigest()


class RequestConstraintBinding(BaseModel):
    """A prompt has one explicit requested shape, including an explicitly unconstrained choice."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    constraint: OutputConstraint | None


class RequestConstraintMap(BaseModel):
    """Share the complete map across collectors; missing entries never fall back."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["prompt-output-contracts-v1"] = "prompt-output-contracts-v1"
    entries: tuple[RequestConstraintBinding, ...] = Field(min_length=1, max_length=256)

    @field_validator("entries")
    @classmethod
    def canonical_entries(
        cls, entries: tuple[RequestConstraintBinding, ...]
    ) -> tuple[RequestConstraintBinding, ...]:
        """Ordering has no request meaning; identical duplicate bindings still fail."""
        if len({entry.prompt_sha256 for entry in entries}) != len(entries):
            raise ValueError("duplicate prompt constraint binding")
        return tuple(sorted(entries, key=lambda entry: entry.prompt_sha256))

    @model_validator(mode="after")
    def bounded(self) -> Self:
        """Keep the full frozen map small enough for producer and benchmark artifact envelopes."""
        if len(self.model_dump_json().encode()) > MAXIMUM_MAP_BYTES:
            raise ValueError("constraint map exceeds byte budget")
        return self

    def digest(self) -> str:
        """Include every declared binding and schema version, even entries unused by one cohort."""
        encoded = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def resolve(self, prompt: str) -> OutputConstraint | None:
        """Use prompt identity without case IDs, families, evaluator kinds or expected answers."""
        identity = prompt_digest(prompt)
        for entry in self.entries:
            if entry.prompt_sha256 == identity:
                return entry.constraint
        raise ValueError("missing prompt constraint binding")

    def require_prompts(self, prompts: Iterable[str]) -> None:
        """Preflight the whole population before offering even a warmup request."""
        for prompt in prompts:
            self.resolve(prompt)


def read_constraint_map(path: Path) -> RequestConstraintMap:
    """Load a bounded caller sidecar, rejecting duplicate JSON keys before typed validation."""
    with path.open("rb") as handle:
        raw = handle.read(MAXIMUM_MAP_BYTES + 1)
    if len(raw) > MAXIMUM_MAP_BYTES:
        raise ValueError("constraint map exceeds byte budget")
    return RequestConstraintMap.model_validate(json.loads(raw, object_pairs_hook=unique_fields))
