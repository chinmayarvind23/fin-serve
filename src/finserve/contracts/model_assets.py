"""Immutable bounded model-file identities for producers, before runtime measurements exist."""

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from finserve.contracts.deployment import ImmutableModel


def safe_asset_path(value: str) -> str:
    """Portable data paths cannot select executable files or escape the model volume."""
    path = PurePosixPath(value)
    allowed = {".json", ".safetensors", ".model", ".txt", ".tiktoken"}
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if (
        not value
        or len(value) > 256
        or path.is_absolute()
        or str(path) != value
        or any(part.startswith(".") for part in path.parts)
        or any(part.endswith(".") for part in path.parts)
        or any(character in value for character in ("\\", ":", "\x00"))
        or path.suffix not in allowed
        or value == "finserve-model-manifest.json"
        or any(part.split(".")[0].casefold() in reserved for part in path.parts)
        or not re.fullmatch(
            r"[A-Za-z0-9_-][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*", value
        )
    ):
        raise ValueError("portable model data path required")
    return value


class SourceFile(ImmutableModel):
    """Hub Git blobs and LFS objects have different authoritative checksum algorithms."""

    path: str
    size_bytes: int = Field(ge=0, le=64 * 1024**3, strict=True)
    checksum_kind: Literal["git-blob-sha1", "sha256"]
    checksum: str

    @field_validator("path")
    @classmethod
    def data_path(cls, value: str) -> str:
        """Apply the same path rules before metadata planning and local publication."""
        return safe_asset_path(value)

    @model_validator(mode="after")
    def checksum_shape(self) -> Self:
        """Prevent a Git blob identity being treated as a raw-file SHA256 digest."""
        length = 40 if self.checksum_kind == "git-blob-sha1" else 64
        if len(self.checksum) != length or any(c not in "0123456789abcdef" for c in self.checksum):
            raise ValueError("checksum does not match algorithm")
        return self


class HubRevision(ImmutableModel):
    """Only an explicit namespace/repository and full commit may construct a Hub origin URL."""

    repository: str = Field(
        pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,95}$"
    )
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ModelFetchSpec(HubRevision):
    """Freeze selected source files and a total disk budget before downloading any weights."""

    schema_version: Literal["model-fetch-v1"] = "model-fetch-v1"
    files: tuple[SourceFile, ...] = Field(min_length=1, max_length=1024)
    maximum_bytes: int = Field(gt=0, le=128 * 1024**3, strict=True)

    @model_validator(mode="after")
    def bounded_unique_files(self) -> Self:
        """Case-fold collisions are rejected so the same snapshot works on Windows and Linux."""
        paths = [item.path.casefold() for item in self.files]
        if (
            len(set(paths)) != len(paths)
            or sum(item.size_bytes for item in self.files) > self.maximum_bytes
        ):
            raise ValueError("duplicate files or model byte budget exceeded")
        return self

    def canonical(self) -> str:
        """Source order is irrelevant; names and content identities define the frozen snapshot."""
        value = self.model_dump()
        value["files"] = [
            item.model_dump() for item in sorted(self.files, key=lambda item: item.path)
        ]
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self) -> str:
        """The fetch identity also names the exclusively published local model directory."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()


class VerifiedFile(ImmutableModel):
    """Every downloaded file gets a raw SHA256 regardless of its source checksum algorithm."""

    source: SourceFile
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def raw_checksum_agrees(self) -> Self:
        """An LFS object's expected raw digest cannot contradict its recorded verified digest."""
        if self.source.checksum_kind == "sha256" and self.source.checksum != self.sha256:
            raise ValueError("verified SHA256 differs from source identity")
        return self


class ModelManifest(ImmutableModel):
    """A small manifest can enter evidence CAS while large model files stay on a bounded volume."""

    schema_version: Literal["verified-model-manifest-v1"] = "verified-model-manifest-v1"
    specification: ModelFetchSpec
    files: tuple[VerifiedFile, ...]

    @model_validator(mode="after")
    def complete_snapshot(self) -> Self:
        """The result must account for exactly every frozen source file, including empty files."""
        expected = {item.path: item for item in self.specification.files}
        actual = {item.source.path: item.source for item in self.files}
        if expected != actual or len(actual) != len(self.files):
            raise ValueError("verified files do not match the frozen snapshot")
        return self

    def canonical(self) -> str:
        """Sort both source and verified lists for an ordering-independent profile identity."""
        value = self.model_dump()
        value["specification"] = json.loads(self.specification.canonical())
        value["files"] = [
            item.model_dump() for item in sorted(self.files, key=lambda item: item.source.path)
        ]
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def digest(self) -> str:
        """ServingProfileV1 records this digest after the producer verifies actual model bytes."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()
