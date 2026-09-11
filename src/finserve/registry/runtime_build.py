"""Build archived source with a verified model manifest and retain actual Docker identities."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from pydantic import Field, field_validator

from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.model_assets import ModelManifest
from finserve.registry.model_assets import external_root

ENGINE_BASE = (
    "mirror.gcr.io/vllm/vllm-openai:v0.29.0@"
    "sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b"
)


class RuntimeBuildSpec(ImmutableModel):
    """Source and model inputs are fixed before a builder can pull, build or label an image."""

    schema_version: Literal["runtime-build-v1"] = "runtime-build-v1"
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    platform: Literal["linux/amd64"] = "linux/amd64"
    maximum_source_bytes: int = Field(default=64 * 1024**2, gt=0, le=512 * 1024**2, strict=True)
    timeout_seconds: float = Field(default=3600, gt=0, le=14400, allow_inf_nan=False)


class RuntimeImage(ImmutableModel):
    """Local Docker config identity and OCI manifest identity are separate recorded values."""

    specification: RuntimeBuildSpec
    source_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    image_manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    image_local_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    base_image: str = ENGINE_BASE

    @field_validator("base_image")
    @classmethod
    def fixed_base(cls, value: str) -> str:
        """A build receipt cannot substitute a mutable tag or a different engine image."""
        if value != ENGINE_BASE:
            raise ValueError("pinned official engine manifest required")
        return value


class CommandRunner(Protocol):
    """Production commands retain private logs; tests can inspect exact argv without Docker."""

    def __call__(self, arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Execute one finite command with an exclusive stdout/stderr evidence file."""
        ...


def run_command(arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
    """Pass an argv array without shell expansion and retain failed command output for review."""
    with output.open("xb") as destination:
        result = subprocess.run(
            arguments,
            cwd=directory,
            stdout=destination,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError("producer command failed; inspect its retained log")


def bounded_document(path: Path, maximum_bytes: int = 4 * 1024**2) -> Any:
    """Untrusted tool JSON is bounded before decoding; errors never echo command output."""
    with path.open("rb") as source:
        value = source.read(maximum_bytes + 1)
    if len(value) > maximum_bytes:
        raise ValueError("producer metadata exceeds byte limit")
    return json.loads(value)


def validate_archive_member(member: tarfile.TarInfo, spellings: dict[str, str]) -> None:
    """Reject linked paths and Windows aliases, including conflicting parent-directory casing."""
    name = PurePosixPath(member.name)
    reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
    if (
        name.is_absolute()
        or ".." in name.parts
        or "\\" in member.name
        or ":" in member.name
        or any(
            part.endswith((".", " ")) or part.split(".")[0].casefold() in reserved
            for part in name.parts
        )
        or any(character in member.name for character in '<>"?*')
        or any(ord(character) < 32 for character in member.name)
        or not (member.isfile() or member.isdir())
    ):
        raise ValueError("source archive contains a nonportable or linked path")
    for count in range(1, len(name.parts) + 1):
        prefix = "/".join(name.parts[:count])
        key = prefix.casefold()
        if key in spellings and spellings[key] != prefix:
            raise ValueError("source archive contains case-fold path collisions")
        spellings[key] = prefix


def extract_source(archive: Path, destination: Path, maximum_bytes: int) -> str:
    """A Git archive may contain only bounded regular source files and portable directories."""
    if archive.stat().st_size > maximum_bytes + 8 * 1024**2:
        raise ValueError("source archive exceeds byte budget")
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(1024**2):
            digest.update(chunk)
    with tarfile.open(archive, "r:") as source:
        members: list[tarfile.TarInfo] = []
        spellings: dict[str, str] = {}
        seen: set[str] = set()
        total_bytes = 0
        for member in source:
            total_bytes += member.size
            if len(members) >= 8192 or total_bytes > maximum_bytes:
                raise ValueError("source archive exceeds file or byte budget")
            validate_archive_member(member, spellings)
            key = str(PurePosixPath(member.name)).casefold()
            if key in seen:
                raise ValueError("source archive contains duplicate member paths")
            seen.add(key)
            members.append(member)
        source.extractall(destination, members=members, filter="data")
    return digest.hexdigest()


def expected_labels(specification: RuntimeBuildSpec) -> dict[str, str]:
    """Image inspection must observe the same source/model identities supplied to the build."""
    return {
        "org.opencontainers.image.revision": specification.source_revision,
        "finserve.model_manifest_sha256": specification.model_manifest_sha256,
        "finserve.engine": "vllm",
        "finserve.engine_version": "0.29.0",
    }


def verify_image_inspection(document: Any, image: RuntimeImage) -> None:
    """A mutable tag or fabricated metadata record cannot substitute a different local image."""
    if not isinstance(document, list):
        raise ValueError("exactly one inspected image required")
    items = cast(list[dict[str, Any]], document)
    if len(items) != 1:
        raise ValueError("exactly one inspected image required")
    actual = items[0]
    labels = actual["Config"]["Labels"]
    if (
        actual["Id"] != image.image_local_id
        or image.image_local_id not in {image.image_config_digest, image.image_manifest_digest}
        or (
            actual.get("Descriptor") is not None
            and actual["Descriptor"]["digest"] != image.image_manifest_digest
        )
        or actual["Os"] != "linux"
        or actual["Architecture"] != "amd64"
        or any(
            labels.get(key) != value for key, value in expected_labels(image.specification).items()
        )
    ):
        raise ValueError("inspected image differs from build evidence")


def build_runtime(
    repository: Path,
    specification: RuntimeBuildSpec,
    manifest: ModelManifest,
    output: Path,
    *,
    command: CommandRunner = run_command,
) -> RuntimeImage:
    """Build an immutable source archive; uncommitted checkout changes never acquire its label."""
    specification = RuntimeBuildSpec.model_validate_json(specification.model_dump_json())
    manifest = ModelManifest.model_validate_json(manifest.model_dump_json())
    if manifest.digest() != specification.model_manifest_sha256:
        raise ValueError("model manifest differs from frozen build input")
    workspace = external_root(output.parent) / output.name
    workspace.mkdir(exist_ok=False)
    archive = workspace / "source.tar"
    command(
        [
            "git",
            "archive",
            "--format=tar",
            "--output=" + str(archive),
            specification.source_revision,
        ],
        repository.resolve(),
        workspace / "archive.log",
        60,
    )
    context = workspace / "context"
    context.mkdir()
    archive_digest = extract_source(archive, context, specification.maximum_source_bytes)
    dockerfile = context / "infra/docker/Dockerfile.engine"
    if f"FROM {ENGINE_BASE}" not in dockerfile.read_text().splitlines():
        raise ValueError("archived engine Dockerfile does not pin the expected official base")
    generated = context / "producer"
    generated.mkdir(exist_ok=False)
    (generated / "model-manifest.json").write_text(manifest.canonical())
    metadata = workspace / "build-metadata.json"
    local_tag = "finserve-build:" + uuid4().hex
    command(
        [
            "docker",
            "buildx",
            "build",
            "--load",
            "--platform",
            specification.platform,
            "--provenance=false",
            "--tag",
            local_tag,
            "--metadata-file",
            str(metadata),
            "--build-arg",
            "SOURCE_REVISION=" + specification.source_revision,
            "--build-arg",
            "MODEL_MANIFEST_SHA256=" + manifest.digest(),
            "--file",
            str(dockerfile),
            str(context),
        ],
        workspace,
        workspace / "build.log",
        specification.timeout_seconds,
    )
    actual = bounded_document(metadata)
    inspection = workspace / "image-inspection.json"
    command(["docker", "image", "inspect", local_tag], workspace, inspection, 60)
    inspected = bounded_document(inspection)
    image = RuntimeImage(
        specification=specification,
        source_archive_sha256=archive_digest,
        image_config_digest=actual["containerimage.config.digest"],
        image_manifest_digest=actual["containerimage.digest"],
        image_local_id=inspected[0]["Id"],
    )
    verify_image_inspection(inspected, image)
    (workspace / "runtime-image.json").write_text(image.model_dump_json(indent=2))
    return image
