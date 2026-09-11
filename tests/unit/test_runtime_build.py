"""Runtime evidence must bind actual immutable tool outputs to archived source and model bytes."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from test_model_assets import specification

from finserve.contracts.model_assets import ModelFetchSpec, ModelManifest, VerifiedFile
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.registry import engine_entrypoint
from finserve.registry.engine_entrypoint import VLLMParameters, engine_arguments
from finserve.registry.runtime_build import (
    ENGINE_BASE,
    RuntimeBuildSpec,
    RuntimeImage,
    bounded_document,
    build_runtime,
    expected_labels,
    extract_source,
    run_command,
    verify_image_inspection,
)


def manifest() -> ModelManifest:
    """Construct an explicitly synthetic source fixture with the actual SHA256 of its bytes."""
    source = specification()
    source = ModelFetchSpec.model_validate(
        {
            **source.model_dump(),
            "files": [
                {**source.files[0].model_dump(), "path": name}
                for name in (
                    "config.json",
                    "model.safetensors",
                    "tokenizer_config.json",
                    "tokenizer.json",
                )
            ],
        }
    )
    return ModelManifest(
        specification=source,
        files=tuple(
            VerifiedFile(source=item, sha256=hashlib.sha256(b"{}").hexdigest())
            for item in source.files
        ),
    )


def profile(model: ModelManifest) -> ServingProfileV1:
    """Both tokenizer and model point at the same verified snapshot for this serving contract."""
    return ServingProfileV1(
        engine="vllm",
        engine_version="0.29.0",
        engine_parameters_json=VLLMParameters().model_dump_json(),
        model_revision=model.specification.revision,
        tokenizer_revision=model.specification.revision,
        model_manifest_sha256=model.digest(),
        tokenizer_manifest_sha256=model.digest(),
        base_url="http://127.0.0.1:9010/v1",
        served_model="qwen",
    )


def test_structured_backend_is_pinned_and_changes_profile_identity() -> None:
    """Grammar selection is explicit runtime configuration; absent fields preserve old profiles."""
    model = manifest()
    original = profile(model)
    assert "structured_output_backend" not in VLLMParameters().model_dump()
    # Captured from source 65669c6; historical profiles must remain byte-identical.
    assert hashlib.sha256(VLLMParameters().model_dump_json().encode()).hexdigest() == (
        "a542107694ce8bb59bef72ab35f6c95a9245899108b42f06cddfd2bfb4518634"
    )
    changed = original.model_copy(
        update={
            "engine_parameters_json": VLLMParameters(
                structured_output_backend="xgrammar"
            ).model_dump_json()
        }
    )
    assert changed.digest() != original.digest()
    command = engine_arguments(changed, model, "0.29.0")
    assert json.loads(command[command.index("--structured-outputs-config") + 1]) == {
        "backend": "xgrammar",
        "disable_any_whitespace": False,
    }
    assert "--structured-output-backend" not in command
    assert "--structured-outputs-config" not in engine_arguments(original, model, "0.29.0")
    with pytest.raises(ValueError):
        VLLMParameters.model_validate({"structured_output_backend": "auto"})


def test_engine_parameters_and_model_identity_are_not_runtime_overrides() -> None:
    """Runtime command construction rejects unknown flags, package changes and unverified models."""
    model = manifest()
    serving = profile(model)
    command = engine_arguments(serving, model, "0.29.0")
    assert command[command.index("--model") + 1] == "/models"
    assert "--no-enable-prefix-caching" in command
    assert "--enforce-eager" in command
    assert "--trust-remote-code" not in command
    with pytest.raises(ValueError, match="identity"):
        engine_arguments(serving, model, "0.30.0")
    with pytest.raises(ValueError, match="identity"):
        engine_arguments(
            serving.model_copy(update={"model_manifest_sha256": "0" * 64}), model, "0.29.0"
        )
    for parameters in (
        '{"api_key":"x"}',
        '{"trust_remote_code":true}',
        '{"enforce_eager":"false"}',
    ):
        with pytest.raises(ValueError):
            engine_arguments(
                serving.model_copy(update={"engine_parameters_json": parameters}), model, "0.29.0"
            )


def add_tar(
    archive: tarfile.TarFile, name: str, content: bytes, kind: bytes = tarfile.REGTYPE
) -> None:
    """Explicit member types make traversal tests exercise archive extraction."""
    member = tarfile.TarInfo(name)
    member.type, member.size = kind, len(content)
    archive.addfile(member, io.BytesIO(content))


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../outside.py", tarfile.REGTYPE),
        ("/outside.py", tarfile.REGTYPE),
        ("a\\b.py", tarfile.REGTYPE),
        ("C:outside.py", tarfile.REGTYPE),
        ("link.py", tarfile.SYMTYPE),
        ("source./file.py", tarfile.REGTYPE),
        ("CON.py", tarfile.REGTYPE),
    ],
)
def test_source_archive_rejects_escape_and_links(tmp_path: Path, name: str, kind: bytes) -> None:
    """Archived source cannot create files outside its exclusive build context."""
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as archive:
        add_tar(archive, name, b"", kind)
    with pytest.raises(ValueError, match="path"):
        extract_source(source, tmp_path / "context", 1024)


def test_archive_and_metadata_byte_limits(tmp_path: Path) -> None:
    """Bound source extraction and metadata decoding before they enter build receipt validation."""
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as archive:
        add_tar(archive, "source.py", b"too large")
    with pytest.raises(ValueError, match="budget"):
        extract_source(source, tmp_path / "context", 1)
    source.write_bytes(b"x" * (8 * 1024**2 + 2))
    with pytest.raises(ValueError, match="budget"):
        extract_source(source, tmp_path / "context", 1)
    document = tmp_path / "metadata.json"
    document.write_text("[0,1]")
    with pytest.raises(ValueError, match="limit"):
        bounded_document(document, 2)


@pytest.mark.parametrize(
    "names", [("Foo.py", "foo.py"), ("Foo/x.py", "foo/y.py"), ("x.py", "x.py")]
)
def test_archive_case_collisions_are_rejected(tmp_path: Path, names: tuple[str, str]) -> None:
    """Extraction cannot overwrite files or alias differently spelled parent directories."""
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as archive:
        for name in names:
            add_tar(archive, name, b"")
    with pytest.raises(ValueError, match="paths|collisions"):
        extract_source(source, tmp_path / "context", 1024)


def test_archive_member_budget_is_checked_while_reading_headers(tmp_path: Path) -> None:
    """Many-header archives stop at the member limit before full materialization."""
    source = tmp_path / "source.tar"
    with tarfile.open(source, "w") as archive:
        for index in range(8193):
            add_tar(archive, f"file-{index}.py", b"")
    with pytest.raises(ValueError, match="file or byte budget"):
        extract_source(source, tmp_path / "context", 1024)


class DockerFixture:
    """Use real Git archives with synthetic Docker responses; no image build is claimed."""

    def __init__(self, model: ModelManifest, source_revision: str) -> None:
        """Bind fixture metadata to the exact inputs and retain argv for provenance assertions."""
        self.model, self.source_revision = model, source_revision
        self.calls: list[list[str]] = []
        self.wrong = False

    def __call__(self, arguments: list[str], directory: Path, output: Path, timeout: float) -> None:
        """Only Git executes; Docker branches return explicitly synthetic digests and labels."""
        self.calls.append(arguments)
        if arguments[0] == "git":
            run_command(arguments, directory, output, timeout)
        elif arguments[1:3] == ["buildx", "build"]:
            Path(arguments[arguments.index("--metadata-file") + 1]).write_text(
                json.dumps(
                    {
                        "containerimage.config.digest": "sha256:" + "b" * 64,
                        "containerimage.digest": "sha256:" + "c" * 64,
                    }
                )
            )
            output.write_text("synthetic build output")
        else:
            specification = RuntimeBuildSpec(
                source_revision=self.source_revision, model_manifest_sha256=self.model.digest()
            )
            labels = expected_labels(specification)
            if self.wrong:
                labels["finserve.model_manifest_sha256"] = "0" * 64
            output.write_text(
                json.dumps(
                    [
                        {
                            "Id": "sha256:" + "b" * 64,
                            "Os": "linux",
                            "Architecture": "amd64",
                            "Config": {"Labels": labels},
                        }
                    ]
                )
            )


def git_source(tmp_path: Path) -> tuple[Path, str]:
    """Create an isolated source fixture commit; the shared project checkout is never committed."""
    repository = tmp_path / "source"
    repository.mkdir()
    (repository / "infra/docker").mkdir(parents=True)
    (repository / "infra/docker/Dockerfile.engine").write_text("FROM " + ENGINE_BASE + "\n")
    (repository / "tracked.txt").write_text("committed")
    for arguments in (
        ["git", "init"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "Source fixture",
        ],
    ):
        subprocess.run(arguments, cwd=repository, check=True, capture_output=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    (repository / "tracked.txt").write_text("uncommitted modification")
    return repository, revision


def test_build_archives_exact_commit_and_checks_tool_identity(tmp_path: Path) -> None:
    """Dirty source cannot be relabelled; an inspected temporary tag resolves to immutable IDs."""
    repository, revision = git_source(tmp_path)
    model = manifest()
    frozen = RuntimeBuildSpec(source_revision=revision, model_manifest_sha256=model.digest())
    commands = DockerFixture(model, revision)
    receipt = build_runtime(repository, frozen, model, tmp_path / "build", command=commands)
    assert (tmp_path / "build/context/tracked.txt").read_text() == "committed"
    assert receipt.image_manifest_digest != receipt.image_config_digest
    assert commands.calls[-1][-1].startswith("finserve-build:")
    assert receipt.image_local_id == receipt.image_config_digest
    assert json.loads((tmp_path / "build/runtime-image.json").read_text()) == receipt.model_dump()
    commands.wrong = True
    with pytest.raises(ValueError, match="inspected"):
        build_runtime(repository, frozen, model, tmp_path / "wrong-build", command=commands)
    assert not (tmp_path / "wrong-build/runtime-image.json").exists()
    with pytest.raises(ValueError, match="model manifest"):
        build_runtime(
            repository,
            frozen.model_copy(update={"model_manifest_sha256": "0" * 64}),
            model,
            tmp_path / "wrong-input",
            command=commands,
        )


@pytest.mark.parametrize("document", [{}, [], [{}, {}]])
def test_inspection_requires_one_exact_image(document: Any) -> None:
    """Ambiguous inspection responses cannot become image provenance."""
    frozen = RuntimeBuildSpec(source_revision="a" * 40, model_manifest_sha256=manifest().digest())
    image = RuntimeImage(
        specification=frozen,
        source_archive_sha256="a" * 64,
        image_config_digest="sha256:" + "b" * 64,
        image_manifest_digest="sha256:" + "c" * 64,
        image_local_id="sha256:" + "b" * 64,
    )
    with pytest.raises(ValueError, match="exactly one"):
        verify_image_inspection(document, image)


@pytest.mark.parametrize("manifest_id", [False, True])
def test_legacy_and_containerd_image_identity_require_matching_digests(manifest_id: bool) -> None:
    """Docker29 manifest IDs and legacy config IDs remain bound to actual build metadata."""
    frozen = RuntimeBuildSpec(source_revision="a" * 40, model_manifest_sha256=manifest().digest())
    local_id = "sha256:" + ("c" if manifest_id else "b") * 64
    image = RuntimeImage(
        specification=frozen,
        source_archive_sha256="a" * 64,
        image_config_digest="sha256:" + "b" * 64,
        image_manifest_digest="sha256:" + "c" * 64,
        image_local_id=local_id,
    )
    actual = [
        {
            "Id": local_id,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {"Labels": expected_labels(frozen)},
            "Descriptor": {"digest": image.image_manifest_digest},
        }
    ]
    verify_image_inspection(actual, image)
    actual[0]["Descriptor"] = {"digest": "sha256:" + "0" * 64}
    with pytest.raises(ValueError, match="inspected"):
        verify_image_inspection(actual, image)
    with pytest.raises(ValueError, match="pinned"):
        RuntimeImage.model_validate({**image.model_dump(), "base_image": "vllm/vllm-openai:latest"})


def test_failed_command_retains_its_log(tmp_path: Path) -> None:
    """Failed commands retain private diagnostics and never become a build receipt."""
    output = tmp_path / "failure.log"
    with pytest.raises(RuntimeError, match="retained log"):
        run_command(
            [sys.executable, "-c", "print('fixture failure'); raise SystemExit(7)"],
            tmp_path,
            output,
            10,
        )
    assert "fixture failure" in output.read_text()


@pytest.mark.parametrize("verify_only", [False, True])
def test_entrypoint_verifies_real_files_before_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    verify_only: bool,
) -> None:
    """The entrypoint checks real fixture bytes before reporting or invoking an engine."""
    model, directory = manifest(), tmp_path / "models"
    directory.mkdir()
    for item in model.files:
        (directory / item.source.path).write_bytes(b"{}")
    baked, configured = tmp_path / "manifest.json", tmp_path / "profile.json"
    baked.write_text(model.canonical())
    serving = profile(model)
    configured.write_text(serving.canonical())
    monkeypatch.setattr(engine_entrypoint, "BAKED_MANIFEST", baked)
    monkeypatch.setattr(engine_entrypoint, "MODEL_DIRECTORY", directory)
    monkeypatch.setattr(engine_entrypoint, "RUNTIME_HOME", tmp_path / "home")

    def installed_version(_: str) -> str:
        """The synthetic runtime package matches the pinned engine version for this unit test."""
        return "0.29.0"

    monkeypatch.setattr(engine_entrypoint.importlib.metadata, "version", installed_version)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    launched: list[list[str]] = []

    def execute(_: str, arguments: list[str], environment: dict[str, str]) -> None:
        """Record the exact command without loading a GPU model in a unit test."""
        launched.append(arguments)
        assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"

    monkeypatch.setattr(engine_entrypoint.os, "execve", execute)
    arguments = [
        "--profile",
        str(configured),
        "--profile-sha256",
        serving.digest(),
        "--expected-model",
        serving.served_model,
        "--expected-base-url",
        serving.base_url,
    ] + (["--verify-only"] if verify_only else [])
    engine_entrypoint.run(arguments)
    if verify_only:
        observed = json.loads(capsys.readouterr().out)
        assert observed["profile_sha256"] == serving.digest()
        assert observed["model_manifest_sha256"] == model.digest()
        assert launched == []
    else:
        assert launched == [engine_arguments(serving, model, "0.29.0")]
    (directory / "model.safetensors").write_bytes(b"xx")
    with pytest.raises(ValueError, match="checksum"):
        engine_entrypoint.run(arguments)


@pytest.mark.parametrize(
    "option, value, error",
    [
        ("--profile-sha256", "f" * 64, "digest"),
        ("--expected-model", "other-model", "model"),
        ("--expected-base-url", "http://other:8000/v1", "endpoint"),
        ("--expected-credential-env", "FINSERVE_ENGINE_API_KEY", "credential"),
    ],
)
def test_launch_binding_fails_before_model_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str, value: str | None, error: str
) -> None:
    """Deployment mismatch must fail before touching the model volume or loading GPU packages."""
    model = manifest()
    baked, configured = tmp_path / "manifest.json", tmp_path / "profile.json"
    baked.write_text(model.canonical())
    configured.write_text(profile(model).canonical())
    monkeypatch.setattr(engine_entrypoint, "BAKED_MANIFEST", baked)
    monkeypatch.setattr(engine_entrypoint, "MODEL_DIRECTORY", tmp_path / "missing-model")
    arguments = ["--profile", str(configured), "--verify-only", option]
    if value is not None:
        arguments.append(value)
    with pytest.raises(ValueError, match=error):
        engine_entrypoint.run(arguments)


def test_entrypoint_rejects_incomplete_snapshot_and_oversized_identity(tmp_path: Path) -> None:
    """Metadata-only snapshots fail engine preflight; mounted identity files are bounded."""
    source = specification()
    partial = ModelManifest(
        specification=source,
        files=(VerifiedFile(source=source.files[0], sha256=hashlib.sha256(b"{}").hexdigest()),),
    )
    with pytest.raises(ValueError, match="complete"):
        engine_arguments(profile(partial), partial, "0.29.0")
    path = tmp_path / "identity.json"
    path.write_bytes(b"12345")
    with pytest.raises(ValueError, match="limit"):
        engine_entrypoint.read_bounded(path, 4)


def test_auth_and_compatibility_environment_are_owned_by_profile() -> None:
    """Missing auth fails closed; secrets stay out of argv and inherited vLLM flags cannot win."""
    serving = profile(manifest()).model_copy(update={"credential_env": "ENGINE_SECRET"})
    for values in ({}, {"ENGINE_SECRET": ""}, {"ENGINE_SECRET": "bad\ncredential"}):
        with pytest.raises(ValueError, match="credential"):
            engine_entrypoint.runtime_environment(serving, values)
    resolved = engine_entrypoint.runtime_environment(
        serving,
        {
            "ENGINE_SECRET": "test-secret",
            "PATH": "/bin",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
            "VLLM_API_KEY": "wrong",
            "VLLM_UNKNOWN_OVERRIDE": "bad",
        },
    )
    assert resolved["VLLM_API_KEY"] == "test-secret"
    assert resolved["VLLM_USE_V2_MODEL_RUNNER"] == resolved["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "VLLM_UNKNOWN_OVERRIDE" not in resolved and resolved["PATH"] == "/bin"
    assert "test-secret" not in " ".join(engine_arguments(serving, manifest(), "0.29.0"))
    public = engine_entrypoint.runtime_environment(profile(manifest()), {"VLLM_API_KEY": "unused"})
    assert "VLLM_API_KEY" not in public
    changed = serving.model_copy(
        update={
            "engine_parameters_json": VLLMParameters(
                use_v2_model_runner=True, use_flashinfer_sampler=True
            ).model_dump_json()
        }
    )
    explicit = engine_entrypoint.runtime_environment(changed, {"ENGINE_SECRET": "test-secret"})
    assert explicit["VLLM_USE_V2_MODEL_RUNNER"] == explicit["VLLM_USE_FLASHINFER_SAMPLER"] == "1"
    assert changed.digest() != serving.digest()
