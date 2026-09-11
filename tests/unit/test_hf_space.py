"""Space packaging and local process ownership are independent of hosted credentials."""

import asyncio
import importlib.util
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest


def module(name: str) -> ModuleType:
    """Load deployment helpers directly, without changing the installed inference package."""
    path = Path(__file__).resolve().parents[2] / "infra/huggingface" / f"{name}.py"
    spec = importlib.util.spec_from_file_location("space_" + name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("first,second", [("", ""), ("a" * 16, "a" * 16), ("x\n" * 16, "b" * 16)])
def test_secret_validation_precedes_data_creation(
    monkeypatch: pytest.MonkeyPatch, first: str, second: str
) -> None:
    """Missing, reused or header-invalid secrets stop startup without revealing their values."""
    monkeypatch.setenv("FINSERVE_API_KEY", first)
    monkeypatch.setenv("FINSERVE_WEB_KEY", second)
    with pytest.raises(ValueError, match="two distinct printable"):
        module("launch").checked_environment()


def test_owned_children_are_reaped() -> None:
    """Two real CPU processes exit before the supervisor returns its cleanup acknowledgement."""
    children = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) for _ in range(2)
    ]
    try:
        module("launch").stop_children(children, grace_seconds=0.2)
        assert all(child.poll() is not None for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_ambient_worker_and_reload_settings_are_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Operator environment cannot silently change the direct-child ownership contract."""
    launch = module("launch")
    monkeypatch.setenv("FINSERVE_API_KEY", "a" * 16)
    monkeypatch.setenv("FINSERVE_WEB_KEY", "b" * 16)
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    monkeypatch.setenv("UVICORN_RELOAD", "true")

    def local_path(_path: str) -> Path:
        """Redirect only the fixture data-directory creation outside the source repository."""
        return tmp_path / "artifacts"

    monkeypatch.setattr(launch, "Path", local_path)
    environment = launch.checked_environment()
    assert "WEB_CONCURRENCY" not in environment and "UVICORN_RELOAD" not in environment


def test_cleanup_failure_does_not_skip_other_child() -> None:
    """An unexpected signal failure is reported only after every known child is attempted."""
    calls: list[str] = []

    class Child:
        """Minimal subprocess stand-in injects an error that cannot be reproduced portably."""

        def __init__(self, name: str) -> None:
            """Track one distinct cleanup target."""
            self.name = name

        def poll(self) -> None:
            """Report an active child until the test's simulated reap."""
            return None

        def terminate(self) -> None:
            """Fail only the first target while recording both signal attempts."""
            calls.append(self.name)
            if self.name == "first":
                raise PermissionError("injected")

        def wait(self, *, timeout: float) -> int:
            """Simulate completed reaping without an additional native process."""
            return 0

    with pytest.raises(RuntimeError, match="cleanup"):
        module("launch").stop_children([Child("first"), Child("second")])
    assert calls == ["first", "second"]


def test_bundle_is_exclusive_external_and_cannot_copy_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Path checks run before copying data, and an existing output is never merged."""
    pack = module("package")
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(ValueError, match="outside"):
        pack.package(repository, repository / "bundle")
    (repository / "source.py").write_text("safe")

    def fake_git(_repository: Path, *arguments: str) -> bytes:
        """Provide only the two metadata reads required after fixed test source selection."""
        return b"a" * 40 if arguments[0] == "rev-parse" else b""

    monkeypatch.setattr(pack, "git", fake_git)
    # Supply the two required root aliases without broadening source selection.
    selected: set[str] = {
        "source.py",
        "infra/huggingface/Dockerfile",
        "infra/huggingface/README.md",
    }

    def fake_selected(_repository: Path) -> set[str]:
        """Return the narrow fixture set without involving the working repository."""
        return selected

    monkeypatch.setattr(pack, "selected_files", fake_selected)
    for name in selected - {"source.py"}:
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("public")
    output = tmp_path / "bundle"
    pack.package(repository, output)
    with pytest.raises(FileExistsError):
        pack.package(repository, output)
    assert not (output / ".git").exists()
    assert (output / "space-package.json").exists()
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    (repository / "source.py").unlink()
    try:
        (repository / "source.py").symlink_to(outside)
    except OSError:
        pytest.skip("platform does not permit fixture symlinks")
    with pytest.raises(ValueError, match="regular file"):
        pack.package(repository, tmp_path / "rejected")


def test_only_tracked_sources_and_named_public_files_are_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Private evidence, env files and untracked modules cannot enter the upload bundle."""
    pack = module("package")
    tracked: set[str] = (
        pack.ROOT_FILES | pack.PUBLIC_FILES | {"src/runtime.py", "evidence/raw.json", ".env"}
    )
    for name in pack.PUBLIC_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"public\r\n" if name.endswith(".json") else b"public\n")

    def fake_git(_repository: Path, *arguments: Any) -> bytes:
        """Model Git's LF normalization for reviewed text assets only."""
        return "\0".join(tracked).encode() if arguments[0] == "ls-files" else b"public\n"

    monkeypatch.setattr(pack, "git", fake_git)
    selected = pack.selected_files(tmp_path)
    assert "src/runtime.py" in selected
    assert not {"evidence/raw.json", ".env", "src/untracked.py"} & selected
    (tmp_path / sorted(pack.PUBLIC_FILES)[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="aggregate differs"):
        pack.selected_files(tmp_path)


@pytest.mark.asyncio
async def test_probe_never_follows_redirect_and_drains_slow_body() -> None:
    """A redirect cannot forward credentials, and a trickling body cannot extend startup time."""
    calls: list[str] = []

    def redirect(request: httpx.Request) -> httpx.Response:
        """Observe the sole trusted request and return an untrusted redirect destination."""
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://untrusted.invalid/"})

    assert not await module("launch").ready("a" * 16, 1, transport=httpx.MockTransport(redirect))
    assert calls == ["http://127.0.0.1:8050/graphql"]
    closed = False

    class SlowBody(httpx.AsyncByteStream):
        """Delay every body chunk beyond the total allowed probe interval."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            """Yield only if the deadline fails to cancel the pending read."""
            await asyncio.sleep(1)
            yield b"{}"

        async def aclose(self) -> None:
            """Record ownership cleanup after the overall deadline fires."""
            nonlocal closed
            closed = True

    def slow(_request: httpx.Request) -> httpx.Response:
        """Return the synthetic native HTTP response stream."""
        return httpx.Response(200, stream=SlowBody())

    async with asyncio.timeout(0.5):
        assert not await module("launch").ready("a" * 16, 0.01, transport=httpx.MockTransport(slow))
    assert closed


def test_lost_docker_create_result_is_recovered_and_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI timeout retains daemon allocation identity and never starts an unknown ID."""
    smoke = module("smoke")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "space-package.json").write_text("{}")
    state: dict[str, Any] = {"created": False}
    calls: list[str] = []
    image = "sha256:" + "b" * 64

    def fake_docker(*arguments: str, environment: dict[str, str] | None = None) -> str:
        """Model a stopped daemon-created container whose successful CLI response was lost."""
        calls.append(arguments[0])
        if arguments[0] == "image":
            return json.dumps(image)
        if arguments[0] == "create":
            state.update(
                created=True,
                name=arguments[arguments.index("--name") + 1],
                label=arguments[arguments.index("--label") + 1].split("=", 1)[1],
            )
            raise subprocess.TimeoutExpired("docker create", 0.1)
        if arguments[0] == "ps":
            return "a" * 64 if state["created"] else ""
        if arguments[0] == "inspect":
            if arguments[2] == "{{json .State}}":
                return '{"Running":false,"ExitCode":0}'
            return json.dumps(
                {
                    "Name": "/" + state["name"],
                    "Config": {
                        "Image": image,
                        "Labels": {"finserve.space-smoke": state["label"]},
                    },
                }
            )
        if arguments[0] == "rm":
            state["created"] = False
        return ""

    monkeypatch.setattr(smoke, "docker", fake_docker)
    output = tmp_path / "evidence"
    with pytest.raises(subprocess.TimeoutExpired):
        smoke.run("local-image", bundle, output)
    receipt = json.loads((output / "smoke.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["attempts"][0]["cleanup"]["status"] == "removed"
    assert not state["created"] and "start" not in calls


def test_reconciliation_does_not_remove_mismatched_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused name is insufficient ownership evidence without the exact nonce and image."""
    smoke = module("smoke")
    calls: list[str] = []

    def fake_docker(*arguments: str, environment: dict[str, str] | None = None) -> str:
        """Return a name match with a foreign ownership label."""
        calls.append(arguments[0])
        if arguments[0] == "ps":
            return "a" * 64
        return json.dumps(
            {
                "Name": "/owned",
                "Config": {
                    "Image": "image",
                    "Labels": {
                        "finserve.space-smoke": "foreign",
                    },
                },
            }
        )

    monkeypatch.setattr(smoke, "docker", fake_docker)
    with pytest.raises(ValueError, match="ownership"):
        smoke.recover_owned("owned", "expected", "image")
    assert calls == ["ps", "inspect"]
