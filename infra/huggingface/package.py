"""Create an exclusive external Space bundle from an explicit public-source allowlist."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SPACE_FILES = {
    "Dockerfile",
    "README.md",
    "launch.py",
    "package.py",
    "space_server.ts",
    "landing.html",
    "landing.css",
}
ROOT_FILES = {"pyproject.toml", "uv.lock", "package.json", "bun.lock"}
PUBLIC_FILES = {"docs/assets/inference-demo.gif"}


def git(repository: Path, *arguments: str) -> bytes:
    """Read version-control identity without invoking a shell or copying its private metadata."""
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True, capture_output=True
    ).stdout


def selected_files(repository: Path) -> set[str]:
    """Include tracked application/runtime source and seven named Space files, never raw data."""
    tracked = set(git(repository, "ls-files", "--cached", "-z").decode().strip("\0").split("\0"))
    if not (ROOT_FILES | PUBLIC_FILES) <= tracked:
        raise ValueError("required public asset or lock file is not committed")
    selected = (
        ROOT_FILES
        | PUBLIC_FILES
        | {name for name in tracked if name.startswith(("src/", "apps/api/", "apps/web/"))}
        | {"infra/huggingface/" + name for name in SPACE_FILES}
    )
    for name in PUBLIC_FILES:
        committed = git(repository, "show", "HEAD:" + name)
        working = (repository / name).read_bytes()
        if committed != working and not (
            name.endswith((".json", ".svg")) and working.replace(b"\r\n", b"\n") == committed
        ):
            raise ValueError("public asset differs from committed bytes")
    return selected


def package(repository: Path, output: Path, *, static: bool = False) -> None:
    """Copy only reviewed paths into fresh external storage, retaining exact byte provenance."""
    repository, output = repository.resolve(), output.resolve()
    if output == repository or repository in output.parents:
        raise ValueError("Space bundle must be outside the source repository")
    selected = selected_files(repository)
    contents: dict[str, bytes] = {}
    for name in sorted(selected):
        source = repository / name
        if source.is_symlink() or repository not in source.resolve().parents:
            raise ValueError("source must be a regular file inside the repository")
        contents[name] = (
            git(repository, "show", "HEAD:" + name) if name in PUBLIC_FILES else source.read_bytes()
        )
    contents["Dockerfile"] = contents["infra/huggingface/Dockerfile"]
    contents["README.md"] = contents["infra/huggingface/README.md"]
    if static:
        contents = static_contents(repository, contents)
    status = git(repository, "status", "--porcelain=v1")
    manifest = {
        "scope": "static product guide; no API or inference"
        if static
        else "CPU evidence explorer; public asset files only; empty initial registry",
        "git_revision": git(repository, "rev-parse", "HEAD").decode().strip(),
        "dirty": bool(status),
        "git_status_sha256": hashlib.sha256(status).hexdigest(),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in contents.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (output / "space-package.json").write_text(json.dumps(manifest, indent=2) + "\n")


def static_contents(repository: Path, source: dict[str, bytes]) -> dict[str, bytes]:
    """Publish only a static page and committed demo assets; never ship API code or secrets."""
    page = repository / "infra/huggingface/static.html"
    if page.is_symlink() or repository not in page.resolve().parents:
        raise ValueError("static page must remain inside the repository")
    result = {
        "index.html": page.read_bytes(),
        "landing.css": source["infra/huggingface/landing.css"],
        "README.md": (
            "---\ntitle: FinServe\nemoji: 📊\ncolorFrom: blue\n"
            "colorTo: green\nsdk: static\napp_file: index.html\n---\n\n"
            "Product introduction and local setup guide. No inference or API runs "
            "in this Space. Open [local setup instructions](./local.html) in the app "
            "or read [the full guide](./run-free.md). GitHub source requires repository access.\n"
        ).encode(),
    }
    for destination, name in {
        "local.html": "infra/huggingface/local.html",
        "run-free.md": "docs/run-free.md",
    }.items():
        path = repository / name
        if path.is_symlink() or repository not in path.resolve().parents:
            raise ValueError("static instructions must remain inside the repository")
        result[destination] = path.read_bytes()
    result.update({"public-assets/" + Path(name).name: source[name] for name in PUBLIC_FILES})
    return result


def main() -> None:
    """Accept one fresh external destination; callers cannot broaden the source allowlist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--static", action="store_true", help="package the free static product page"
    )
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[2], args.output, static=args.static)


if __name__ == "__main__":
    main()
