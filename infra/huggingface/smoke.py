"""Exercise a real CPU Space container with ephemeral credentials and retained safe receipts."""

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


def docker(*arguments: str, environment: dict[str, str] | None = None) -> str:
    """Pass structured arguments to Docker; secret values are inherited environment only."""
    result = subprocess.run(
        ["docker", *arguments],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return (result.stdout + (result.stderr if arguments[0] == "logs" else "")).strip()


def require(condition: bool, message: str) -> None:
    """Stop on failed evidence instead of interpreting an HTTP error as a successful query."""
    if not condition:
        raise ValueError(message)


def verify_http(url: str, key: str, internal_key: str, bundle: Path) -> dict[str, Any]:
    """Check the actual pages, public byte identities and bounded authenticated read boundary."""
    checks: dict[str, Any] = {}
    with httpx.Client(base_url=url, trust_env=False, follow_redirects=False, timeout=5) as client:
        for path in ("/", "/about", "/explorer"):
            response = client.get(path)
            require(response.status_code == 200, "page unavailable")
            require(
                key not in response.text and internal_key not in response.text, "credential leak"
            )
            checks[path] = hashlib.sha256(response.content).hexdigest()
            for asset in re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', response.text):
                resolved = str(httpx.URL(url + path).join(asset))
                data = client.get(resolved)
                require(data.status_code == 200, "bundled browser asset unavailable")
                require(
                    key not in data.text and internal_key not in data.text, "bundle credential leak"
                )
        query = {"query": "{ runs(first: 1) { id } }"}
        for label, credential in (
            ("missing", None),
            ("wrong", "wrong"),
            ("internal", internal_key),
        ):
            headers = {} if credential is None else {"Authorization": "Bearer " + credential}
            response = client.post("/graphql", json=query, headers=headers)
            require(response.status_code == 401, "edge authentication bypass")
            checks["auth_" + label] = response.status_code
        headers = {"Authorization": "Bearer " + key}
        response = client.post("/graphql", json=query, headers=headers)
        require(
            response.status_code == 200 and response.json() == {"data": {"runs": []}},
            "registry is not explicitly empty",
        )
        checks["authenticated_empty_registry"] = response.json()
        response = client.post("/graphql", json={"query": "mutation { missing }"}, headers=headers)
        require(response.status_code == 400, "mutation was not rejected")
        checks["mutation_status"] = response.status_code
        response = client.post("/graphql", content=b"x" * 16385, headers=headers)
        require(response.status_code == 413, "request byte cap was not enforced")
        checks["oversized_status"] = response.status_code
        response = client.post("/v1/completions", json={}, headers=headers)
        require(response.status_code == 404, "inference route exposed")
        checks["inference_status"] = response.status_code
        filename = "inference-demo.gif"
        response = client.get("/public-assets/" + filename)
        expected = (bundle / "docs/assets" / filename).read_bytes()
        require(
            response.status_code == 200 and response.content == expected,
            "public artifact changed",
        )
        checks[filename] = hashlib.sha256(expected).hexdigest()
    return checks


def wait_ready(identity: str) -> str:
    """Bound localhost startup polling and fail if the owned container exits first."""
    deadline = time.monotonic() + 70
    address = docker("port", identity, "7860/tcp")
    require(re.fullmatch(r"127\.0\.0\.1:[0-9]+", address) is not None, "unexpected port mapping")
    url = "http://" + address
    with httpx.Client(trust_env=False, timeout=1) as client:
        while time.monotonic() < deadline:
            require(
                docker("inspect", "--format", "{{.State.Running}}", identity) == "true",
                "container exited before readiness",
            )
            try:
                if client.get(url + "/healthz").status_code == 200:
                    return url
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
    raise TimeoutError("Space startup exceeded the declared smoke budget")


def persist(output: Path, receipt: dict[str, Any]) -> None:
    """Publish intent and cleanup receipts atomically before acquiring remote Docker state."""
    temporary = output / ".smoke.pending.json"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(output / "smoke.json")


def recover_owned(name: str, label: str, image: str) -> str | None:
    """Recover an ambiguous create only by exact name, nonce label and immutable image argument."""
    identity = docker(
        "ps", "--all", "--no-trunc", "--filter", "name=^/" + name + "$", "--format", "{{.ID}}"
    )
    if not identity:
        return None
    require(re.fullmatch(r"[0-9a-f]{64}", identity) is not None, "ambiguous container inventory")
    inspection = json.loads(docker("inspect", "--format", "{{json .}}", identity))
    require(
        inspection["Name"] == "/" + name
        and inspection["Config"]["Image"] == image
        and inspection["Config"]["Labels"].get("finserve.space-smoke") == label,
        "container ownership did not match persisted intent",
    )
    return identity


def finish_attempt(attempt: dict[str, Any], output: Path, key: str, internal_key: str) -> None:
    """Reconcile lost create responses and drain only a container matching the recorded fence."""
    attempt["cleanup"] = {"status": "unresolved"}
    identity = recover_owned(attempt["name"], attempt["label"], attempt["image"])
    if identity is None:
        # A timed-out create could still allocate an inactive container after this observation.
        # It was never started without a known ID, so absence is not claimed as a cleanup ACK.
        attempt["cleanup"] = {"status": "create_not_observed; outcome_unknown"}
        require(attempt["status"] != "complete", "completed attempt lost cleanup ownership")
        return
    try:
        docker("stop", "--time", "30", identity)
        logs = docker("logs", identity)
        require(key not in logs and internal_key not in logs, "secret in container output")
        (output / f"container-{int(attempt['child_failure'])}.log").write_text(logs)
        attempt["final_state"] = json.loads(
            docker("inspect", "--format", "{{json .State}}", identity)
        )
    finally:
        docker("rm", "--force", identity)
    require(
        recover_owned(attempt["name"], attempt["label"], attempt["image"]) is None,
        "container still present after removal",
    )
    attempt["cleanup"] = {"status": "removed", "container_id": identity}


def run(image: str, bundle: Path, output: Path) -> None:
    """Retain both normal and child-failure shutdown attempts without persisting credentials."""
    repository = Path(__file__).resolve().parents[2]
    output = output.resolve()
    require(output != repository and repository not in output.parents, "output must be external")
    output.mkdir(parents=True, exist_ok=False)
    key, internal_key = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    environment = dict(os.environ, FINSERVE_WEB_KEY=key, FINSERVE_API_KEY=internal_key)
    receipt: dict[str, Any] = {
        "status": "running",
        "scope": "local CPU container; no HF upload or GPU inference",
        "image": image,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attempts": [],
    }
    try:
        receipt["image_identity"] = json.loads(
            docker("image", "inspect", "--format", "{{json .Id}}", image)
        )
        receipt["bundle_manifest_sha256"] = hashlib.sha256(
            (bundle / "space-package.json").read_bytes()
        ).hexdigest()
        for failure in (False, True):
            nonce = uuid.uuid4().hex
            attempt: dict[str, Any] = {
                "child_failure": failure,
                "status": "creating",
                "label": nonce,
                "name": "finserve-hf-smoke-" + nonce,
                "image": receipt["image_identity"],
            }
            receipt["attempts"].append(attempt)
            persist(output, receipt)
            try:
                identity = docker(
                    "create",
                    "--name",
                    attempt["name"],
                    "--label",
                    "finserve.space-smoke=" + nonce,
                    "--publish",
                    "127.0.0.1::7860",
                    "--memory",
                    "512m",
                    "--cpus",
                    "2",
                    "--pids-limit",
                    "128",
                    "--env",
                    "FINSERVE_WEB_KEY",
                    "--env",
                    "FINSERVE_API_KEY",
                    "--env",
                    "WEB_CONCURRENCY=2",
                    "--env",
                    "UVICORN_RELOAD=true",
                    receipt["image_identity"],
                    environment=environment,
                )
                attempt["container_id"] = identity
                require(
                    identity == recover_owned(attempt["name"], nonce, receipt["image_identity"]),
                    "created container identity differs",
                )
                attempt["status"] = "starting"
                persist(output, receipt)
                docker("start", identity)
                url = wait_ready(identity)
                attempt["checks"] = verify_http(url, key, internal_key, bundle)
                attempt["user"] = docker("exec", identity, "id", "-u")
                require(attempt["user"] == "1000", "Space UID differs")
                code = (
                    "import pathlib,json; print(json.dumps([{'pid':int(p.name),"
                    "'argv':(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()} "
                    "for p in pathlib.Path('/proc').iterdir() "
                    "if p.name.isdigit() and (p/'cmdline').is_file()]))"
                )
                processes = json.loads(docker("exec", identity, "python", "-c", code))
                workers = [
                    row
                    for row in processes
                    if " -m uvicorn " in row["argv"] and "python -c " not in row["argv"]
                ]
                require(
                    len(workers) == 1 and "--workers 1" in workers[0]["argv"],
                    "unexpected Uvicorn worker topology",
                )
                attempt["uvicorn"] = workers[0]
                if failure:
                    docker(
                        "exec",
                        identity,
                        "python",
                        "-c",
                        f"import os,signal; os.kill({workers[0]['pid']},signal.SIGTERM)",
                    )
                    status = int(docker("wait", identity))
                    require(status == 1, "child failure did not fail the container")
                else:
                    docker("stop", "--time", "30", identity)
                    status = int(docker("inspect", "--format", "{{.State.ExitCode}}", identity))
                    require(status == 0, "normal shutdown failed")
                attempt.update(status="complete", exit_code=status)
            finally:
                try:
                    finish_attempt(attempt, output, key, internal_key)
                finally:
                    persist(output, receipt)
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(
            status="interrupted"
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "failed",
            error=type(error).__name__,
        )
        raise
    finally:
        persist(output, receipt)


def main() -> None:
    """Require explicit local image, immutable bundle and fresh external evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.image, args.bundle, args.output)


if __name__ == "__main__":
    main()
