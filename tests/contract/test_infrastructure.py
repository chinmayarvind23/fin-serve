"""Render actual Helm charts and validate resource ownership without contacting a cluster."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest


def helm_binary() -> str:
    """An explicit optional Helm installation keeps default CPU test environments lightweight."""
    binary = os.getenv("FINSERVE_HELM") or shutil.which("helm")
    if not binary:
        pytest.skip("Helm CLI not installed; infrastructure verification tier")
    return binary


def fixture_values() -> dict[str, Any]:
    """Render-only digests and commits are synthetic and never name deployable image evidence."""
    return {
        "ray": {"image": "example.invalid/fixture-ray@sha256:" + "a" * 64},
        "engine": {
            "image": "example.invalid/fixture-engine@sha256:" + "b" * 64,
            "model": "fixture/model",
            "revision": "c" * 40,
            "tokenizerRevision": "d" * 40,
            "credentialsSecret": "fixture-credentials",
        },
    }


def render(tmp_path: Path, values: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    """Render with local fixture input; no kubeconfig or Kubernetes API connection is used."""
    chart = Path(__file__).resolve().parents[2] / "infra/kubernetes/workload"
    parameters = tmp_path / "values.json"
    parameters.write_text(json.dumps(values))
    return subprocess.run(
        [
            helm_binary(),
            "template",
            "fixture",
            str(chart),
            "--namespace",
            "finserve",
            "-f",
            str(parameters),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_gpu_ownership_and_internal_backend_contract(tmp_path: Path) -> None:
    """Only the engine owns a GPU; CPU actors receive the exact internal model/auth contract."""
    yaml = pytest.importorskip("yaml")
    result = render(tmp_path, fixture_values())
    assert result.returncode == 0, result.stderr
    documents = cast(list[dict[str, Any]], list(yaml.safe_load_all(result.stdout)))
    engine = next(item for item in documents if item["kind"] == "Deployment")
    pod = engine["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"finserve.io/pool": "gpu"}
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert engine["spec"]["strategy"]["type"] == "Recreate"
    ray = next(item for item in documents if item["kind"] == "RayService")
    cluster = ray["spec"]["rayClusterConfig"]
    assert cluster["headGroupSpec"]["serviceType"] == "ClusterIP"
    assert not cluster["enableInTreeAutoscaling"]
    head_request = cluster["headGroupSpec"]["template"]["spec"]["containers"][0]["resources"][
        "requests"
    ]["cpu"]
    assert head_request == "500m"
    assert cluster["workerGroupSpecs"][0]["replicas"] == 2
    # Declared staging budget: 1900m allocatable/node minus 300m system requests.
    # Actual node allocatable/daemon requests must still be checked before installation.
    worker_request = cluster["workerGroupSpecs"][0]["template"]["spec"]["containers"][0][
        "resources"
    ]["requests"]["cpu"]
    head_millicpu = int(head_request.removesuffix("m"))
    worker_millicpu = int(worker_request) * 1000
    assert head_millicpu + worker_millicpu <= 1900 - 300
    assert worker_millicpu <= 1900 - 300
    for group in [cluster["headGroupSpec"], *cluster["workerGroupSpecs"]]:
        assert group["rayStartParams"]["num-gpus"] == "0"
        container = group["template"]["spec"]["containers"][0]
        assert "nvidia.com/gpu" not in container["resources"]["limits"]
        assert container["env"][0]["name"] == "FINSERVE_ENGINE_API_KEY"
        assert "readinessProbe" in container and "startupProbe" in container
    application = yaml.safe_load(ray["spec"]["serveConfigV2"])["applications"][0]
    assert application["import_path"] == "finserve.engines.ray_backends:build_application"
    assert application["args"]["backends"] == {"engine-a": "http://fixture-engine:8000/v1"}
    assert application["args"]["model"] == "finserve-qwen"
    assert all(
        item["spec"]["type"] == "ClusterIP" for item in documents if item["kind"] == "Service"
    )


@pytest.mark.parametrize(
    "invalid", ["defaults", "mutable_image", "unbounded_workers", "mutable_model"]
)
def test_install_values_fail_closed(tmp_path: Path, invalid: str) -> None:
    """An omitted identity, mutable tag or capacity override must fail before installation."""
    values = fixture_values()
    if invalid == "defaults":
        values = {}
    elif invalid == "mutable_image":
        values["engine"]["image"] = "vllm/vllm-openai:latest"
    elif invalid == "unbounded_workers":
        values["ray"]["workers"] = 100
    else:
        values["engine"]["revision"] = "main"
    result = render(tmp_path, values)
    assert result.returncode != 0 and "schema" in result.stderr


def test_rayservice_matches_actual_pinned_operator_crd(tmp_path: Path) -> None:
    """Use the downloaded operator's real OpenAPI schema, rather than a hand-maintained mock."""
    yaml = pytest.importorskip("yaml")
    jsonschema = pytest.importorskip("jsonschema")
    directory = Path(__file__).resolve().parents[2] / "infra/kubernetes/operators/charts"
    operator = directory / "kuberay-operator-1.6.1.tgz"
    if not operator.exists():
        pytest.skip("Run helm dependency build before CRD verification")
    result = subprocess.run(
        [helm_binary(), "show", "crds", str(operator)],
        check=True,
        capture_output=True,
        text=True,
    )
    definitions = list(yaml.safe_load_all(result.stdout))
    definition = next(
        item for item in definitions if item and item["metadata"]["name"] == "rayservices.ray.io"
    )
    schema = next(item for item in definition["spec"]["versions"] if item["name"] == "v1")[
        "schema"
    ]["openAPIV3Schema"]
    rendered = render(tmp_path, fixture_values())
    assert rendered.returncode == 0, rendered.stderr
    resource = next(
        item for item in yaml.safe_load_all(rendered.stdout) if item["kind"] == "RayService"
    )
    jsonschema.Draft4Validator(schema).validate(resource)
