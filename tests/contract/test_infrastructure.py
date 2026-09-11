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
            "modelClaim": "fixture-model",
            "profileConfigMap": "fixture-profile",
            "profileSha256": "c" * 64,
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
    assert pod["securityContext"]["runAsUser"] == 10001
    runtime = pod["containers"][0]
    assert runtime["command"] == ["python3", "-m", "finserve.registry.engine_entrypoint"]
    assert runtime["securityContext"]["readOnlyRootFilesystem"]
    assert runtime["args"][runtime["args"].index("--expected-credential-env") + 1] == (
        "FINSERVE_ENGINE_API_KEY"
    )
    assert runtime["args"][runtime["args"].index("--profile-sha256") + 1] == "c" * 64
    assert runtime["args"][runtime["args"].index("--expected-base-url") + 1] == (
        "http://fixture-engine:8000/v1"
    )
    mounts = {item["name"]: item for item in runtime["volumeMounts"]}
    assert mounts["model"]["readOnly"] and mounts["model"]["mountPath"] == "/models"
    assert mounts["profile"]["readOnly"]
    volumes = {item["name"]: item for item in pod["volumes"]}
    assert volumes["model"]["persistentVolumeClaim"] == {
        "claimName": "fixture-model",
        "readOnly": True,
    }
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
    "invalid",
    [
        "defaults",
        "mutable_image",
        "unbounded_workers",
        "missing_profile",
        "legacy_model",
        "invalid_secret",
    ],
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
    elif invalid == "legacy_model":
        values["engine"]["revision"] = "main"
    elif invalid == "invalid_secret":
        values["engine"]["credentialsSecret"] = "invalid-"
    else:
        values["engine"]["profileSha256"] = ""
    result = render(tmp_path, values)
    assert result.returncode != 0 and "schema" in result.stderr


@pytest.mark.parametrize("with_gateway", [False, True])
@pytest.mark.parametrize("autoscaling", [False, True])
def test_rayservice_matches_actual_pinned_operator_crd(
    tmp_path: Path, with_gateway: bool, autoscaling: bool
) -> None:
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
    values = gateway_values() if with_gateway else fixture_values()
    values["ray"]["autoscaling"] = {"enabled": autoscaling}
    rendered = render(tmp_path, values)
    assert rendered.returncode == 0, rendered.stderr
    resource = next(
        item for item in yaml.safe_load_all(rendered.stdout) if item["kind"] == "RayService"
    )
    jsonschema.Draft4Validator(schema).validate(resource)


def test_cpu_autoscaling_keeps_engine_capacity_and_worker_permissions_fixed(tmp_path: Path) -> None:
    """The sidecar fits the CPU budget and gets its own head identity, never engine credentials."""
    yaml = pytest.importorskip("yaml")
    values = gateway_values()
    values["ray"]["autoscaling"] = {"enabled": True, "idleTimeoutSeconds": 120}
    result = render(tmp_path, values)
    assert result.returncode == 0, result.stderr
    documents = list(yaml.safe_load_all(result.stdout))
    ray = next(item for item in documents if item["kind"] == "RayService")
    cluster = ray["spec"]["rayClusterConfig"]
    assert cluster["enableInTreeAutoscaling"] is True
    options = cluster["autoscalerOptions"]
    assert options["version"] == "v2" and options["upscalingMode"] == "Conservative"
    assert options["idleTimeoutSeconds"] == 120
    assert options["resources"]["requests"] == {"cpu": "100m", "memory": "512Mi"}
    assert "env" not in options and "image" not in options
    head = cluster["headGroupSpec"]["template"]["spec"]
    assert "serviceAccountName" not in head  # Operator creates a cluster-specific head account.
    worker = cluster["workerGroupSpecs"][0]
    assert (worker["replicas"], worker["minReplicas"], worker["maxReplicas"]) == (1, 1, 2)
    assert worker["template"]["spec"]["serviceAccountName"] == "fixture-runtime"
    account = next(
        item
        for item in documents
        if item["kind"] == "ServiceAccount" and item["metadata"]["name"] == "fixture-runtime"
    )
    assert account["automountServiceAccountToken"] is False
    assert 500 + 100 + 1000 <= 1900 - 300
    engine = next(
        item
        for item in documents
        if item["kind"] == "Deployment" and item["metadata"]["name"] == "fixture-engine"
    )
    assert engine["spec"]["replicas"] == 1
    app = yaml.safe_load(ray["spec"]["serveConfigV2"])["applications"][0]
    assert app["args"]["backends"] == {"engine-a": "http://fixture-engine:8000/v1"}
    assert app["args"]["capacity_per_worker"] == 16


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": "true"},
        {"idleTimeoutSeconds": 0},
        {"idleTimeoutSeconds": 601},
        {"maxWorkers": 100},
        {"image": "unreviewed:latest"},
    ],
)
def test_autoscaling_rejects_unsafe_overrides(tmp_path: Path, override: dict[str, Any]) -> None:
    """Scaling cannot bypass the reviewed resource, image or worker bounds through extra values."""
    values = fixture_values()
    values["ray"]["autoscaling"] = override
    result = render(tmp_path, values)
    assert result.returncode != 0 and "schema" in result.stderr


def gateway_values() -> dict[str, Any]:
    """An enabled render fixture binds separate ingress and Ray hop credentials by Secret keys."""
    values = fixture_values()
    values["gateway"] = {
        "enabled": True,
        "image": "example.invalid/fixture-gateway@sha256:" + "e" * 64,
        "sourceRevision": "f" * 40,
        "credentialsSecret": "fixture-gateway",
    }
    return values


def test_gateway_and_ray_share_only_the_internal_hop_credential(tmp_path: Path) -> None:
    """The public SSE process connects to the operator's internal NDJSON Service with fixed auth."""
    yaml = pytest.importorskip("yaml")
    rendered = render(tmp_path, gateway_values())
    assert rendered.returncode == 0, rendered.stderr
    documents = list(yaml.safe_load_all(rendered.stdout))
    gateway = next(
        item
        for item in documents
        if item["kind"] == "Deployment" and item["metadata"]["name"] == "fixture-gateway"
    )
    pod = gateway["spec"]["template"]["spec"]
    runtime = pod["containers"][0]
    env = {item["name"]: item for item in runtime["env"]}
    assert env["FINSERVE_ENGINE"]["value"] == "ray-http"
    assert env["FINSERVE_REQUIRE_AUTH"]["value"] == "1"
    assert env["FINSERVE_ENGINE_URL"]["value"] == "http://fixture-ray-serve-svc:8000"
    assert env["FINSERVE_API_KEY"]["valueFrom"]["secretKeyRef"]["key"] == "api-key"
    hop = env["FINSERVE_RAY_API_KEY"]["valueFrom"]
    assert hop == {"secretKeyRef": {"name": "fixture-gateway", "key": "ray-api-key"}}
    assert "FINSERVE_ENGINE_API_KEY" not in env
    assert pod["securityContext"]["runAsUser"] == 10001
    assert runtime["securityContext"]["readOnlyRootFilesystem"]
    assert "nvidia.com/gpu" not in runtime["resources"]["limits"]
    ray = next(item for item in documents if item["kind"] == "RayService")
    cluster = ray["spec"]["rayClusterConfig"]
    for group in [cluster["headGroupSpec"], *cluster["workerGroupSpecs"]]:
        actor_env = {
            item["name"]: item for item in group["template"]["spec"]["containers"][0]["env"]
        }
        assert actor_env["FINSERVE_RAY_API_KEY"]["valueFrom"] == hop
        assert actor_env["FINSERVE_REQUIRE_AUTH"]["value"] == "1"
        assert "FINSERVE_API_KEY" not in actor_env


@pytest.mark.parametrize("field", ["image", "sourceRevision", "credentialsSecret"])
def test_enabled_gateway_requires_deployment_identity(tmp_path: Path, field: str) -> None:
    """An enabled API needs immutable image/source identities and a credential reference."""
    values = gateway_values()
    values["gateway"][field] = ""
    result = render(tmp_path, values)
    assert result.returncode != 0 and "schema" in result.stderr


def test_model_storage_preserves_weights_and_waits_for_consumer_topology() -> None:
    """A claim cannot silently bind in the wrong AZ, lose encryption or delete retained weights."""
    yaml = pytest.importorskip("yaml")
    manifest = Path(__file__).resolve().parents[2] / "infra/kubernetes/storage/model-storage.yaml"
    storage, claim = list(yaml.safe_load_all(manifest.read_text()))
    assert storage["provisioner"] == "ebs.csi.aws.com"
    assert storage["parameters"]["encrypted"] == "true"
    assert storage["parameters"]["type"] == "gp3"
    assert storage["volumeBindingMode"] == "WaitForFirstConsumer"
    assert storage["reclaimPolicy"] == "Retain"
    assert (
        storage["metadata"]["annotations"]["storageclass.kubernetes.io/is-default-class"] == "false"
    )
    assert claim["spec"]["storageClassName"] == storage["metadata"]["name"]
    assert claim["metadata"]["namespace"] == "finserve"
    assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert claim["spec"]["resources"]["requests"]["storage"] == "20Gi"


def render_node_autoscaler(
    tmp_path: Path,
    values: dict[str, Any],
    namespace: str = "kube-system",
    release: str = "finserve-cluster-autoscaler",
    version: str = "1.35.0",
) -> subprocess.CompletedProcess[str]:
    """Render the node controller without credentials or Kubernetes API discovery."""
    chart = Path(__file__).resolve().parents[2] / "infra/kubernetes/node-autoscaler"
    parameters = tmp_path / "node-values.json"
    parameters.write_text(json.dumps(values))
    return subprocess.run(
        [
            helm_binary(),
            "template",
            release,
            str(chart),
            "--namespace",
            namespace,
            "--kube-version",
            version,
            "-f",
            str(parameters),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def node_autoscaler_values() -> dict[str, Any]:
    """Synthetic account identity is only a render fixture and has no credentials."""
    return {
        "clusterName": "finserve-staging",
        "awsAccountId": "123456789012",
        "awsRegion": "us-east-1",
    }


def test_node_controller_identity_capacity_and_permissions(tmp_path: Path) -> None:
    """Controller identity, lock names and fixed flags must agree with its scoped foundation."""
    yaml = pytest.importorskip("yaml")
    result = render_node_autoscaler(tmp_path, node_autoscaler_values())
    assert result.returncode == 0, result.stderr
    docs = {item["kind"]: item for item in yaml.safe_load_all(result.stdout)}
    account = docs["ServiceAccount"]
    assert account["automountServiceAccountToken"] is True
    assert account["metadata"]["annotations"]["eks.amazonaws.com/role-arn"] == (
        "arn:aws:iam::123456789012:role/finserve-staging-node-autoscaler"
    )
    deployment = docs["Deployment"]["spec"]
    assert deployment["replicas"] == 1 and deployment["strategy"]["type"] == "Recreate"
    pod = deployment["template"]["spec"]
    assert pod["serviceAccountName"] == "finserve-cluster-autoscaler"
    assert pod["nodeSelector"] == {"finserve.io/pool": "cpu"}
    assert "volumes" not in pod
    runtime = pod["containers"][0]
    assert runtime["command"] == ["/cluster-autoscaler"]
    assert runtime["image"] == (
        "registry.k8s.io/autoscaling/cluster-autoscaler@sha256:"
        "aac369dc283927a623deb1af54696efcc722ae79255aa07788422e495bab887d"
    )
    flags = dict(argument[2:].split("=", 1) for argument in runtime["args"])
    assert flags["max-nodes-total"] == "4"
    assert flags["node-group-auto-discovery"] == (
        "asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/finserve-staging"
    )
    assert flags["max-drain-parallelism"] == flags["max-scale-down-parallelism"] == "1"
    assert (
        flags["skip-nodes-with-local-storage"]
        == flags["skip-nodes-with-custom-controller-pods"]
        == "true"
    )
    assert runtime["resources"]["requests"] == {"cpu": "100m", "memory": "600Mi"}
    env = {entry["name"]: entry["value"] for entry in runtime["env"]}
    assert env == {
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }
    assert runtime["securityContext"]["readOnlyRootFilesystem"]
    rules = docs["Role"]["rules"]
    restricted = {rule["resources"][0]: rule for rule in rules if "resourceNames" in rule}
    assert restricted["leases"]["resourceNames"] == [flags["leader-elect-resource-name"]]
    assert restricted["configmaps"]["resourceNames"] == [flags["status-config-map-name"]]
    assert all("resourceNames" not in rule for rule in rules if "create" in rule["verbs"])
    cluster_rules = docs["ClusterRole"]["rules"]
    assert all("*" not in rule["verbs"] + rule["resources"] for rule in cluster_rules)
    assert not any("secrets" in rule["resources"] for rule in cluster_rules)
    dra = next(rule for rule in cluster_rules if rule["apiGroups"] == ["resource.k8s.io"])
    assert set(dra["resources"]) == {"resourceslices", "deviceclasses", "resourceclaims"}
    assert set(dra["verbs"]) == {"get", "list", "watch"}
    for kind in ("RoleBinding", "ClusterRoleBinding"):
        assert docs[kind]["subjects"] == [
            {
                "kind": "ServiceAccount",
                "name": account["metadata"]["name"],
                "namespace": "kube-system",
            }
        ]


@pytest.mark.parametrize(
    "override",
    [
        {},
        {"awsAccountId": "wrong"},
        {"clusterName": "bad,name"},
        {"awsRegion": "us-east-1 --evil"},
        {"extraArgs": {"max-nodes-total": "100"}},
        {"image": "unreviewed:latest"},
        {"replicas": 10},
    ],
)
def test_node_controller_values_fail_closed(tmp_path: Path, override: dict[str, Any]) -> None:
    """Unknown overrides cannot expand controller images, credentials, flags or capacity."""
    values = node_autoscaler_values() | override if override else {}
    result = render_node_autoscaler(tmp_path, values)
    assert result.returncode != 0 and "schema" in result.stderr


@pytest.mark.parametrize(
    "namespace,release,version",
    [
        ("default", "finserve-cluster-autoscaler", "1.35.0"),
        ("kube-system", "second-controller", "1.35.0"),
        ("kube-system", "finserve-cluster-autoscaler", "1.34.0"),
        ("kube-system", "finserve-cluster-autoscaler", "1.36.0"),
    ],
)
def test_node_controller_rejects_wrong_cluster_contract(
    tmp_path: Path, namespace: str, release: str, version: str
) -> None:
    """IRSA/leader identity and the Kubernetes minor cannot drift at installation."""
    result = render_node_autoscaler(tmp_path, node_autoscaler_values(), namespace, release, version)
    assert result.returncode != 0
