# Kubernetes stages

Install `operators/` before `workload/`. The pinned operator chart includes KubeRay
1.6.1 and NVIDIA device plugin 0.20.0. The plugin uses `nvidia.com/gpu` integer resources;
do not install NVIDIA DRA on the same nodes. Use the AL2023 NVIDIA AMI selected by the
foundation and confirm device allocation before serving.

```sh
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo add nvidia https://nvidia.github.io/k8s-device-plugin
helm dependency build infra/kubernetes/operators
helm lint infra/kubernetes/operators --namespace finserve-operators
helm template finserve-operators infra/kubernetes/operators --namespace finserve-operators --include-crds
```

For an authorized cluster deployment, install operators in **finserve-operators**,
wait for the RayService CRD and operator readiness, then install the workload in
**finserve**. Namespace identities are intentional: network policy allows operator
dashboard access, and IRSA is scoped to `finserve:finserve-evidence`.
An authenticated private EKS context and network access are prerequisites.
The [model storage stage](storage/README.md) supplies an explicit encrypted gp3
claim after the foundation's pinned EBS CSI add-on is healthy. Snapshot population
must use the GPU pool's topology and retain a verified file receipt.

The workload has one separately managed vLLM engine Deployment owning one physical
GPU. Its Ray head and two CPU worker Pods own zero GPUs. Ray's internal NDJSON endpoint
is not the public FastAPI SSE/Bun API. The application builder is
`finserve.engines.ray_backends:build_application`; it receives one distinct internal
engine URL, the served model ID and bounded worker capacity. Worker credentials use
`FINSERVE_ENGINE_API_KEY`, drawn from the same Secret key as `VLLM_API_KEY`.

Supply an external values file with real image digests, a populated model PVC,
an immutable profile ConfigMap, its canonical SHA256 and an existing Secret with an
`api-key` key. Empty identities, mutable image tags and old model-download values
fail schema validation. The Ray image must
contain FinServe, Ray 2.58.0, compatible locked dependencies, Python, Bash and wget;
it runs as UID/GID 1000. The engine image is built by `registry/runtime_build.py` and
runs as UID/GID 10001. It must include the current verified entrypoint's deployment
binding flags; older images fail startup instead of bypassing verification.

The PVC root must contain the exact snapshot described by the manifest baked into
the engine image, readable by UID/GID 10001. The chart mounts it read-only at `/models`.
The existing immutable ConfigMap contains `profile.json`, a canonical
`ServingProfileV1` with the internal URL `http://<release>-engine:8000/v1`, the same
`served_model` as `engine.servedModel`, and
`credential_env=FINSERVE_ENGINE_API_KEY`. `engine.profileSha256` is `profile.digest()`.
Model/tokenizer commits, manifests and inference parameters live in that profile.
The entrypoint checks its hash, endpoint, model and credential name, then rehashes
the actual snapshot and compares the installed engine version before exec. Startup
performs no Hub download. A ready PVC and profile remain deployment prerequisites;
this chart does not fabricate their receipts or upload local model files.

The root filesystem is read-only. Bounded `/tmp` scratch must permit native library
execution for Triton, and `/dev/shm` has a separate memory-backed bound. PVC access
modes alone do not enforce read-only use, so both the claim source and container
mount specify it. Storage-driver support, volume ownership and actual node mount
flags require cluster validation. See the Kubernetes documentation for
[volume mounts](https://kubernetes.io/docs/concepts/storage/volumes/) and
[security contexts](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/).

```sh
helm lint infra/kubernetes/workload --namespace finserve -f "$FINSERVE_WORKLOAD_VALUES"
helm template finserve infra/kubernetes/workload --namespace finserve -f "$FINSERVE_WORKLOAD_VALUES"
```

Enable `gateway.enabled` to deploy the actual FastAPI text/chat SSE boundary.
Supply `gateway.image` by digest, its exact `gateway.sourceRevision`, and
`gateway.credentialsSecret` with independent `api-key` and `ray-api-key` values.
The gateway receives the ingress key and Ray hop key. Ray receives the hop key and
the separate engine key; the ingress key never enters Ray. The gateway connects
to KubeRay's generated `<release>-ray-serve-svc:8000` Service. That internal
[Serve Service](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayservice-quick-start.html)
speaks FinServe NDJSON, while the gateway keeps the OpenAI-style public contract.

The chart sets `FINSERVE_REQUIRE_AUTH=1` in the gateway and both Ray pod groups.
Their factories and actor constructors reject missing, empty, short or malformed
credentials before allocating clients or serving actors. Keys must contain 16 to
4,096 printable non-whitespace ASCII characters; generate independent random
values of at least 32 characters. The ordinary local fixture mode remains explicit
and does not enable this production requirement by default.

The gateway runs on CPU as UID/GID 10001 with a read-only root, bounded scratch and
explicit admission capacity. `/healthz` probes establish process readiness only.
After installation, verify an authenticated streamed request, authoritative token
usage, the expected image/source identity and the quality gate before promotion.
For private operator access, forward the gateway Service over the authenticated
Kubernetes connection:

```sh
kubectl -n finserve port-forward service/finserve-gateway 8038:8000 --address 127.0.0.1
```

The chart exposes only ClusterIP Services. TLS ingress and the Bun edge remain
separate deployment work. Redis quota, vision backends and durable visual jobs are
not enabled by this text-gateway chart. The gateway is optional so operators can
validate the internal engine/Ray stage before admitting API traffic.

CPU requests are 500m for the head and 1000m per worker, plus operator/daemon capacity;
logical head CPU is zero. This fits the two-node staging resource budget in principle,
but actual allocatable resources and system Pods must be checked before installation.
One CPU proxy actor addresses the one engine; two worker Pods do not imply two model
replicas or additional GPU throughput.

The engine uses a Recreate rollout because one GPU cannot accommodate a surge Pod.
Updates interrupt service. Separate candidate and known-good capacity is required
before claiming canary promotion or fast rollback. Ray process probes and engine
`/health` establish basic readiness only: quality, exact model identity and measured
performance gates remain necessary. Network policies rely on the foundation's enabled
VPC CNI network-policy support; dashboard/GCS and engine Services remain internal.

Current verification covers Helm rendering, actual pinned RayService CRD schema,
identity/capacity rejection and Terraform mocked plans. No Kubernetes server-side
admission, GPU allocation, AWS apply or cloud inference was executed. Visual worker,
registry/Airflow workloads, TLS ingress, application secret provisioning
and cross-system promotion/rollback adapters are separate deployment work.
