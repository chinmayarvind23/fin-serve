# Kubernetes stages

Install `operators/` before `workload/`. The pinned operator chart includes KubeRay
1.6.1 and NVIDIA device plugin 0.20.0. The plugin uses `nvidia.com/gpu` integer resources;
do not install NVIDIA DRA on the same nodes. Use the AL2023 NVIDIA AMI selected by the
foundation and confirm device allocation before serving.

```sh
helm dependency build infra/kubernetes/operators
helm lint infra/kubernetes/operators --namespace finserve-operators
helm template finserve-operators infra/kubernetes/operators --namespace finserve-operators --include-crds
```

For an authorized cluster deployment, install operators in **finserve-operators**,
wait for the RayService CRD and operator readiness, then install the workload in
**finserve**. Namespace identities are intentional: network policy allows operator
dashboard access, and IRSA is scoped to `finserve:finserve-evidence`.
An authenticated private EKS context and network access are prerequisites.

The workload has one separately managed vLLM engine Deployment owning one physical
GPU. Its Ray head and two CPU worker Pods own zero GPUs. Ray's internal NDJSON endpoint
is not the public FastAPI SSE/Bun API. The application builder is
`finserve.engines.ray_backends:build_application`; it receives one distinct internal
engine URL, the served model ID and bounded worker capacity. Worker credentials use
`FINSERVE_ENGINE_API_KEY`, drawn from the same Secret key as `VLLM_API_KEY`.

Supply an external values file with real image digests, exact model and tokenizer
commit IDs, and an existing Kubernetes Secret name with an `api-key` key. Empty values,
mutable image tags and branch revisions fail schema validation. The Ray image must
contain FinServe, Ray 2.58.0, compatible locked dependencies, Python, Bash and wget;
the engine image must contain the tested vLLM API-server entry point. Both must run as
UID/GID 1000. The engine cache is bounded ephemeral storage and needs model-download
egress; private/gated model credentials are a separate deployment prerequisite.

```sh
helm lint infra/kubernetes/workload --namespace finserve -f "$FINSERVE_WORKLOAD_VALUES"
helm template finserve infra/kubernetes/workload --namespace finserve -f "$FINSERVE_WORKLOAD_VALUES"
```

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
admission, GPU allocation, AWS apply or cloud inference was executed. Public gateway,
visual worker, registry/Airflow workloads, TLS ingress, application secret provisioning
and cross-system promotion/rollback adapters are separate deployment work.
