# EKS node controller

This chart installs Cluster Autoscaler 1.35.2 for the FinServe EKS 1.35 foundation.
It can change desired node counts within the foundation's CPU range of two to
three and GPU range of zero to one. It does not add model replicas, make a failed
quality gate pass, or remove the engine Pod when HTTP traffic stops.

Apply the foundation with `enable_node_autoscaling=true` first. Verify its actual
ASG tags and `node_autoscaler_role_arn`, the GPU AZ/model PV match, and available
CPU capacity. Use the authorized private Kubernetes API connection. Do not install
this alongside another node autoscaler managing the same groups.

Store these three values outside the repository, using the actual foundation
inputs. Unknown fields, image overrides and arbitrary arguments are rejected:

```yaml
clusterName: finserve-staging
awsAccountId: "YOUR_12_DIGIT_ACCOUNT_ID"
awsRegion: us-east-1
```

The example account placeholder intentionally fails validation. The chart derives
`arn:aws:iam::<account>:role/<cluster>-node-autoscaler` and uses the exact IRSA
subject `kube-system/finserve-cluster-autoscaler`. The existing Terraform account
fence still applies; Helm rendering cannot verify the caller's AWS identity.

```sh
helm template finserve-cluster-autoscaler infra/kubernetes/node-autoscaler \
  --namespace kube-system --kube-version 1.35.0 -f "$FINSERVE_NODE_SCALING_VALUES"
helm upgrade --install finserve-cluster-autoscaler infra/kubernetes/node-autoscaler \
  --namespace kube-system -f "$FINSERVE_NODE_SCALING_VALUES" --atomic --timeout 5m
```

The release name, namespace and Kubernetes minor are checked before rendering.
The image is pinned to the official 1.35.2 manifest-list digest
`sha256:aac369dc283927a623deb1af54696efcc722ae79255aa07788422e495bab887d`.
The Linux amd64 child is
`sha256:8c8cc9d71988a62ec9365c0bfac9e0a5aeb57e4d95875e641a7bee325dca596e`.
Registry response bytes and a real binary argument check are retained privately.

The controller requests 100m CPU and 600Mi memory on the CPU pool, with limits of
500m and 600Mi. Include this in scheduling headroom alongside the Ray head,
autoscaler sidecar, workers, gateway and system Pods. It runs as a non-root user
with a read-only filesystem and no host mounts. Kubernetes mounts its API token;
the EKS webhook must inject the separate IRSA web-identity token. EC2 metadata
credential fallback is disabled. No AWS access keys belong in Helm values.

Cluster-wide reads cover scheduling objects and DRA metadata required by CA 1.35;
FinServe still uses integer device-plugin GPUs. Node updates and Pod eviction are
needed for drain. The namespaced Role can create leases/configmaps in kube-system,
because Kubernetes cannot restrict create by object name; subsequent writes are
limited to this controller's lock and status names. There is no Secret read grant.
Optional provisioning-request and capacity-buffer controllers are disabled.

Total nodes are capped at four, with one drain at a time and ten-minute scale-down
delays. Local-storage and custom-controller Pods prevent scale-down under the
selected flags. The system-Pod check has an upstream one-hour blocking timeout,
so it is not an absolute no-eviction guarantee. Active Ray actors can prevent
worker removal, and warm engine ownership still needs explicit application drain
proof. Do not infer safe release-runtime reclamation from node-controller health.

After installation, retain controller logs, the scoped status ConfigMap, actual
node-group desired counts and Pod/node events. Exercise CPU pending demand, GPU
return from zero in the model PV's AZ, failed provisioning, and a drained scale-down
with all request outcomes retained. A healthy `/health-check` is not evidence that
IAM, AWS capacity, model mounts or these scale cycles work. Before uninstalling,
quiesce the controller and inspect ongoing node operations; Helm removal does not
undo desired-size changes, restore evicted Pods, or drain application leases.

Current verification covers twelve Helm contract tests and an actual binary
parsing the rendered flags with `--help`, network disabled and no credentials.
No controller reconciliation, AWS deployment or live drain ran.

References: [AWS Cluster Autoscaler guidance](https://docs.aws.amazon.com/eks/latest/best-practices/cas.html),
[upstream 1.35.2 flags](https://github.com/kubernetes/autoscaler/blob/cluster-autoscaler-1.35.2/cluster-autoscaler/config/flags/flags.go),
[upstream Helm RBAC](https://github.com/kubernetes/autoscaler/tree/cluster-autoscaler-chart-9.59.0/cluster-autoscaler/charts/cluster-autoscaler/templates).
