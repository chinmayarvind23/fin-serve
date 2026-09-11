# Model storage

Apply `model-storage.yaml` after the foundation's pinned `aws-ebs-csi-driver`
add-on is healthy and the `finserve` namespace exists. The manifest creates an
explicit, non-default encrypted gp3 StorageClass and a 20 GiB model claim. It
does not fetch weights or declare the claim ready.

```sh
kubectl apply -f infra/kubernetes/storage/model-storage.yaml
kubectl -n kube-system rollout status deployment/ebs-csi-controller --timeout=180s
kubectl -n kube-system get daemonset ebs-csi-node
kubectl -n finserve get pvc finserve-model
```

`WaitForFirstConsumer` intentionally leaves the claim Pending until a consumer
can be scheduled. The first population Pod must use the engine's GPU pool node
selector and taint toleration, even though copying files consumes no GPU resource.
The foundation now pins GPU nodes to the first configured AZ so replacement nodes
can return to the volume's zone. For an existing deployment, confirm the retained
PV matches the `gpu_availability_zone` output before replacing a multi-AZ GPU group.
That places the zonal volume where an engine can mount it. Do not bind the claim
through a CPU-only bootstrap Pod in another AZ. A later GPU node in another AZ
cannot mount that EBS volume; node availability and volume topology require an
explicit recovery plan.

Populate the claim with the exact verified model snapshot and readable ownership
for UID/GID 10001. Record the population attempt and rehash every file against the
producer manifest. Stop the writer before installing the inference workload with
`engine.modelClaim=finserve-model`. The engine independently rehashes the mounted
snapshot and mounts it read-only. `ReadWriteOnce` describes node attachment, not
an immutability or single-Pod guarantee. Automated verified population is a
separate deployment step; an empty allocated PVC cannot pass engine readiness.

The StorageClass uses the standard `ebs.csi.aws.com` driver and the account/region's
default EBS encryption key. Preflight `get-ebs-default-kms-key-id` and inspect the
selected key before deployment: an account default may be customer-managed.
Customer-managed KMS keys require scoped key policy and driver permissions; the
current controller role does not add those grants implicitly.
`Retain` preserves the underlying volume when a claim is removed, so deleting the
workload or claim does not stop storage charges or erase model files. Teardown
must identify the retained PV and EBS volume explicitly. No snapshot controller
or cross-AZ data replication is installed here.

The foundation uses the existing cluster OIDC provider with a dedicated IRSA
role bound only to `kube-system:ebs-csi-controller-sa`. This avoids adding a Pod
Identity agent solely for storage. Its AWS-managed V2 policy limits managed
volume operations by CSI tags; it is not a custom per-cluster resource boundary.
Check policy existence and the exact compatible add-on build during authenticated
preflight. Local provider mocks and YAML checks cannot establish IAM acceptance,
actual volume encryption, node mounts or driver readiness.

References: [AWS EBS CSI setup](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html),
[V2 policy ARN and permissions](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonEBSCSIDriverPolicyV2.html),
and [Kubernetes binding and reclaim behavior](https://kubernetes.io/docs/concepts/storage/storage-classes/).
