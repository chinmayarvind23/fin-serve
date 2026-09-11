# AWS foundation

`foundation/` provisions a private EKS 1.35 cluster, initially two CPU nodes bounded at three, a GPU group bounded
to one node, private PostgreSQL and Redis, an encrypted versioned evidence bucket,
immutable ECR repository and scoped workload IAM. It uses Terraform 1.16.2 and the
locked AWS 6.61.0 provider. No Kubernetes provider is configured in this stage.

Local checks require no AWS credentials:

```sh
terraform -chdir=infra/terraform/foundation init -backend=false
terraform -chdir=infra/terraform/foundation fmt -check -recursive
terraform -chdir=infra/terraform/foundation validate
terraform -chdir=infra/terraform/foundation test
```

The tests use a mocked provider. They do not verify AWS service acceptance, capacity,
quotas, costs or deployment health. Store `TF_DATA_DIR`, plans and state outside the
source tree. The checked dependency lock records provider checksums.

Before an authenticated plan, resolve exact EKS AMI releases, add-on builds and
PostgreSQL minor availability in the selected region. `addon_versions` requires
`coredns`, `kube_proxy`, `vpc_cni` and `ebs_csi`; mock test builds are fixtures.
The EBS CSI controller has its own OIDC service-account role, with no volume
permissions added to the EC2 node role. Confirm the V2 managed policy ARN in the
[AWS policy reference](https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonEBSCSIDriverPolicyV2.html)
during preflight; AWS's setup examples currently show a different policy path.
Supply two AZs, intended AWS
account ID and an existing operator role. `variables.tf` documents every input.
Provide the Redis token through a secret input channel; it is sensitive in output
but exists in Terraform state. Use an existing restricted S3 state bucket with versioning,
encrypted storage and least-privilege access. The backend enables encryption and
S3 lockfiles; provide its bucket/key/region through an external backend configuration.

```sh
terraform -chdir=infra/terraform/foundation init -reconfigure -backend-config="$FINSERVE_BACKEND_CONFIG"
terraform -chdir=infra/terraform/foundation plan -var-file="$FINSERVE_FOUNDATION_INPUTS" -out="$FINSERVE_REVIEW_PLAN"
```

Review that saved plan and its full cost footprint before the separate apply stage.
The private API needs VPC/VPN/SSM network connectivity and the configured EKS access
role; an IAM grant alone supplies no network route. No public API endpoint or bastion
is created. RDS master credentials are managed by Secrets Manager; application DB
users, secret injection and schema migration remain deployment tasks.

Staging tradeoffs are explicit: one NAT gateway, single-AZ RDS, one Redis cache node,
two initial CPU nodes and GPU desired capacity zero by default. Desired sizes are
bootstrap values: Terraform ignores later desired-size changes so it cannot reset
the node controller's decisions. Without a controller, use an explicit EKS desired-size
update to bring the GPU group to one before installing the inference workload.
Changing `gpu_desired_nodes` after creation does not resize the group. This is not a high-availability
configuration or proof of any availability target. Include NAT, EKS, CPU/GPU, volumes,
data services and storage in cost evidence. RDS deletion protection and a final snapshot
require deliberate teardown handling; the evidence bucket and ECR repository do not
force-delete contents.

Install the [Kubernetes stages](../kubernetes/README.md) only after foundation readiness.

### Optional node scaling foundation

`enable_node_autoscaling=true` creates a dedicated IRSA role and applies discovery
tags to the actual EKS-managed Auto Scaling Groups. It does not install or start
Cluster Autoscaler. The role trusts only
`system:serviceaccount:kube-system:finserve-cluster-autoscaler` with the STS
audience. Its scaling writes require both exact managed ASG name patterns and
this cluster's two discovery tags; it cannot change node-group bounds or tag new
groups. Read-only capacity discovery is separate. The controller role ARN is
exported for the later Kubernetes installation, which must match EKS minor 1.35.

CPU bounds are two to three nodes in both modes; GPU bounds are zero to one.
Disabling the option removes role/tag configuration without reducing a group's
maximum below its current desired size. Stop the controller before revoking its
role. It does not automatically drain or shrink nodes on disable. When managing
desired sizes manually, inspect live Pods, disruption budgets and volume placement
before an explicit EKS update. Terraform continues to own the minimum and maximum.

The GPU group uses only the first configured AZ's private subnet. This prevents a
return from zero in an AZ that cannot mount the retained EBS model volume. Scale-zero
metadata records the GPU pool label, accelerator label, taint and one GPU resource.
This remains a single-AZ availability tradeoff, and EC2 capacity may be unavailable
when the group needs to return. Keeping the engine Pod scheduled consumes a GPU;
zero HTTP traffic alone does not remove that Pod or guarantee scale-down.

For an existing foundation, review the GPU node-group replacement caused by the
subnet change. Confirm existing model PVs are in `gpu_availability_zone` before
enabling scaling or migrating workloads; the change neither moves nor copies EBS
data. Review desired-size ownership and the new CPU maximum before applying.

Mocked plans verify scope, bounds and metadata. Live controller discovery, pending
Pod scheduling, drain and scale cycles remain unverified. The controller chart and
deployment are separate work. The design follows the
[AWS Cluster Autoscaler guidance](https://docs.aws.amazon.com/eks/latest/best-practices/cas.html)
and [zonal storage constraints](https://docs.aws.amazon.com/eks/latest/eksctl/autoscaling.html).
The [model storage stage](../kubernetes/storage/README.md) defines an encrypted
retained PVC; verified snapshot population and actual mount checks are separate.
