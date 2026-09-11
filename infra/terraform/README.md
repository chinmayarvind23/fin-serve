# AWS foundation

`foundation/` provisions a private EKS 1.35 cluster, two CPU nodes, a GPU group bounded
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
fixed CPU capacity and GPU desired capacity zero by default. Set GPU desired capacity
to one before installing the inference workload. This is not a high-availability
configuration or proof of any availability target. Include NAT, EKS, CPU/GPU, volumes,
data services and storage in cost evidence. RDS deletion protection and a final snapshot
require deliberate teardown handling; the evidence bucket and ECR repository do not
force-delete contents.

Install the [Kubernetes stages](../kubernetes/README.md) only after foundation readiness.
The [model storage stage](../kubernetes/storage/README.md) defines an encrypted
retained PVC; verified snapshot population and actual mount checks are separate.
