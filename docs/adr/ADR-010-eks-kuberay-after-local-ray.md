# ADR 010: Validate local Ray before EKS

## Decision

Keep native engine and Ray HTTP integration independently testable before adding the AWS control plane. Provision AWS foundations separately from Kubernetes operators and application workloads.

## Implementation and limits

The Terraform foundation defines private EKS, bounded CPU/GPU node groups, RDS, Redis, evidence storage and scoped IAM. The Helm stages install pinned operators, then a RayService plus a separately managed GPU engine. Two Ray CPU worker Pods do not represent two model replicas.

Local Ray inference, Terraform validation/mock tests and Helm/CRD schema checks have passed. Authenticated AWS plan/apply, private-network access, GPU node allocation and deployed inference remain separate acceptance work. The single-GPU staging engine uses a disruptive Recreate update; it cannot provide overlapping warm canary capacity.

This ordering lets transport and ownership defects be reproduced without a cluster, while preserving the need to test cloud-specific permissions, networking, storage and recovery. See [deployment](../deployment.md).
