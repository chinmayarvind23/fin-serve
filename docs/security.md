# Security

FinServe is not primarily an autonomous agent, but transferable production principles apply: typed contracts, least privilege, external enforcement, state reliability, observability, and governance.

## Roles

`inference_client`, `benchmark_operator`, `model_operator`, `deployment_operator`, `admin`.

Inference credentials cannot deploy/delete models.

## Model supply chain

Allowlisted source, pinned revision, checksum, license record, safe archive scan, explicit review for remote model code, immutable artifact record.

## Container supply chain

Pinned bases, dependency scan, SBOM, image digest, non-root where compatible, managed secrets, no cloud credentials in images.

## Input controls

Max bytes/context/image size/output tokens/concurrency/timeouts prevent trivial resource exhaustion.

## Streaming

Disconnect/cancellation must abort expensive generation.

## Redis/RDS

Private networking, auth/TLS where supported, TTL for ephemeral state, parameterized SQL, least-privilege DB role. RDS must not be required per token.

## Kubernetes

Service-account/RBAC boundaries, NetworkPolicy, pod security, resource requests/limits, managed secrets, controlled ingress.

## GraphQL

Auth, resolver authorization, depth/complexity limits, pagination.

## Logging

Redact authorization headers, provider/model tokens, sensitive prompt/media, signed URLs.
