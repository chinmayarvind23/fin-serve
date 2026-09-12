# Deployment

## Local processes

Follow [local setup](run-free.md) for a CPU explorer and a separate NVIDIA inference engine. The [Compose stack](../infra/docker/README.md) provides FastAPI and Redis, with an optional monitoring override. Its default engine is a transport fixture.

The [Bun edge](../apps/api/README.md), [explorer](../apps/web/README.md), [gRPC visual worker](../src/finserve/multimodal/README.md), and [Airflow worker](../pipelines/airflow_dags/README.md) have separate entrypoints. Use distinct credentials for public ingress, internal engine traffic, and explorer access.

The runtime producer downloads a pinned model manifest, verifies file bytes, builds an engine image from an exact Git archive, and records the resulting Docker identity. Startup rechecks the mounted model and canonical profile. Managed launch and stop stages bind receipts to the exact container; readiness requires a bounded inference request.

## Explorer container and static site

The [explorer container](../infra/huggingface/README.md) runs Bun and a loopback Python read service. Its initial registry is empty. Mount persistent state and import your own runs for durable use. The landing page explains the application and links to the authenticated explorer.

The static package contains a product introduction and local setup instructions. It runs no model, registry, or API. Build it with `python infra/huggingface/package.py --static --output /absolute/external/space`.

## AWS and Kubernetes configuration

The [Terraform foundation](../infra/terraform/README.md) defines EKS networking, CPU/GPU node groups, data services, artifact storage, and workload identities. Follow the [Kubernetes guide](../infra/kubernetes/README.md) to install operators before workloads. These are operator-run deployment definitions; local setup uses Docker.

Prepare account credentials, private cluster access, application secrets, model storage, TLS ingress, and region-compatible images before applying. Keep Terraform state and plans outside Git. Validate authenticated inference and storage permissions in the target environment.

The engine chart mounts a model PVC and immutable profile configuration, then verifies their identity at startup. CPU Ray proxies route to the external engine. Scale engine processes and GPU nodes separately from proxy replicas. The single-GPU Recreate deployment requires a maintenance window for engine replacement.

## Release operations

The shared gate verifies model, tokenizer, image, and configuration identity before approving a candidate. The warm-route controller uses expected revision and generation checks to switch already-running endpoints. In-flight requests retain their original backend. Rollback rechecks the selected revision and its health.

Store release artifacts on a persistent volume or authorized object store. Reconcile interrupted operations before retrying. See [quality gates](quality-gates.md) and the [rollback runbook](runbooks/rollback.md).
