# Deployment

The [free static Hugging Face Space](https://huggingface.co/spaces/chinmayarvind/finserve) is live. It serves measured aggregate results, not inference or an API. The full explorer and GPU engines run locally; follow the [free setup guide](run-free.md). AWS/Kubernetes configuration is validated but not deployed. No billed GPU cost or cloud rollback result is claimed, and paid hosting is outside the current deployment scope.

## Local runtime

The [Compose stack](../infra/docker/README.md) runs FastAPI and ephemeral Redis, with an optional Prometheus/Grafana override. Its default engine is a transport fixture. GPU engines, Ray, the gRPC visual worker, registry and Bun services have separate processes and explicit credentials.

The runtime producer fetches a pinned model snapshot, verifies each file, builds from an exact Git archive and records the resulting image identities. Its entrypoint rechecks the mounted model and canonical profile before engine startup. Native experiments completed before this producer keep their original undeclared image fields.

Managed launch/stop stages bind those upstream receipts to the exact Docker image,
container ID and start time. Readiness uses a real bounded inference stream;
completed replay checks the existing start without creating a replacement.
The local cold-start proof took 152.970 seconds and verified reconciliation and
cleanup. That measurement is distinct from warm rollback.

## Hugging Face and local explorer

The [Space package](../infra/huggingface/README.md) reuses the Bun explorer and
Python read service. An explicit allowlist includes source, locks and six committed
aggregate assets. Its public landing page states the failed quality gate; the
authenticated registry starts empty on ephemeral storage. Local container checks
cover actual HTTP, auth/body limits and normal/faulted child shutdown. The account's
Docker Space creation request returned HTTP 402 requiring PRO, so this container
remains a local option. No subscription or paid hardware was provisioned.

The separate `--static` package was uploaded from source commit `16d9794` to Space
commit `cce3ea8af83a1f31cc6960c802e028187e718490`. Hugging Face reports `RUNNING`.
Actual HTTPS checks verified the CSS and all six aggregate assets byte-for-byte.
The HTML matches the uploaded source after removing the single observed
Hugging Face creator-variable injection. The static site includes no private
request records, credentials, registry, backend or model weights. Browser visual
verification remains pending.

## AWS and Kubernetes

The [Terraform foundation](../infra/terraform/README.md) describes private EKS networking, bounded CPU/GPU groups, RDS, Redis, S3, ECR and workload identities. The [Kubernetes stages](../infra/kubernetes/README.md) install KubeRay/device-plugin operators before Ray and GPU engine workloads. The current chart serves one external GPU engine through CPU Ray proxies; extra proxies are not extra model capacity.

The engine chart runs the verified producer entrypoint with an expected canonical profile hash, model, internal endpoint and credential environment. It mounts an existing model PVC and immutable profile ConfigMap read-only, then verifies actual model bytes before startup. An optional FastAPI Deployment exposes text/chat SSE through a private Service and enforces required ingress/Ray credentials. Local Helm tests render both modes against the actual pinned RayService schema. Existing storage and credentials, authenticated inference, and cluster enforcement still need deployment evidence.

Provider mocks and Helm/CRD validation verify configuration contracts. Deployment still requires a valid account session, private-cluster connectivity, region-specific image/add-on resolution, application secrets, storage, TLS ingress and actual service checks. Keep Terraform state, plans and credentials outside the source checkout.

The staging foundation uses a single NAT gateway, single-AZ RDS, one Redis node and a GPU group bounded to one node. These are explicit resource constraints, not high-availability guarantees. The existing GPU Deployment uses Recreate and cannot provide an overlapping warm canary on that single GPU allocation.

## Release and rollback

The shared release gate verifies immutable identities and recomputes quality/performance before approval. A failed quality gate blocks promotion even when throughput improves. Airflow and CLI call the same implementation.

The implemented warm-route controller switches already-running endpoints with expected revision/generation checks. In-flight requests retain their original backend. Readiness requires a successful inference stream and exact revision verification after cutover. The local fixture drill measured 0.680 seconds from detection to verified recovery; it excludes image pulls, weight loads and node replacement. A separate cold/cloud experiment is required to assess the 94-second target.

Replica scaling, engine process capacity and GPU node scaling require separate evidence. Multi-engine capacity experiments and the artifact-bound end-to-end lifecycle are in progress. Current [results](results.md) identify the scope of completed checks.
