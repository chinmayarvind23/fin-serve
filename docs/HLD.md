# High-level design

FinServe separates active inference, durable jobs, release control and evidence browsing. A slow registry query or an Airflow retry should not own an engine's token scheduler. Each boundary has its own admission, identity and failure semantics.

## Request paths and ownership

| Boundary | Owns | Does not establish |
| --- | --- | --- |
| Bun inference edge | Bearer authentication, route allowlist, bounded bytes, forwarding deadline, disconnect propagation | GPU availability or model quality |
| FastAPI ingress | Typed contracts, request deadline, optional Redis quota, application admission, SSE response ownership | Engine-level batching or hardware capacity |
| Ray HTTP router | Endpoint eligibility, routing decisions, reservations, worker lease retirement | Extra GPU capacity merely from extra CPU proxy replicas |
| vLLM/SGLang process | Model weights, prefill/decode scheduling, KV cache, engine token accounting | Quality correctness solely from successful HTTP completion |
| Image route | Bounded PNG decoding and canonicalization, image admission, image-capable backend selection | Independent vision-encoder disaggregation |
| Visual job coordinator | Durable state, idempotency, worker generation fencing and artifact completion | Pretrained image generation from the untrained JAX reference |
| Evidence service | Bounded read-only GraphQL over SQL references and checksum-verified artifacts | Permission to deploy a candidate |
| Airflow/shared gate | Frozen inputs, recomputation, durable lifecycle transitions and deployment callback | A successful cloud rollout without actual deployment evidence |

A normal text request flows through Bun and FastAPI to a configured engine, optionally via Ray. Redis can enforce a shared principal quota before admission; it is not a mandatory token queue. The engine returns visible deltas, an authoritative output-token count and a completion marker. Failure after visible output remains a failed stream rather than a transparent retry.

An image/text request uses a separate capability and admission pool. It accepts one inline PNG, strips metadata through canonical RGB conversion, then invokes a real multimodal Chat Completions endpoint. The tested stage split moves CPU normalization across an HTTP boundary. Vision encoding and language decoding remain inside one vLLM process.

A visual-generation job uses a different asynchronous API. SQLite persists accepted work and its generation fence. A gRPC worker runs the JAX/Flax reference generator and returns a verified PNG. Worker cancellation names the correct worker instance and generation; a late result cannot complete a newer job attempt.

## State placement

| State | Implemented location | Deployment distinction |
| --- | --- | --- |
| Request lease / pool ownership | Process memory; Ray actor state | Ephemeral; cleanup follows the owning request |
| Shared rate quota / optional cache | Redis Lua and expiring keys | Quota errors fail closed; optional cache errors are misses |
| Durable visual jobs | SQLite WAL with full synchronization | Requires a writable persistent volume; not Redis job truth |
| Model/run/revision/gate metadata | SQLAlchemy registry, exercised with SQLite | PostgreSQL schema support is distinct from a deployed RDS database |
| Immutable run/output artifacts | Local content-addressed store; S3 adapter | Real S3 access still requires authenticated deployment validation |
| Warm active traffic route | Separate SQLite store with CAS generations | Current local controller is not a distributed consensus database |
| Experiment tracking | MLflow adapter and local SDK execution | Remote hosting and service integration require deployment checks |
| Observations | Raw requests, GPU samples, sampled OTel spans, Prometheus metrics | Local Langfuse ingestion/query and Prometheus/Grafana alert recovery are verified; cloud hosting remains pending |

Artifacts are immutable; SQL refers to their digest, length and namespace. A native run may explicitly omit a deployment image. A canonical release profile additionally binds model/tokenizer manifests, engine parameters and endpoint identity, while its Revision binds the actual image digest. Historical native results are never relabeled with an image built later.

## Release and rollback

The shared gate materializes verified baseline/candidate evidence, recomputes performance and quality, verifies revision/profile identity, and persists either approval or rejection. Airflow and CLI use the same gate. A job freezes its required gate mode before profile publication so a crash cannot downgrade a canonical release into a legacy drill.

Producer stages precede that decision. A model stage downloads a frozen file manifest and verifies its bytes; an image stage binds those files to committed source; a quality stage records every offered response against an unchanged suite. Each stage publishes immutable artifact references through a durable attempt journal. A completed receipt can be replayed after verification. Interrupted work requires evidence that its owned resources have drained before retry; an ambiguous Docker build or failed HTTP close stays unresolved. Completing collection does not approve a release.

Managed local launch and stop stages connect the model/image receipts to an exact Docker process and a real inference readiness check. The actual cold-start proof reached readiness in 152.970 seconds and reconciled the same container start before verified removal. This timing includes local engine startup and is separate from the warm rollback measurement. The performance stage now binds recomputed benchmark evidence to observations of that same runtime start. Its integration tests use synthetic transports and Docker responses; the producer DAG is connected and scheduler-tested with fixtures, and sequential producer releases reuse a stable baseline with fresh evidence. A full live GPU DAG run remains pending.

The warm deployment adapter changes route truth with an expected revision and generation. Existing requests retain their pinned backend; new requests read the new route. Rollback is healthy only after a real inference probe observes the expected revision and the active generation remains unchanged. This operation switches running endpoints; it does not imply a bounded cold image pull, model load or node recovery.

## Infrastructure and current limits

Text clients can explicitly request a bounded output shape through the gateway, Ray and warm vLLM routing. The engine applies constrained decoding, and the adapter checks final syntax before reporting success. Shapes contain requested types and formatting rules, not expected answers; the release quality gate still decides correctness. Transport and CPU grammar checks cover this implementation. A new constraint-aware GPU experiment and its performance evidence remain pending.

Quality and performance collectors share a frozen map from exact prompt identity to requested output shape. The producer preserves that map while deriving actual runtime identities from model and image receipts. Missing bindings and changed mappings reject collection or comparison. Synthetic integration covers the producer path; the map does not retroactively change earlier measurements or establish that its author selected shapes independently of answers.

The AWS foundation specifies private EKS networking, bounded CPU/GPU node groups, RDS, Redis, S3, ECR and workload identities. Helm separates CPU Ray proxies from GPU engine Deployments. Local configuration and mocked provider checks pass; authenticated AWS/Hugging Face deployment remains incomplete. A Docker image and a Terraform plan are not proof of a running cloud service.

The Hugging Face deployment package serves the existing evidence explorer on CPU with a public aggregate-results page and an authenticated, initially empty registry. Its local container passed actual HTTP/authentication and shutdown checks. It contains no GPU inference service or private raw evidence. Hosted deployment and browser rendering remain pending.

Local experiments include a real Ray-to-GPU path, 6,144 measured text requests, pretrained image/text inference, actual gRPC reference jobs and an HTTP warm rollback drill. After the first two-engine session failed availability, bounded admission waiting completed a separate 256/256-request run on the same workload. Scale-down, restart and failure-after-content drills also produced retained evidence, with driver-wrapper shutdown ambiguity recorded. This establishes a bounded local two-process path; artifact-bound end-to-end deployment, cloud hosting and the browser demo are still being completed. [Results](results.md) states which measured gains failed the quality gate.


## CPU worker scaling boundary

The Kubernetes chart offers opt-in KubeRay V2 scaling between one and two CPU worker Pods per RayCluster. Ray actor/task demand drives this loop. The current fixed router and one proxy per engine fit on a single worker; higher HTTP load alone does not add actors or GPUs. The single routing authority and engine admission limits remain intact. Two fixed workers remain the default. GPU engine scaling, node scaling and live scaling evidence are still incomplete. The dedicated head account has namespaced API access; worker and engine accounts remain tokenless. RayService upgrades can overlap clusters, so the per-cluster maximum does not bound total upgrade resources. See infra/kubernetes/README.md for the resource budget and validation scope.


## Node scaling foundation

The AWS foundation now separates node desired-size ownership from Terraform configuration. CPU nodes are bounded at two to three; GPU nodes at zero to one in the first configured AZ, matching retained zonal model storage. Terraform initializes desired sizes and continues to own bounds, while subsequent desired-size changes belong to a controller or explicit EKS operations. Opt-in node autoscaling creates scoped IRSA and discovery metadata on actual managed ASGs. It does not install Cluster Autoscaler, scale model replicas, or establish live capacity evidence. The separate controller and pending-Pod/drain tests remain required.
