# Stack and claim verification

FinServe's stack has separate serving, evaluation, control and infrastructure responsibilities. The full local GPU release and automatic model-replica lifecycle passed. The links below let a reviewer trace each technology to its implementation and verification scope.

## Original stack

| Technology | Implemented responsibility | Code and verification |
| --- | --- | --- |
| PyTorch | Inspectable causal decoder, cached/uncached equivalence and cancellation ownership | [Reference engine](../src/finserve/engines/pytorch_reference.py), [tests](../tests/unit/test_pytorch_reference.py). Random reference weights; pretrained performance belongs to vLLM. |
| JAX / Flax | Image-conditioned reference visual-token generation | [Generator](../src/finserve/multimodal/jax_generator.py), [tests](../tests/unit/test_jax_generator.py). Real CPU JAX/gRPC execution appears in the demo; this is an untrained reference generator. |
| Ray Serve | Distributed HTTP routing, per-engine eligibility and leases | [Serve deployment](../src/finserve/engines/ray_serve.py), [backends](../src/finserve/engines/ray_backends.py), [integration](../tests/integration/test_ray_backends.py). CPU proxy replicas and model/GPU capacity are counted separately. |
| vLLM | Pretrained GPU execution, continuous batching, compiled serving and prefix caching | [Adapter](../src/finserve/engines/vllm_adapter.py), [pinned runtime](../src/finserve/registry/engine_entrypoint.py), [actual measurements](results.md). The separate n-gram speculation trial was rejected. |
| SGLang | Alternative external engine using the same bounded request/accounting contract | [Adapter](../src/finserve/engines/sglang_adapter.py), [shared protocol tests](../tests/contract/test_engine_adapter.py). Published speed gains are vLLM results, not a vLLM-versus-SGLang comparison. |
| Redis | Optional gateway admission quotas; separate cache primitives | [State adapter](../src/finserve/cache/redis_state.py), [gateway wiring](../src/finserve/gateway/app.py), [integration](../tests/integration/test_redis.py). Quotas are wired into serving; cache helpers are library/test components. Redis is not the streaming token queue or durable release database. |
| FastAPI / gRPC | Authenticated HTTP/SSE ingress and a separate visual-worker RPC boundary | [Gateway](../src/finserve/gateway/app.py), [visual RPC](../src/finserve/multimodal/visual_rpc.py), [RPC integration](../tests/integration/test_visual_rpc.py), [demo](demo.md). |
| MLflow | Optional mirror of verified run artifacts and recorded gate decisions | [Mirror](../src/finserve/registry/mlflow.py), [operator CLI](../src/finserve/registry/mlflow_cli.py), [real local SDK integration](../tests/integration/test_registry_optional.py). SQL and content-addressed artifacts retain release authority. |
| Airflow | Offline model/build/collection/gate/deployment/probation/cleanup orchestration | [Producer DAG](../pipelines/airflow_dags/finserve_producer.py), [pipeline](../src/finserve/registry/producer_pipeline.py), [passed GPU run](results.md). Airflow is outside online token serving. |
| EKS / KubeRay | Optional GPU/CPU cluster placement, RayService and bounded infrastructure scaling | [EKS foundation](../infra/terraform/foundation/cluster.tf), [RayService](../infra/kubernetes/workload/templates/ray.yaml), [deployment guide](deployment.md). Configuration is locally validated; AWS was not deployed. |
| Terraform | AWS networking, state services, storage and scoped workload identities | [Foundation](../infra/terraform/foundation/), [provider-mocked tests](../infra/terraform/foundation/tests/foundation.tftest.hcl). No cloud billing or high-availability claim. |

## Supporting stack from the project plan

- **Docker** isolates model engines and binds launches to exact image/container/start identities. See [runtime builds](../src/finserve/registry/runtime_build.py).
- **SQL, PostgreSQL/RDS and S3** separate release metadata from immutable artifacts. Local SQLite is the verified control path; PostgreSQL/S3 adapters and optional RDS/S3 infrastructure have their own verification scope. See [data and registry](data-and-registry.md).
- **Bun, TypeScript and GraphQL** provide the edge and bounded read-only evidence explorer. See [web application](../apps/web/README.md) and [free setup](run-free.md).
- **OpenTelemetry, Prometheus and Grafana** cover traces, counters and dashboards. Local checks are described in [observability](observability.md); hosted services are not implied.
- **pytest, Ruff, Pyright and CI gates** verify contracts, control logic and code quality. The retained full CPU result is 1,134 passing tests on `66b5501`, with 49 optional checks skipped; later focused checks have separate receipts.
- **Ollama, TensorRT-LLM and Elasticsearch** were conditional research ideas in the plan. They are not claimed as implemented FinServe serving components. Adding an unused dependency would not establish an integration.

## Original targets and retained measurements

| Original target | Verified evidence |
| --- | --- |
| 38 to 94 sustained requests/s | Historical ordered comparison: 12.99 to 34.59 requests/s. The target was not achieved. |
| 2.4x token throughput | Historical compiled comparison: **2.64x**, measured from generated tokens and run durations. That candidate failed its separate quality gate. |
| 6,000+ requests | **6,144 measured requests**, plus 128 separate warmups, in the sustained comparison. |
| 690 to 295 ms median TTFT | Separate 512-request prefix-cache trial: **247 to 116 ms client TTFT**. These are different baseline values and a different experiment. |
| 3.8 to 1.9 s p95 | Historical sustained comparison: **2.131 to 0.804 s**, a **62.3% reduction**. |
| 37% lower GPU cost per million tokens | No measured cloud cost comparison. Local zero provider charges are not zero operating cost. |
| 99.2% output-quality parity | Latest release: **100% baseline parity on 32 consumed cases**. This does not establish broad or fresh-holdout parity. |
| 99.95% successful requests under load | 6,144/6,144 completed in the finite sustained comparison; the separate capacity stimulus retained 1,170 failures. No production availability claim. |
| 81% mean GPU utilization | Historical candidate: **57.91%** time-weighted physical GPU utilization. |
| GPU rollback within 94 seconds | Warm HTTP recovery was demonstrated with synthetic backends; no equivalent cold-GPU/cloud recovery measurement establishes this target. |

The [results](results.md) and [methodology](benchmark-methodology.md) define populations and limitations. Code establishes mechanisms; measured runs establish numerical outcomes. These rows must not be combined into one nonexistent experiment.

## Latest integration check

The operator MLflow command exported the retained successful GPU candidate run into a real local MLflow experiment. Readback verified `FINISHED` status, the recorded accuracy/parity metrics, all three run artifacts byte for byte, and the exact decision document. The [verification receipt](assets/mlflow-release-verification.json) records the tracking/run identities and hashes. This export reused existing GPU evidence and made no new inference calls. The original registry was opened read-only for a consistent copy and left untouched.
