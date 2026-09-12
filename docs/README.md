# FinServe documentation

Start with local setup, then follow a request through the architecture and service contracts.

| Task | Guide |
| --- | --- |
| Start the application | [Local setup](run-free.md), [commands](commands.md) |
| Understand service boundaries | [High-level design](HLD.md), [low-level design](LLD.md) |
| Call the APIs | [API contracts](api-contracts.md), [multimodal serving](multimodal-serving.md) |
| Understand routing | [Scheduling and batching](scheduling-and-batching.md), [inference fundamentals](inference-fundamentals.md) |
| Operate releases | [Quality gates](quality-gates.md), [Airflow pipeline](airflow-pipeline.md), [registry](data-and-registry.md) |
| Deploy and observe | [Deployment](deployment.md), [observability](observability.md), [security](security.md) |
| Diagnose failures | [Failure modes](failure-modes.md), [queue saturation](runbooks/queue-saturation.md), [GPU memory](runbooks/gpu-oom.md), [rollback](runbooks/rollback.md) |

Service-specific commands live beside the [Bun edge](../apps/api/README.md), [explorer](../apps/web/README.md), [visual worker](../src/finserve/multimodal/README.md), and [container definitions](../infra/docker/README.md).
