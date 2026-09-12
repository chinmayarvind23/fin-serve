# FinServe design and evidence guide

FinServe combines GPU inference, reproducible measurement and automated release control. The local GPU release and automatic replica lifecycle have passed verification. Start with the demo and results, then follow a request through the design.

| What you want to understand | Read |
| --- | --- |
| See it working and run it yourself | [Demo](demo.md), [free local setup](run-free.md) |
| Understand the measured improvements | [Results](results.md), [benchmark methodology](benchmark-methodology.md) |
| Explain the architecture and state placement | [HLD](HLD.md), [LLD](LLD.md), [system design](system-design.md) |
| Defend the technology choices | [Architecture alternatives](architecture-alternatives.md), [decision records](adr/) |
| Follow streaming, routing and cancellation | [API contracts](api-contracts.md), [scheduling and batching](scheduling-and-batching.md) |
| Explain evaluation, promotion and rollback | [Quality gates](quality-gates.md), [Airflow pipeline](airflow-pipeline.md), [failure modes](failure-modes.md) |
| Deploy and observe the system | [Deployment](deployment.md), [observability](observability.md), [security](security.md) |
| Present the project | [Resume bullets](resume-bullets.md) |

Historical rejection labels refer to individual experiments. The latest passed 3B release, historical compiled-serving speed results and broader 7B quality study have separate evidence populations. Optional AWS infrastructure is provided as setup definitions; the public Hugging Face site is free and static.
