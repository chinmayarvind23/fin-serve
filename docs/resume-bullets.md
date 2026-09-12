# FinServe

PyTorch, JAX/Flax, Ray Serve, vLLM/SGLang, Redis, FastAPI/gRPC, MLflow, Airflow, EKS/KubeRay, Terraform

- Built a distributed text and multimodal inference platform with adaptive scheduling, continuous batching, GPU-aware routing, and speculative-decoding experiments; measured **2.64x higher token throughput** with compiled serving in local benchmarks.
- Benchmarked **6,144 GPU inference requests with 100% completion**, reducing **p95 latency by 62.3% (2.13 s to 0.80 s)** and measuring **129 ms median server TTFT** on a local RTX 4070 Laptop GPU.
- Automated model evaluation, gated deployment, and warm rollback using MLflow and Airflow; added **EKS/KubeRay and Terraform infrastructure definitions** and verified platform behavior with **1,134 passing CPU tests**.

## Evidence for interviews

The performance figures come from one ordered RTX 4070 Laptop GPU comparison using the same pinned model and workload, with 3,072 measured requests and 64 separate warmups per configuration. Token throughput was 373.52 versus 987.54 tokens/s; p95 was 2.13148 versus 0.804339 seconds. This candidate failed its separate quality gate. The bullets describe measured speed, not quality parity or a production-qualified release. See [results](results.md) and [measurement definitions](benchmark-methodology.md).

The CPU count is from clean source `66b5501`: 1,134 passed and 49 optional checks skipped. Combined statement/branch coverage was 88.20%. Release automation is implemented and integration-tested; the real GPU Airflow attempt retained all baseline requests but failed a clock-consistency check before candidate collection. Do not describe that attempt as a successful GPU deployment. See [architecture](HLD.md) and [recorded demo scope](demo.md).

Median server TTFT of 129 ms is the compiled run's rounded absolute measurement; it is not an improvement claim, since the eager baseline measured 105.30 ms. The 100% completion figure covers the finite local benchmark, not a production availability SLA. Speculative decoding was evaluated separately and rejected; it did not produce the 2.64x gain. EKS/KubeRay/Terraform are infrastructure definitions, not a claimed live AWS deployment.

The original request-throughput, TTFT improvement, cloud cost saving, quality parity, utilization, availability, and GPU rollback targets are omitted. No additional benchmark is required to use these scoped bullets.
