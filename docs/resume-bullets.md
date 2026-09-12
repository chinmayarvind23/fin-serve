# FinServe

PyTorch, JAX/Flax, Ray Serve, vLLM/SGLang, Redis, FastAPI/gRPC, MLflow, Airflow, EKS/KubeRay, Terraform

- Built a distributed text and multimodal inference platform with adaptive scheduling, continuous batching, GPU-aware routing, and speculative-decoding experiments; measured **2.64x higher token throughput** with compiled serving in local benchmarks.
- Benchmarked **6,144 GPU inference requests with 100% completion**, reducing **p95 latency by 62.3% (2.13 s to 0.80 s)**; a separate 512-request prefix-cache trial reduced **median client TTFT from 247 ms to 116 ms** on a local RTX 4070 Laptop GPU.
- Automated model evaluation, gated deployment, and warm rollback using MLflow and Airflow; added **EKS/KubeRay and Terraform infrastructure definitions** and verified platform behavior with **1,134 passing CPU tests**.

## Evidence for interviews

The sustained performance figures come from one ordered RTX 4070 Laptop GPU comparison using the same pinned model and workload, with 3,072 measured requests and 64 separate warmups per configuration. Token throughput was 373.52 versus 987.54 tokens/s; p95 was 2.13148 versus 0.804339 seconds. This candidate failed its separate quality gate. The bullets describe measured speed, not quality parity or a production-qualified release. See [results](results.md) and [measurement definitions](benchmark-methodology.md).

The CPU count is from clean source `66b5501`: 1,134 passed and 49 optional checks skipped. Combined statement/branch coverage was 88.20%. The full real-GPU Airflow lifecycle passed on source `3069a28`, including model/runtime production, canonical gates, deployment, 60 probation probes and baseline cleanup. Both engines passed the unchanged 32-case consumed release suite, with 100% candidate accuracy and baseline parity; this is not fresh-holdout quality. See [architecture](HLD.md) and [recorded demo scope](demo.md).

The separate prefix-cache trial used 256 measured requests and 16 warmups per arm on Qwen2.5-3B-Instruct-AWQ, with identical 512 MiB cache budgets. Median client TTFT was 247.429 versus 116.072 ms, a 53.1% reduction. This ordered, shared-GPU operational trial is separate from the sustained compiled comparison, whose median server TTFT worsened from 105.30 to 128.57 ms. The 100% completion figure covers the finite local benchmark, not a production availability SLA. Speculative decoding was evaluated separately and rejected; it did not produce the 2.64x gain. EKS/KubeRay/Terraform are infrastructure definitions, not a claimed live AWS deployment.

The original request-throughput, TTFT improvement, cloud cost saving, quality parity, utilization, availability, and GPU rollback targets are omitted. No additional benchmark is required to use these scoped bullets.
