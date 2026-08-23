# Architecture Alternatives and Tradeoffs

## Build an inference engine from scratch?

Build a small PyTorch reference server for KV cache, decode loops, batching, and streaming. Do not pretend it replaces mature vLLM/SGLang optimization.

## vLLM only?

Simple and powerful, but SGLang comparison provides engine-abstraction and benchmarking signal. Decision: vLLM primary, SGLang comparison.

## Kubernetes directly versus Ray Serve?

Direct Kubernetes would require more bespoke application-level routing and replica logic. Ray Serve is selected for distributed serving after the single-engine baseline.

## Redis queue?

Rejected for synchronous streaming text. Selected for rate limits, cache, ephemeral routing metadata, and explicit async visual jobs.

## Multimodal universal worker?

Rejected. Text, JAX visual research, and production multimodal workers can use different pools while sharing control/observability.

## JAX role

JAX is not decorative. It owns the multimodal autoregressive reference/comparison path where compilation and functional execution matter.

## GraphQL role

Rejected for inference. Selected for dashboard/deployment/benchmark nested reads.

## Airflow role

Rejected for online request scheduling. Selected for model fetch/verify/optimize/build/eval/deploy workflow.

## Elasticsearch, Spark, Kafka

Not part of MVP. Add only if offline benchmark/event/search volume earns them.

## Ollama

Useful local smoke/baseline path, not evidence for distributed headline metrics.

## TensorRT-LLM

Optional optimization comparison when hardware/runtime/project time permit. It can be an Airflow optimization branch without becoming a mandatory serving dependency.
