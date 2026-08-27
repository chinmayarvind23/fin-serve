# Observability

## Questions

Where did latency occur? Was queueing at ingress/Ray/engine? Was GPU underfed or saturated? Did speculation accept useful tokens? Which replica served the request? Did autoscaling lag? Why did a version roll back?

## OpenTelemetry trace

```text
HTTP inference
  +-- admission
  +-- model routing
  +-- replica routing
  +-- engine request
      +-- queue wait
      +-- prefill
      +-- decode
  +-- stream completion
```

Carry request ID, trace ID, deployment revision, model, engine, replica, benchmark ID. Avoid raw sensitive prompts by default.

Use current GenAI semantic conventions where stable and pin versions.

## Prometheus

Request counts/failures/duration, TTFT, generated tokens, queue depth, ongoing requests, model-load time, rollback counts, engine KV/cache/speculation metrics, GPU utilization/memory.

Do not use request IDs/prompts as labels.

## Langfuse

Use for inference/model/eval trace exploration where it improves debugging. It complements OTel rather than replacing infrastructure tracing.

## Grafana/CloudWatch

Grafana: benchmark latency/throughput/GPU/Ray/speculation dashboards.

CloudWatch: AWS/EKS/ALB/system alarms and deployment signals.

## Sampling

100% traces for small evals; sampled traces for heavy load. Measure instrumentation overhead explicitly.
