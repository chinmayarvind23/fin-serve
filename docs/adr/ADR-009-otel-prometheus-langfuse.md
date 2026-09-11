# ADR 009: Separate operational telemetry from release evidence

## Decision

Use bounded OpenTelemetry traces for private request causality, Prometheus for process metrics, and Langfuse as an optional trace exploration sink. Immutable benchmark records remain the source for release timing, quality and identity checks.

FastAPI starts a fresh trace root. Internal HTTP and Ray calls propagate only W3C trace identity. Each process owns its provider; generator context detaches before yielding, and cleanup retains exporter ownership. Sanitization removes prompt/output bodies, exception text, event payloads and arbitrary labels before export.

## Evidence and tradeoffs

An actual CPU Ray/HTTP integration verifies the gateway, route and engine parent chain. OTLP tests exercise a real loopback collector, bounded acknowledgements, partial rejection and failed transport. These spans describe service boundaries; they do not reveal vLLM kernel or decode-stage timings.

Prometheus histograms support operational estimates with fixed labels. The benchmark uses exact raw timestamps and its frozen population instead. Sampling and bounded export queues limit telemetry work but may omit observations. Exporter shutdown and native I/O have explicit limits documented in [observability](../observability.md). Local Langfuse ingestion is verified through its authenticated observations API, with private content absent. Its web/worker shutdown exceeded the 30-second budget. A balanced local JSON tracing experiment retained all requests and full-sample spans, but workstation/I/O variation prevents an isolated overhead estimate. Hosted sinks and operational dashboard behavior require separate execution evidence.
