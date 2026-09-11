# Telemetry

Fixed-cardinality Prometheus metrics and sampled OpenTelemetry spans have separate owners. Text tracing propagates through private Ray/HTTP boundaries; JSONL and OTLP/HTTP sinks receive sanitized metadata. See [observability](../../../docs/observability.md) for setup, verified scope and native I/O limits.
