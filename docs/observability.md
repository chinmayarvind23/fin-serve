# Observability

Gateway metrics, physical GPU samples and distributed traces answer different questions. Operational telemetry is separate from the immutable benchmark record. An exporter failure cannot turn missing samples into a valid performance claim.

## Verified trace boundary

The text path now carries a linked OpenTelemetry identity from FastAPI through actual Ray routing and worker processes to the engine HTTP request. A real CPU Ray/HTTP integration test checks the exported trace IDs and parent span IDs. This verifies transport causality and cleanup; it does not expose GPU kernel, prefill or decode spans inside vLLM.

The gateway starts its own root rather than accepting an external caller's parent/sampling decision. Private hops carry only a bounded W3C `traceparent`. Baggage and vendor trace state are excluded. Async generators attach context only while awaiting their next item or closing owned work; context is detached before yielding to a consumer. Concurrent requests therefore keep separate identities.

Each process owns a private provider and bounded batch queue. Sampling defaults to 1%; the queue holds at most 2,048 spans and exports batches of at most 128. Export happens outside token iteration. Shutdown drains the owned exporter off the serving event loop, and construction failures close newly allocated resources.

## Configure a sink

`FINSERVE_TRACE_PATH=/absolute/private/traces.jsonl` selects an exclusive local JSONL file. Gateway output uses that exact path. Ray route/engine components use unique sibling filenames so independent actors and restarts cannot overwrite one another.

Install the `telemetry` extra and set `FINSERVE_OTLP_ENDPOINT` to an exact OTLP/HTTP trace endpoint to use the official protobuf exporter instead. HTTPS is supported for remote collectors; HTTP is restricted to loopback. An optional `FINSERVE_OTLP_AUTHORIZATION` header stays in process memory. Do not configure both outputs. `FINSERVE_TRACE_SAMPLE_RATIO` is finite in [0,1], and `FINSERVE_OTLP_TIMEOUT_SECONDS` is finite in [0.1,10], defaulting to 3 seconds.

Collector requests disable ambient proxies and redirects. Acknowledgements are uncompressed, limited to 16 KiB and parsed as protobuf. Invalid responses, HTTP errors and partial success with rejected spans count as export failures. Actual loopback tests cover these paths and stalled/trickling responses. Native DNS/TLS and a blocking socket read are not forcibly interruptible; one read can finish after an elapsed budget by its socket timeout. The local file sink likewise has no hard native I/O shutdown deadline.

Both sinks receive sanitized spans. The exporter retains fixed span names, trace identity, times, status and a small allowlist of bounded model/count/outcome attributes. It removes event bodies, exception text, links, arbitrary resource/scope attributes and vendor trace state. Private benchmark artifacts deliberately retain prompts and outputs for grading; they are not operational trace payloads.

## Metrics and remaining integrations

The gateway publishes fixed-cardinality request, token, duration, TTFT and active-owner metrics. Request IDs and prompts are not metric labels. The routing observation path separately reads actual vLLM running/waiting/KV gauges and shared physical `nvidia-smi` measurements. A proxy lease count is neither GPU utilization nor KV occupancy.

The local Compose monitoring override configures Prometheus and Grafana. Hosted Langfuse and CloudWatch ingestion, operational dashboard/alert validation and a controlled instrumentation-overhead experiment remain incomplete. OTLP transport verification alone does not establish those deployments.

Implementation follows the [OpenTelemetry Python propagation API](https://opentelemetry.io/docs/languages/python/propagation/) and [official OTLP exporter](https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html). See `telemetry/tracing.py`, `telemetry/propagation.py` and the actual Ray/HTTP integration tests for FinServe's narrower supported boundary.
