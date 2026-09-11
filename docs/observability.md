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

The separate [local Langfuse stack](../infra/docker/langfuse.md) has verified actual OTLP ingestion and an authenticated database query for the same trace/span IDs. The stored synthetic span retained model and token-count metadata and omitted input/output, event text and an injected private marker. Langfuse 4.33.0 requires the explicit `langfuse-v4` protocol option: it returns a bounded JSON queue acknowledgement instead of the standard OTLP protobuf reply. A queue acknowledgement alone does not prove stored data; the probe verifies that separately.

All six local services stopped after the probe, with volumes retained. Web and worker exceeded the 30-second graceful-stop budget and exited 137; no graceful worker shutdown under load is claimed. Hosted Langfuse and CloudWatch ingestion remain unverified.

A six-cohort real HTTP fixture experiment compared tracing off, 1% local JSON sampling and full local JSON tracing in a balanced sequence. All 6,144 measured requests completed and both full-trace cohorts exported all 1,056 spans including warmup. Per-cohort throughput ranged from 49.25 to 57.00 requests/s. Workstation and mounted-file I/O variation prevent a defensible isolated overhead estimate from this sequence. It does not measure GPU, Ray or hosted OTLP overhead. The source-controlled harness is `scripts/measure_trace_overhead.py`; raw evidence remains in `trace-overhead-01`.

The [Compose monitoring override](../monitoring/README.md) provisions a text-gateway dashboard and four alert rules. Pinned Promtool checks passed. A live local probe verified actual scrapes, the provisioned Grafana datasource/dashboard API, a 30-second scrape-outage alert, missing-data masking and recovery. All three owned test containers stopped with exit code 0. Browser rendering and notification delivery remain separate checks. Vision, visual jobs and explorer metrics are not represented by the text gateway registry.

Implementation follows the [OpenTelemetry Python propagation API](https://opentelemetry.io/docs/languages/python/propagation/) and [official OTLP exporter](https://opentelemetry-python.readthedocs.io/en/latest/exporter/otlp/otlp.html). See `telemetry/tracing.py`, `telemetry/propagation.py` and the actual Ray/HTTP integration tests for FinServe's narrower supported boundary.
