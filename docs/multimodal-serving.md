# Multimodal serving

FinServe has a pretrained image-question path and a separate untrained visual-token reference. Their contracts and worker lifecycles differ.

## Pretrained vision requests

`POST /v1/vision/completions` accepts text and one inline PNG. Authentication and admission precede body upload. Its 1,450,000-byte HTTP cap accommodates a PNG of at most 1 MiB encoded as base64 plus the bounded prompt. The request deadline includes upload, parsing, preparation and response transmission.

[images.py](../src/finserve/multimodal/images.py) checks PNG structure, CRCs, dimensions and byte limits before decoding. Supported input is non-interlaced 8-bit RGB/RGBA with sides at most 512 pixels. Palette, animation and tRNS transparency are rejected. RGBA is composited on white; ancillary metadata is removed and the canonical RGB PNG is hashed. Callers cannot provide a URL for the service to fetch.

[vision_openai.py](../src/finserve/engines/vision_openai.py) sends actual chat messages with text and the canonical image data URI. It validates bounded SSE frames, terminal finish reason, authoritative completion-token usage and stream completion. The model uses its engine chat template, following vLLM's [multimodal chat input protocol](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).

Configure the vision route with the exact served model and pinned revision.

## Stage experiment and quality

[preprocess_http.py](../src/finserve/multimodal/preprocess_http.py) provides a separate authenticated CPU PNG preparation service. The experiment compares local normalization with a real HTTP round trip carrying the identical canonical PNG. It records worker time and caller round-trip time, validates media type and artifact hash, and rejects impossible timing relationships. Their difference includes transport, scheduling and client work; it is not pure network time.

Vision encoding and language decoding remain together inside vLLM. There is no measured independent encoder GPU pool or tensor-transfer path.



## JAX/Flax visual-token reference

[jax_generator.py](../src/finserve/multimodal/jax_generator.py) conditions a seeded recurrent decoder on an 8-by-8 RGB image and its previous output token. It emits 64 IDs from a 16-color palette and converts them to a PNG. Flax owns parameters; JAX executes fixed-shape scan and compiled paths on CPU. Tests compare exact tokens, image conditioning and baseline/compiled parity. Timing waits for device completion and separates compilation from warmed execution.

The parameters are random and untrained. Palette output demonstrates autoregression and compilation mechanics, without useful image-synthesis claims. Numerical batches support one to eight images; each durable job accepts one image.

## Durable job boundary

[jobs.py](../src/finserve/multimodal/jobs.py) persists SQLite intent and tenant-bound idempotency before returning acceptance. States include queued, running, succeeded, failed, cancel requested and cancelled. Disconnecting a status poll does not cancel an accepted job. Explicit cancellation fences late output; generation claims prevent old attempts from overwriting newer state.

[visual_rpc.py](../src/finserve/multimodal/visual_rpc.py) isolates JAX dependencies in a real gRPC worker. Its stream reports admission, then a terminal PNG and SHA-256 or a typed failure. Worker-instance barriers distinguish cancellation acknowledgement from an unknown attempt. Native generation drains before capacity is freed, including deadline and disconnect paths. Small PNG bytes are in-band; large-artifact transport is outside this reference design.

This is local SQLite with one coordinator. Restart ambiguity remains inspectable and fails closed; unknown old attempts are not silently retried or declared drained. Redis/RDS/S3-backed production job execution is not demonstrated by this implementation.
