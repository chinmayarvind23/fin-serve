# API contracts

The gateway exposes a bounded subset of the OpenAI text/chat protocol. It does not implement every OpenAI parameter. Unknown request fields are rejected by the corresponding typed contract.

| Route | Behavior |
| --- | --- |
| `GET /healthz` | Process liveness; does not prove engine inference readiness |
| `GET /metrics` | Fixed-cardinality Prometheus metrics; keep internal to the service network |
| `POST /v1/completions` | Bounded prompt, model, output count, temperature and deadline; SSE or collected JSON |
| `POST /v1/chat/completions` | System/user/assistant text messages preserved for the backend chat template |
| `POST /v1/vision/completions` | Optional authenticated single-PNG image plus text capability |
| `GET /v1/visual/models` | Optional durable-job worker revision and supported image shape |
| `POST /v1/visual/jobs` | Authenticated durable submission with an idempotency key |
| `GET /v1/visual/jobs/{job_id}` | Tenant-scoped job status |
| `DELETE /v1/visual/jobs/{job_id}` | Explicit fenced cancellation intent |
| `GET /v1/visual/jobs/{job_id}/artifact` | Verified PNG after successful durable completion |

Set `FINSERVE_API_KEY` to protect text inference; image capabilities require it. Current authentication uses a configured service credential, not a user/role management system. Public clients cannot deploy through these routes. Release operations use the separate operator CLI/Airflow path and generation-fenced deployment adapter.

Text bodies are limited to 128 KiB. Chat additionally bounds the duplicated internal prompt/message payload so it fits the Ray transport. The image route permits a larger envelope, but caps the decoded PNG at 1 MiB and 512 pixels per side. Its request fields include `prompt`, `image_png_base64`, `model`, `max_tokens`, `timeout_seconds` and `stream`; arbitrary image URLs and batches are excluded. See `contracts/inference.py`, `contracts/vision.py` and the gateway's generated `/docs` for precise fields.

Successful streaming returns escaped JSON SSE, authoritative engine usage and `[DONE]`. An error after visible output remains a failed stream and is never retried transparently. Nonstream responses consume the same owned iterator. Error codes are stable; status, retryability and request identity depend on the relevant route. A normal text overload is HTTP 429 with code `OVERLOADED`.

The [evidence explorer](../apps/web/README.md) runs as a separate authenticated POST `/graphql` service. Its read-only schema exposes registered runs, bounded request pages, quality/GPU annotations and lifecycle history. It rejects mutations, introspection, fragments and queries exceeding its depth, expansion or byte limits. There are no public REST benchmark/deployment CRUD endpoints.

The gRPC protocol serves the [durable visual worker](../src/finserve/multimodal/README.md). Ray's internal endpoint uses typed NDJSON between trusted services. Neither internal transport is the public SSE protocol.
