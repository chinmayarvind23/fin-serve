# Multimodal services

FinServe has two different image paths. Pretrained image understanding accepts one PNG and a text prompt through a vision-capable vLLM backend. Durable visual generation uses an untrained JAX/Flax reference model and a separate gRPC worker. Neither result should be presented as evidence for the other.

## Durable JAX/Flax jobs

Use a Linux Python 3.12 environment for the tested JAX worker. Install its optional dependencies, set a private `FINSERVE_VISUAL_WORKER_KEY`, then start it:

```sh
uv sync --locked --extra multimodal --extra visual-worker
uv run --no-sync python -m finserve.multimodal.visual_rpc --address 127.0.0.1:50061
```

The gateway environment needs `FINSERVE_API_KEY`, `FINSERVE_VISUAL_GRPC_TARGET=127.0.0.1:50061`, `FINSERVE_VISUAL_SERVICE_KEY` equal to the worker key, and `FINSERVE_VISUAL_DB=/absolute/private/jobs.db`. The gateway needs the `visual-worker` extra for its gRPC client; it does not need to execute JAX. Start the normal gateway factory:

```sh
uv run --no-sync uvicorn finserve.gateway.app:from_env --factory --host 127.0.0.1 --port 8000
```

The worker initializes its fixed reference revision. Query authenticated `GET /v1/visual/models` for that revision and the accepted 8-by-8 shape. Submit `POST /v1/visual/jobs` with an `Idempotency-Key` of 1–128 letters, digits, underscores or hyphens and this body shape:

```json
{
  "image": "replace with an 8-row array, each row containing 8 RGB integer triples",
  "model_revision": "use the exact revision from /v1/visual/models",
  "seed": 17,
  "timeout_seconds": 30
}
```

The illustrative strings must be replaced with typed values. RGB channels are integers from 0 through 255. A successful submission returns 202 and a job ID. Poll `GET /v1/visual/jobs/{job_id}`; retrieve `GET /v1/visual/jobs/{job_id}/artifact` only after success. `DELETE /v1/visual/jobs/{job_id}` records explicit cancellation. All these calls use the gateway bearer credential. Reusing an idempotency key with changed input returns 409.

SQLite WAL/full synchronization retains job truth and verified PNG bytes. The gRPC attempt carries worker instance/generation fences so late results or cancellation cannot complete a replacement attempt. Polling does not own the generation task. Store the database on a writable persistent volume; the configured service key defines the current tenant boundary. Plaintext gRPC is for the private local connection; external service ingress requires transport protection.

## Pretrained image understanding

Install the gateway's `vision` extra. Set `FINSERVE_VISION_ENGINE_URL` to the separately running engine's private `/v1` URL, `FINSERVE_VISION_MODEL` to its served model ID, and `FINSERVE_VISION_ENGINE_KEY` when the engine requires a credential. `FINSERVE_API_KEY` is mandatory for this capability; `FINSERVE_VISION_CAPACITY` defaults to one.

The image route accepts a bounded PNG rather than an arbitrary remote URL. Consult the gateway's generated `/docs` and `contracts/vision.py` for the exact request fields. Image normalization validates checksums, format and resolution before Pillow decoding, then removes metadata and emits canonical RGB. The measured HTTP preprocessing split moves this CPU stage only; vision encoding and decoding remain inside vLLM.

The retained uniform-color suite failed semantic correctness despite exact local/HTTP parity. Three separate bar-chart probes passed as functional examples. [Recorded results](../../../docs/results.md) explains those limits and identifies the evidence.
