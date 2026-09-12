# Engine and routing adapters

`base.py` defines token envelopes. The fixture checks HTTP/accounting behavior; `pytorch_reference.py` implements causal decode with cached and full-prefix paths. Neither establishes pretrained finance-model accuracy. The OpenAI-compatible adapters validate bounded SSE, visible content, finish reason, authoritative usage and completion framing. Image/text has a separate capability and adapter.

`ray_http.py` carries the internal NDJSON protocol. `ray_backends.py` binds one named proxy actor per distinct native endpoint, manages routing reservations and polls actual engine gauges. `backend_observations.py` distinguishes native KV occupancy from physical GPU memory. Production credential checks run before clients or actors are allocated.

Native vLLM/SGLang processes own model execution and continuous batching. Multiple CPU proxy replicas do not create GPU capacity. See [inference mechanics](../../../docs/inference-fundamentals.md), [scheduling](../../../docs/scheduling-and-batching.md), [observability](../../../docs/observability.md) and [Guide](../../../docs/quality-gates.md).
