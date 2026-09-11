# HTTP serving boundary

`app.py` builds application-scoped serving resources and OpenAI-style text/chat routes. Admission, request deadlines, quota checks and response cleanup own the stream from headers through disconnect. Closing the response releases its lease even when generation never began. `body_limit.py` counts received bytes; `vision.py` authenticates and reserves its separate image capacity before receiving or decoding an image.

`from_env()` selects an explicit fixture, reference or external engine. Production deployments set `FINSERVE_REQUIRE_AUTH=1` to validate ingress and selected backend credentials before allocating clients. Engine batching and KV-cache capacity remain outside this package. See [API contracts](../../../docs/api-contracts.md), [low-level ownership](../../../docs/LLD.md) and [run commands](../../../docs/commands.md).
