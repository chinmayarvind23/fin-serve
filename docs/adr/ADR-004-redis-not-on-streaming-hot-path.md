# ADR 004: Restrict Redis to quota and ephemeral cache state

## Decision

Keep synchronous text streaming under gateway/Ray/engine admission. Use Redis for optional shared quotas and bounded cache primitives. Durable visual-job truth resides in SQLite, with generation fencing and explicit cancellation.

## Evidence and consequences

The quota path runs an atomic Redis operation before admission. When configured quota storage is unavailable, the gateway returns 503 instead of silently bypassing the limit. The HMAC key scheme avoids embedding raw principals in Redis keys; replicas share a quota only with matching namespace, secret and policy.

Cache read failure is a miss. The presence of cache primitives does not establish an enabled response-cache or prefix-affinity optimization in the recorded runs. Redis has no token-stream queue, release authority or durable visual-job database role.

Real local Redis integration verifies quotas and failure handling. The bounded local container uses noeviction and ephemeral storage; cloud persistence and secret integration are separate deployment work. See [security](../security.md) and the [container guide](../../infra/docker/README.md).
