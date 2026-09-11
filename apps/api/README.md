# Bun HTTP edge

`bun install --frozen-lockfile` installs the pinned development tools. Set
`FINSERVE_EDGE_KEY` and `FINSERVE_API_KEY` to separate credentials of at least
16 characters, then `bun run start:edge`. `FINSERVE_UPSTREAM` defaults to
`http://127.0.0.1:8000`; the edge binds to `127.0.0.1:8040` by default.

The edge authenticates callers and forwards only inference and visual-job routes
to the fixed FastAPI origin. It replaces upstream credentials, rejects redirects,
caps actual upload bytes and response bytes, bounds active streams, and propagates
socket cancellation. SSE bytes and backpressure pass through without event parsing.
An overall five-minute deadline includes upload and delivery, including responses
the client stops reading. GPU admission and Redis quota accounting belong to Python.

`bun test apps/api`, `bun run typecheck`, `bun run lint` and `bun run build` verify
the boundary. Live loopback tests cover cancellation, deadlines, oversized bodies,
unread responses, authentication, route allowlisting and capacity recovery.

`/healthz` reports only edge liveness. Admin promotion/rollback routes are not exposed
through this inference credential. The visual-job routes are reserved for the
separate durable coordinator integration; their allowlisting alone is not a worker.
