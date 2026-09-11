# Bun HTTP edge

`bun install --frozen-lockfile` installs the pinned development tools. Set
`FINSERVE_EDGE_KEY` and `FINSERVE_API_KEY` to separate credentials of at least
16 characters, then `bun run start:edge`. `FINSERVE_UPSTREAM` defaults to
`http://127.0.0.1:8000`; the edge binds to `127.0.0.1:8040` by default.

The edge authenticates callers and forwards only text/image inference and visual-job routes
to the fixed FastAPI origin. It replaces upstream credentials, rejects redirects,
caps actual upload bytes and response bytes, bounds active streams, and propagates
socket cancellation. SSE bytes and backpressure pass through without event parsing.
An overall five-minute deadline includes upload and delivery, including responses
the client stops reading. GPU admission and Redis quota accounting belong to Python.

`bun test apps/api`, `bun run typecheck`, `bun run lint` and `bun run build` verify
the boundary. Live loopback tests cover cancellation, deadlines, oversized bodies,
unread responses, authentication, route allowlisting and capacity recovery.

`/healthz` reports only edge liveness. Admin promotion/rollback routes are not exposed
through this inference credential. The visual-job routes use the separate durable
coordinator when configured; their allowlisting alone is not a worker.

The text and visual-job request limit is128KiB. `/v1/vision/completions` has a separate
1,450,000-byte envelope for one base64 PNG; Python restricts the decoded image to1MiB
and512pixels per side. Set `FINSERVE_VISION_ENGINE_URL` on the Python gateway to enable
this image-capable backend, optionally `FINSERVE_VISION_MODEL`,
`FINSERVE_VISION_ENGINE_KEY` and `FINSERVE_VISION_CAPACITY` (default1). The gateway must
have an API key and the `vision` extra installed. Its image admission occurs before
body buffering and remains owned through decoding, generation and cancellation.

The website runs a separate `surface=evidence` proxy allowing only POST `/graphql`
with a16KiB request and2MiB response limit. See [the explorer setup](../web/README.md).
