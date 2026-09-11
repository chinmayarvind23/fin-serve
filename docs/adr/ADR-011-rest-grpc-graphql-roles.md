# ADR 011: Give each transport one role

## Decision

Use OpenAI-style HTTP/SSE for public text inference, bounded HTTP for image understanding, gRPC for the internal durable visual-worker boundary, and GraphQL for read-only evidence exploration.

## Implementation and tradeoffs

Bun forwards the public inference routes to FastAPI with bounded upload, streaming and timeout behavior. The private Ray bridge uses NDJSON envelopes to reach actor-owned generation. It is not a second public API. The JAX/Flax worker uses a typed protobuf contract because jobs need explicit identity, cancellation and artifact acknowledgement across the process boundary.

GraphQL reads run comparisons and verified artifact annotations through a separate authenticated service. Query depth, resolver output, byte budgets and concurrent disk work are bounded. The explorer does not schedule inference or authorize promotion.

Real HTTP/gRPC/Ray integration and Bun HTTP tests verify these boundaries. DOM-emulated frontend checks do not establish browser rendering; deployed transport and browser checks require separate evidence. No general embedding endpoint or public deployment-mutation API is claimed.
