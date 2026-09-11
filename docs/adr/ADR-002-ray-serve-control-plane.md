# ADR 002: Use Ray for explicit routing ownership

## Decision

Place a private Ray HTTP bridge between the gateway and separately managed engine processes when distributed routing is needed. Use one named proxy per trusted distinct endpoint, with bounded admission, leased reservations and explicit cancellation retirement.

## Evidence and limits

Actual Ray actors and HTTP transport have been exercised with CPU fixtures and native GPU engines. Linked trace IDs establish gateway/route/engine causality. The first two-engine GPU experiment exposed capacity rejection while cached native occupancy remained full; both models stayed alive. The correction waits within the original deadline for otherwise eligible capacity and retains the failed experiment.

The implementation uses public Ray handles. It does not replace Ray's internal scheduler with a custom request-router plugin. GPU memory and token batching remain engine-owned. Additional CPU proxies cannot create GPU capacity; cloud placement and autoscaling require deployment evidence. See [scheduling](../scheduling-and-batching.md).
