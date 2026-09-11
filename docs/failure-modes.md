# Failure behavior

| Failure | Implemented behavior | Recovery boundary |
| --- | --- | --- |
| Client disconnect or ASGI send failure | Close generation and retain admission until owned work drains | A native kernel cannot be interrupted by cancelling its Python task |
| Partial engine stream or missing final usage | Serving adapter rejects terminal success; visible text is not replayed | A later independent request can use an eligible backend |
| Admission saturation | Bounded overload rejection; no hidden unbounded queue | Change declared capacity or provision measured additional engine capacity |
| Stale/unhealthy backend observation | Exclude that endpoint from new routing decisions | Actual model discovery and fresh observations restore eligibility |
| Ambiguous Ray retirement | Keep the lease reserved rather than double-book work | Operator/process recovery is distinct from a successful cancellation acknowledgement |
| GPU OOM | Expose engine failure; no blind visible-stream retry | Revisit context, active sequences, memory budget or hardware in a new profile |
| Redis quota unavailable | Reject configured quota-dependent inference with 503 | Restore Redis; optional cache read failures separately become misses |
| Registry database unavailable | Block affected reads or lifecycle writes | Already-running engines do not query the registry per token |
| Artifact checksum/identity mismatch | Reject load/import/promotion | Restore the correct immutable artifact; do not relabel its digest |
| Failed quality or performance gate | Persist rejection and do not activate candidate | Evaluate a separately declared change against unchanged acceptance inputs |
| Warm-route regression | Reconcile expected revision/generation and probe actual restored traffic | The local drill excludes cold pulls, weight loads and node recovery |
| Visual worker restart or late result | Worker-instance and attempt-generation fences prevent stale completion | Accepted durable intent remains visible for recovery |
| Duplicate pipeline task | Identical immutable identity succeeds; conflicting reuse fails | A simulated callback test is distinct from actual deployment execution |
| Telemetry export failure | Serving continues; bounded exporter queue can drop observations | Restore collector; incomplete telemetry does not become complete evidence |

Transient retries belong to bounded control operations with explicit idempotency. Invalid input, authentication failure, checksum mismatch and deterministic quality regression are not transient success opportunities. Local integration tests cover HTTP/Ray cancellation, quota failure, stream corruption, artifact tampering and warm rollback. Cloud node loss and cold recovery require separate deployment experiments.
