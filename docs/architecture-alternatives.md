# Architecture alternatives

| Decision | Selected approach | Tradeoff and evidence |
| --- | --- | --- |
| Implement a full inference engine | Small PyTorch reference for decode/KV mechanics; external vLLM/SGLang for pretrained serving | The reference exposes numerical and cancellation invariants but uses random weights. Mature engines own batching and device execution. |
| Couple the API to engine packages | Bounded HTTP adapters with separately managed runtimes | Adds a transport hop and cancellation boundary, but permits incompatible engine environments and explicit GPU ownership. |
| Queue synchronous text in Redis | Direct streaming with admission; Redis only for optional quota/cache | Avoids a second token-work queue. Redis quota failure still rejects requests when that quota is configured. |
| Treat Ray proxy count as model capacity | One named proxy per distinct engine endpoint | Routing leases are inspectable; actual model processes and physical GPUs must be counted separately. |
| One universal multimodal worker | Distinct text, pretrained image-understanding and durable JAX reference paths | Different contracts and resource bounds add integration work but avoid silently sending unsupported media to a text backend. |
| Fully split vision encoding and decoding | Measured CPU normalization split only | Actual HTTP transfer was compared with local normalization. Encoder/decode separation has not been demonstrated. |
| Put GraphQL in inference scheduling | Separate bounded read-only evidence service | Nested run/quality/history views are useful without adding registry reads to every token. |
| Use MLflow labels as release authority | Immutable SQL/CAS records and a recomputed gate | MLflow is a reporting mirror; cross-store reconciliation becomes explicit. |
| Put Airflow in online requests | Offline evidence and lifecycle tasks | Request latency does not depend on the orchestration scheduler. Full producer integration remains separate work. |
| Use a single-GPU update as a warm canary | Explicit disruptive staging update; separate warm-route controller | Local warm switching is tested, but overlapping cloud versions need additional capacity and deployment evidence. |

These choices are implemented at the scopes described in the [HLD](HLD.md), [LLD](LLD.md) and [results](results.md). Additional databases, queues or engine backends should be introduced only for a measured requirement; their presence alone would not improve the current evidence.
