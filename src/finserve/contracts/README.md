# Typed boundaries and immutable identities

Contracts bound text/chat and image requests, model files, serving profiles, deployment revisions, routing state and producer/runtime attempts. Strict numeric and size limits reject invalid inputs before resource acquisition. Immutable profiles canonicalize engine parameters and bind model/tokenizer manifests, endpoint and credential-variable name; actual image identity stays in the revision.

The local managed-runtime specification additionally binds the verified image, absolute model directory, loopback endpoint and resource budgets. A typed receipt is evidence of an observed operation, not permission to skip rechecking its artifacts. [API documentation](../../../docs/api-contracts.md) and the [low-level design](../../../docs/LLD.md) explain externally visible fields and cross-module invariants.
