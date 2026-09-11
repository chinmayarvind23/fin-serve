# Boundary verification

Contract tests verify typed requests, authentication, bounded HTTP streams, the internal Ray wire format, durable visual-job interfaces and infrastructure configuration. Helm tests render the actual charts and validate RayService against the pinned operator's real CRD; they never contact a cluster.

Run `uv run --no-sync pytest tests/contract`. Infrastructure checks also need Helm, PyYAML, jsonschema and downloaded operator dependencies; [Kubernetes setup](../../infra/kubernetes/README.md) gives the bootstrap commands. CI has a separate infrastructure job so missing local tools cannot silently skip that tier there.
