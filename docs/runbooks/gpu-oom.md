# Runbook: GPU OOM

Detect CUDA/engine memory failures. Stop blind retries, classify request, inspect prompt/output limits, active sequences, KV configuration, model revision, routing change, multimodal shape. Recover by restart, lower concurrency/token budget, reject oversized inputs, revert config, or move to larger-memory pool. Preserve as eval evidence.
