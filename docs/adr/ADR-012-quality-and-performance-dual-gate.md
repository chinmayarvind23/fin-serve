# ADR012: performance and quality must both pass

Status: implemented; real rejected local candidates retained.

Promotion recomputes summaries from all raw requests, verifies an unchanged workload/load envelope and checks a predeclared policy. Quality compares candidate outputs with reference outputs and frozen expected answers. A hard correctness failure cannot average away behind throughput or output agreement.

Canonical jobs bind model/tokenizer/source/image/configuration identities, profiles and endpoint. Chat evidence freezes API, system instruction and template digest and binds that mapping to quality. Template metadata is not a remote attestation; the trusted producer must verify actual tokenizer bytes. Missing profile or raw evidence fails closed at both CLI/DAG and shared lifecycle boundaries.

Tests cover faster but incorrect candidates, tampered evidence, profile/mapping mismatch, missing canonical inputs after partial registration and direct bypass attempts. The first actual image-bound chat development pair was rejected: its original three-case smoke accuracy was 1/3 and it showed no speed gain. That smoke does not replace the frozen 32-case release suite.

Rejected optimizations remain evidence. Changed prompts, models, constraints or policies require a new frozen experiment; previous failures stay available. See [benchmark methodology](../benchmark-methodology.md) and [Airflow pipeline](../airflow-pipeline.md).
