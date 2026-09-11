# Performance verification scope

GPU performance runs use the benchmark clients and explicit frozen configurations, not an automatically executed test suite in this directory. Deterministic metric/gate regressions run in unit and integration tests; the configured self-hosted workflow recomputes registered evidence without inventing a new GPU run.

Follow [measurement commands](../../docs/commands.md) and [performance methods](../../docs/performance.md). Keep raw requests, warmups, failed runs, GPU observations and provenance in a fresh external evidence directory. A passing CPU test cannot establish throughput, model quality or a cloud cost reduction.
