# Frozen workloads and declared engine profiles

`text-release-v1.json` contains the 64 synthetic cases used by the recorded sustained comparison. The run configuration separately fixes repetition count, warmup, concurrency, timing mode and token budgets. Reusing this workload hash does not make a short development run equivalent to a 3,072-request sustained run.

The vLLM/SGLang JSON files declare local runtime candidates. A filename or declaration does not prove that its engine launched or passed quality; retained process/readiness records and [results](../../docs/results.md) establish execution. The finance YAML describes workload families and intended evaluation dimensions, rather than the actual frozen release population. See [benchmark populations](../../docs/finance-benchmark-suite.md).
