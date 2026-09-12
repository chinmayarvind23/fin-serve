# Speculative decoding

Speculation is an engine configuration option evaluated through the shared release workflow.

## Mechanism and accounting

A proposer supplies future tokens for the main model to verify. Benefit depends on accepted progress relative to proposal and verification cost. FinServe delegates this algorithm to the engine; it implements neither a draft model nor an adaptive proposal-length controller. Supported methods depend on engine version; see the [vLLM documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

For draft tokens `D`, accepted draft tokens `A` and verification steps `V`, acceptance rate is `A / D`. Where the engine guarantees one additional target token per verification, accepted progress is `1 + A / V`. Missing counters and zero denominators remain missing. Counter deltas must use the same process and interval; an interval containing warmup must say so.

Acceptance is diagnostic. Proposal work can compete with useful target work at high concurrency. TTFT, generated length, successful-request throughput, generated-token throughput, end-to-end tails and task quality determine whether to retain a profile.
