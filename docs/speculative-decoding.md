# Speculative decoding

Speculation remains disabled in the preferred local profile. The measured n-gram candidate lost throughput and failed the correctness gate.

## Mechanism and accounting

A proposer supplies future tokens for the main model to verify. Benefit depends on accepted progress relative to proposal and verification cost. FinServe delegates this algorithm to the engine; it implements neither a draft model nor an adaptive proposal-length controller. Supported methods depend on engine version; see the [vLLM documentation](https://docs.vllm.ai/en/latest/features/speculative_decoding/).

For draft tokens `D`, accepted draft tokens `A` and verification steps `V`, acceptance rate is `A / D`. Where the engine guarantees one additional target token per verification, accepted progress is `1 + A / V`. Missing counters and zero denominators remain missing. Counter deltas must use the same process and interval; an interval containing warmup must say so.

Acceptance is diagnostic. Proposal work can compete with useful target work at high concurrency. TTFT, generated length, successful-request throughput, generated-token throughput, end-to-end tails and task quality determine whether to retain a profile.

## Recorded rejection

At concurrency 16, the 256-request development comparison measured 40.29 requests/s without speculation and 20.59 with three-token n-gram proposals. End-to-end p95 increased from 0.533 to 1.387 seconds. Acceptance was 30.04% across two candidate recordings including warmup. The unchanged correctness suite failed.

Failed cohorts remain in `ngram-load-c16-01` and `quality-ngram-01`; [recorded results](results.md) puts them alongside the sustained comparison. Development measurements do not establish held-out improvement, and these counters do not imply a measured-interval-only acceptance rate.

No visual-token speculation benchmark has passed. The JAX/Flax reference generates discrete palette tokens autoregressively but has no proposal/verification path. Text verification rules do not establish correctness for diffusion or other non-autoregressive generation.
