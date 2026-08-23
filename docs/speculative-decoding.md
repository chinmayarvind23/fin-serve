# Speculative Decoding

## Goal

Use a faster proposal mechanism to suggest several future tokens, then verify them with the main model more efficiently than one-at-a-time main-model decoding.

## Metrics

If `D` proposal tokens are generated, `A` accepted, and `V` verification steps occur:

`draft_acceptance_rate = A / D`

`mean_acceptance_length = 1 + A / V`

## Speed condition

Speculation helps when proposal cost plus verifier cost per unit of progress is lower than baseline main-model decode cost for the same progress.

## Why it can hurt

- proposer too slow,
- low acceptance,
- high-QPS regime where proposal work competes with main work,
- incompatible tokenization/model pairing,
- memory overhead,
- workload dominated by prefill.

## Benchmark matrix

Compare no speculation and supported speculation methods across low/medium/high concurrency, short/long outputs, repeated/diverse prefixes. Track TTFT, ITL, E2E, request/token throughput, proposal/accepted tokens, mean accepted length, GPU memory/utilization, and quality.

## Adaptive proposal length

A later experiment can shorten/disable speculation after low acceptance and allow longer proposals after sustained high acceptance. Tune on a development workload and evaluate on a separate workload.

## Visual-token research

Do not assume diffusion can use text-style token verification. Explore proposal/verification only for an autoregressive discrete visual-token formulation with a defensible correctness rule. A negative result is valid evidence.
