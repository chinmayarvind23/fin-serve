# Benchmark methodology

FinServe keeps tuning cohorts, frozen comparisons and functional probes separate. Freeze workload bytes, grader, reference answers and serving configuration before a comparison. Preserve every attempted run, including startup failures, timeouts and rejected optimizations. A faster candidate is accepted only when the declared performance and quality gates both pass.

## Populations and clocks

A logical inference request counts once; SSE chunks do not count as requests or generated tokens. The engine's final usage is authoritative. The serving adapters reject missing usage or malformed termination. The independent load client can record transport completion with unknown usage; that request cannot support a known token-throughput result. Incomplete termination and malformed output remain failures.

Warmup records are retained separately. The measured interval begins before measured requests are scheduled and ends after their completion or failure. Successful-request throughput is successful measured requests divided by that whole interval. Generated-token throughput sums authoritative output tokens from successful measured requests over the same interval. Failed requests remain in raw records and the attempted-request denominator for success rate.

Client TTFT measures client send to first nonempty visible content. Server TTFT measures server receipt to first visible content on the server clock. They are reported separately; no timestamp subtraction crosses hosts. End-to-end p95 uses successful client send-to-complete durations. In open-loop mode, scheduled-to-complete delay additionally captures waiting before client send. Closed-loop mode fixes concurrency; its configured arrival-rate field is unused.

## Reproducible comparison

Both configurations must use the same frozen workload hash, measured/warmup counts, concurrency, arrival mode/rate, timeout, hardware and model/tokenizer revisions. Record source revision, engine version and arguments, cache/speculation settings and actual image/configuration identities. A native process can declare its image unknown. A later build cannot supply that identity retroactively.

When the gateway supplies routing headers, each new request record retains an optional `routing` object. Its `physical` binding identifies the selected revision/digest and route generation; `anchor` and `pool_generation` separately identify the approved primary and capacity membership. A rejected request may carry only an anchor, which makes no claim that inference ran. Valid attribution remains recorded if the HTTP request or stream later fails. Duplicate, partial or malformed identity headers make the offered request fail validation. Header claims must still be checked against immutable runtime receipts before treating them as process attestation. Older records without attribution retain their original serialized fields and cannot establish which replica served them.

Change one serving mechanism at a time. Use a separate workload for tuning. Repeated randomized ordering is preferable for a performance claim; label an ordered single pair and its workstation/thermal confounds when that is the evidence available. The [recorded results](results.md) include one such eager/compiled pair with 6,144 measured requests. They do not establish a production availability guarantee.

The committed `benchmarks/configs/text-release-v1.json` contains 64 cases across SEC question answering, tables, earnings and general text, with varied context and output budgets. Repetition supplies the declared sustained request count. It is an explicit synthetic financial workload, not captured customer traffic.

## Quality, GPU and cost

New managed performance receipts bind collection to a single process's monotonic clock. Bounds are captured after the opening runtime probe and before the closing probe; the collector's clock domain and measured interval must fit exactly inside them. Wall-clock corrections cannot reverse this causal ordering. Historical receipts retain their original strict wall-clock checks. GPU telemetry still uses the recorded wall/monotonic mapping and reports clock drift rather than treating the new collection proof as reliable GPU utilization.

The separate `evals/golden/correctness-32-v1.json` suite has 32 exact/typed-JSON cases. Correctness compares candidate output with expected answers; parity compares it with the frozen reference output. Equal wrong answers can have high parity. Invalid JSON and exact-format mismatches remain failures. The grader is versioned and its source identity is checked during evidence import.

GPU utilization integrates timestamped physical-device samples across the measured epoch interval. Stale gaps are capped and coverage is published. Below 95% coverage, mean utilization is unknown. Shared-GPU processes do not create extra devices. Per-engine running/waiting counts and KV-cache occupancy are different measurements.

Cost per million generated tokens requires explicit price and billed or modelled time:

`cost_per_million = total_declared_cost * 1_000_000 / authoritative_generated_tokens`.

Report billing scope, idle time, other infrastructure and quality equivalence. Workstation throughput alone cannot establish cloud cost savings.

## Actual artifact layout

Evidence is written to a new external directory, such as `resources/fin_serve/evidence/<experiment>/`:

```text
experiment/
  environment.json
  experiment-status.json
  gpu.jsonl
  gpu-summary.json
  run/
    manifest.json
    requests.jsonl
    summary.json
quality/
  manifest.json
  requests.jsonl
  answers.json
  quality.json
```

Quality filenames are defined by `scripts/run_quality.py`; retain its complete output directory. Additional startup logs, package inventories, frozen profiles and independent audit reports accompany the relevant experiment. [Commands](commands.md) covers the actual CLI. The evidence explorer verifies and imports these artifacts; its UI never changes their recorded measurements or authorizes deployment.
