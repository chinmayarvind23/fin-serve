# Scheduling and batching

FinServe selects an engine and owns request admission. The external engine owns token scheduling, continuous batching and KV allocation. Creating more Ray HTTP proxy actors does not create GPU capacity.

## Admission and ownership

[ReplicaRouter](../src/finserve/scheduler/router.py) serializes snapshot updates, selection and reservation under one lock. Each reservation has a unique lease. Worker snapshots identify reflected leases; the router adds only outstanding leases missing from that observation. Release requires the exact lease and rejects duplicate cleanup.

[RoutedBackends](../src/finserve/engines/ray_backends.py) addresses one named Ray proxy per trusted, distinct HTTP backend. Worker admission precedes generation. Cancellation retires that admission and drains the proxy stream before releasing the routing lease. Closing HTTP proves proxy cleanup; it does not prove a remote GPU kernel has stopped.

The pure scheduler rejects full capacity immediately. The production bridge waits for otherwise eligible, temporarily saturated replicas within the original request deadline. Its admission path has a 128-request cap and fixed status counters for pending admissions, waited requests and accumulated admission time. Each attempt refreshes observations before reserving. No worker generation starts during this wait. Unhealthy, stale, unknown-memory and mismatched-model replicas remain ineligible. Waiting does not guarantee fairness.

## Policy and observations

[policy.py](../src/finserve/scheduler/policy.py) checks health, exact model, positive capacity, observation freshness, requested GPU compatibility and memory limits before ranking. The baseline ranks normalized effective load. The adaptive score adds physical GPU memory pressure and subtracts a bounded prefix-affinity preference. Affinity cannot bypass capacity or attract work beyond the permitted load gap. Stable replica IDs break ties.

The production bridge currently supplies no prefix-cache ownership information, so cache affinity is inactive. There is no request-length or SLO scheduling branch. The engine receives each request's generation budget.

[backend_observations.py](../src/finserve/engines/backend_observations.py) parses bounded vLLM gauges for running requests, waiting requests and KV utilization. Worker load is the larger of current proxy occupancy and the cached native count. Physical VRAM comes from one shared-device sampler; engine KV and physical GPU memory remain separate. Cached engine age and physical sample age are independent of router receipt time. Missing observations fail closed.

The proxy capacity guarantee assumes exclusive routed traffic. Bypass clients do not share its atomic lease fence; native engine limits remain authoritative. Multiple independent routers must partition worker budgets or share one owner.

## Local two-engine evidence

Two vLLM 0.29.0 processes loaded the same pinned Qwen2.5-0.5B model on one RTX 4070 Laptop GPU with memory fraction 0.35 each, four sequences per engine, context 1,024 and prefix caching disabled. Both became ready. The shared device used 5,952 of 8,188 MiB at the idle checkpoint.

The first frozen comparison offered 64 requests per cohort at concurrency eight in least-load/adaptive/adaptive/least-load order. Success counts were 21, 19, 24 and 24. All 168 failures surfaced as engine unavailable; retained Ray logs identify capacity rejection before content. Successful-request throughput ranged from 1.69 to 2.38 requests/s. This failed availability envelope provides no adaptive routing gain. Shared VRAM and inactive affinity also make the policies order-equivalent for identical snapshots.

Retained session `two-engine-routing-session-01` predates bounded admission waiting. Conservative cached occupancy can remain full after proxy completion; CPU tests cover this mechanism without discarding the native gauge.

Scale-down quarantined the second backend, drained it and stopped its owned process; a survivor request completed. Restart failed in the local port preflight before generation two launched. Session 01 remains failed. A Linux regression reproduces a retained TCP TIME_WAIT bind failure; the corrected preflight permits that state while rejecting live listeners.

Session `two-engine-routing-session-02` ran the same frozen workload and engine profile from a clean, read-only checkout of `51c89ea`. All four cohorts completed 64/64 requests after the admission correction. Each generated 4,608 tokens. These are short local load intervals, separate from the sustained benchmark.

| Order / policy | Seconds | Requests/s | Tokens/s | Median client TTFT | p95 E2E | Mean physical GPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / least-load | 34.88 | 1.835 | 132.09 | 355 ms | 7.58 s | 52.54% |
| 2 / adaptive | 36.68 | 1.745 | 125.61 | 432 ms | 7.42 s | 52.19% |
| 3 / adaptive | 34.93 | 1.832 | 131.91 | 385 ms | 8.98 s | 50.49% |
| 4 / least-load | 32.59 | 1.964 | 141.39 | 354 ms | 6.83 s | 56.38% |

The results establish successful bounded admission in this population, with no adaptive throughput gain. All GPU means count the one physical device once and have full coverage under the bounded sample-hold method. The declared TTFT ≤1 s and E2E ≤15 s checks passed for 236/256 requests. Exact output agreement was 62/64 and 63/64 across the two policy pairs, or 97.66% overall. The mechanical load prompts provide no semantic correctness estimate.

The second session also completed scale-down, a survivor request, restart, forced failure after partial output, another survivor request and replacement-backend generation. The failed stream surfaced an error; it was not retried after visibility. Final router leases and pending admissions were zero. All owned engine processes stopped. The session manifest is complete, but the PowerShell launch wrapper reported exit 1 amid Ray shutdown warnings; a separate native driver exit code was not captured. These records establish local process lifecycle behavior, not cloud autoscaling or a clean driver-exit claim.

[audit_routing_session.py](../scripts/audit_routing_session.py) recomputes both sessions without rewriting input evidence. It checks the complete artifact/source set, request-to-decision identity and policy, shared GPU UUID, equal controls and nonoverlapping intervals before comparing outputs.

## Scaling boundary

The current layout uses public handles and one named deployment per endpoint. Ray also exposes a [custom request-router API](https://docs.ray.io/en/latest/serve/advanced-guides/custom-request-router.html); this implementation does not replace Ray's internal replica scheduler. Its single authority favors inspectable ownership over an unmeasured distributed control plane.

The intended infrastructure chain is Serve demand, Ray resources, pods and GPU nodes. Repository configuration alone does not verify this chain in a deployed cloud. Process readiness, image pull, weight loading, compilation and node provisioning require separate timings and failure records.
