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

## Scaling boundary

The current layout uses public handles and one named deployment per endpoint. Ray also exposes a [custom request-router API](https://docs.ray.io/en/latest/serve/advanced-guides/custom-request-router.html); this implementation does not replace Ray's internal replica scheduler. Its single authority favors inspectable ownership over an unmeasured distributed control plane.

The intended infrastructure chain is Serve demand, Ray resources, pods and GPU nodes. Repository configuration alone does not verify this chain in a deployed cloud. Process readiness, image pull, weight loading, compilation and node provisioning require separate timings and failure records.
