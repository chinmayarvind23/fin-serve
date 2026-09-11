# Runbook: Admission rejection and unavailable capacity

Distinguish failed scrapes, gateway rejection and router saturation. `up=0` means Prometheus could not scrape a target; it does not identify an engine fault. `finserve_active` counts gateway owners, including work whose cleanup is still running. `overloaded` counts gateway admission rejection, while `rate_limited` reports the configured quota.

Inspect the gateway outcome rates and active owners, then the private router's health, active reservations and native running/waiting/KV observations. A proxy count is not physical GPU utilization. Conservative native observations can temporarily remain full after proxy work completes; retain their age and provenance when diagnosing rejection.

Keep admission and request deadlines bounded. Increasing a queue can increase waiting time without adding execution capacity. A deployment may lower admitted concurrency, add a distinct available engine or change its serving envelope, but benchmark changes require a new declared comparison. Never hide rejected requests by reporting successful-request latency alone.

After a change, verify sustained real inference, latency and failures at the intended offered load. Check that cancelled work drains and active ownership returns to zero. A routing process restart or extra CPU proxy does not establish extra GPU capacity. See [scheduling](../scheduling-and-batching.md) and [observability](../observability.md).
