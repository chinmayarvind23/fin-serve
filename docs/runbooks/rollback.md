# Runbook: Rollback

Use the trusted lifecycle controller and configured deployment adapter. There is no public HTTP rollback endpoint. The controller must already know the candidate, known-good revision, deployment generation and immutable evidence identities.

1. Retain the regression signal and its detection time. Stop further promotion for that deployment.
2. Read the durable deployment, activation and rollback records. Check the currently observed route generation and revision; do not infer active traffic from the desired configuration.
3. Apply the known-good revision with the controller's expected revision/generation and idempotency key. A stale generation is a conflict, not permission to overwrite a newer route.
4. If an external action was interrupted or unacknowledged, reconcile its actual effect. Do not create a second action merely because the first caller timed out. The durable `needs_reconciliation` state preserves this ambiguity.
5. Verify inference through the active traffic URL, including the exact revision digest and route generation. Process liveness alone is insufficient. Only verified restoration completes the rollback record.
6. Retain failed requests, action receipts, health observations and both revision identities. Report detection-to-verified-health duration, including controller and action delay.

The tested `WarmRouteAdapter` switches between already-running endpoints. Existing streams retain their original route. Its local 0.680-second fixture drill excludes image pull, model loading and GPU/node restart; it does not establish a cloud rollback target. The single-GPU staging engine uses a disruptive update and needs separate recovery procedures and evidence.

See [low-level design](../LLD.md), [lifecycle decision](../adr/ADR-014-rollback-known-good-revision.md) and [results](../results.md).

## Automatic local health monitor

The bounded monitor observes actual one-token streamed health requests through
the warm traffic endpoint. A frozen policy names the candidate revision digest,
deployment generation, maximum probe count and consecutive failed/slow threshold.
Slow means complete synthetic health-probe duration, not TTFT. These probes do not
estimate user-request availability or model correctness and cannot approve quality.

Create a `MonitorPolicy` JSON file outside the checkout using the actual activated
candidate identity. Set `FINSERVE_API_KEY` through the runtime environment, then run:

```sh
uv run --no-sync python -m finserve.reliability.monitor_cli \
  --policy "$FINSERVE_MONITOR_POLICY" \
  --control "$FINSERVE_CONTROL_DB" --routes "$FINSERVE_ROUTES_DB" \
  --journal "$FINSERVE_MONITOR_DB" --artifacts "$FINSERVE_MONITOR_ARTIFACTS" \
  --traffic-url "$FINSERVE_TRAFFIC_URL"
```

This command requires the registry dependencies and existing local control/route
stores. It provisions no engine. Every completed probe has an immutable CAS receipt;
replay reconstructs the same signal. An unresolved interrupted probe cannot start
again without explicit reconciliation. Changed policy, endpoint or store identities
cannot reuse the same monitor ID. A stale generation ends observation without applying
rollback, and a healthy window leaves the prior known-good revision unchanged.

On detection the monitor invokes the existing fenced rollback controller, with
`recovery_attempts` bounded to 1–10 (default 3). Each controller cycle has the policy's
probe timeout; native cleanup can outlive a deadline while retaining ownership.
`finished=true` means the probe window ended. Require `rollback.status=restored` to
claim recovery. The CLI prints the retained result and exits 2 if recovery remains
unverified; a later invocation reuses the same signal and reconciliation state.
