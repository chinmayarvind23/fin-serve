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
