# ADR014: rollback restores an exact known-good revision

Status: durable local controller and warm-route adapter implemented; cloud rollout adapter pending.

The controller records trusted detector UTC time separately from receipt, rollback start and verified recovery. Signals identify the revision and generation observed. Generation fencing prevents a stale signal from rolling back a newer deployment. Immutable model, tokenizer, source, image and configuration identities define known-good.

SQLite persists ownership, attempts and intermediate states. External ambiguity is reconciled with health. Replay requires a declared idempotent adapter and unchanged request fingerprint. Cancellation or failed health cannot produce a restored timestamp. Detection-to-health duration exists only after health verifies the exact known-good revision.

`WarmRouteStore` represents the external route separately from `DeploymentStore`. Its adapter uses atomic generation comparison and idempotency receipts, then probes actual traffic and route identity. It switches already running backends; it does not represent image pull, pod startup or cluster rollout. The trusted integration must acknowledge candidate activation and validate probation before advancing known-good with `mark_stable`.

Tests cover stale signals, concurrent ownership, ambiguous actions, cancellation and unhealthy recovery. A real loopback HTTP drill retains actual failed requests during a controlled backend regression. It establishes local behavior, not 99.95% production availability or an AWS rollback SLA.
