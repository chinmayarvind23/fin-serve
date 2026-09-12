# Promotion and warm rollback

`promotion.py` evaluates performance and quality thresholds. `rollback.py` records revision-aware rollback state and reconciliation. `warm_routes.py` stores active endpoint truth separately from lifecycle metadata, using an expected revision/generation and stable action identity for compare-and-swap cutover.

Requests pin their original route. Recovery requires actual healthy inference on the expected revision after switching; a command acknowledgment alone is insufficient. The recorded local drill switches already-running fixture endpoints and excludes cold model or cloud-node startup. See [deployment](../../../docs/deployment.md), [rollback operations](../../../docs/runbooks/rollback.md) and [Guide](../../../docs/quality-gates.md).
