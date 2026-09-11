# ADR007: durable metadata, immutable artifacts and an optional MLflow mirror

Status: implemented locally; managed cloud deployment pending.

The SQLAlchemy registry owns immutable evidence identities and lifecycle events. SQLite is the tested backend; PostgreSQL/RDS is the deployment target. Local SHA256-addressed artifacts are verified on read. The S3 implementation has boto3 SDK request tests, with no live-cloud execution claim.

MLflow mirrors metrics and decisions after checking their run binding; it cannot approve deployment. Redis remains ephemeral. Rollback controller state and the warm route use separate SQLite stores because approval, external activation and recovery are different observations.

This split permits evidence replay without trusting mutable experiment labels. It requires explicit reconciliation across stores and a trusted deployment integration. Tests cover immutable conflicts, raw artifact tampering, retry state, decision/run binding and local MLflow. [Data and registry](../data-and-registry.md) lists the actual records and limits.
