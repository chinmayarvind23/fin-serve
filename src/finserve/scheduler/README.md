# Routing decisions and reservations

`policy.py` ranks eligible replicas using normalized effective load, optional physical memory pressure and bounded prefix affinity. `router.py` owns immutable reservation leases and rejects unhealthy, stale, wrong-model or full replicas. Its pure reserve operation does not wait; the production Ray bridge owns the bounded wait for temporarily full eligible capacity.

The bridge currently has no prefix ownership signal, so affinity is inactive. Two processes on one GPU observe the same physical memory term. [Scheduling and batching](../../../docs/scheduling-and-batching.md) explains the score, limits and ownership rules.
