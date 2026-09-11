"""Atomic reservation ownership for a bounded, single authoritative routing process."""

import math
import threading
import time
from dataclasses import dataclass
from uuid import uuid4

from finserve.contracts.routing import ReplicaSnapshot, RoutingDecision, RoutingRequest
from finserve.scheduler.policy import Candidate, RoutingPolicy, is_eligible, score_candidate


class NoReplicaAvailable(RuntimeError):
    """Reject admission immediately rather than accumulating an unbounded waiting queue."""


@dataclass(frozen=True)
class RoutingLease:
    """A unique reservation remains owned until completion or acknowledged cancellation."""

    lease_id: str
    decision: RoutingDecision


class ReplicaRouter:
    """One lock serializes telemetry reconciliation, selection, and reservation changes.

    Multiple gateway processes must share one routing actor or partition worker budgets;
    independent routers observing the same capacity cannot provide a global guarantee.
    """

    def __init__(self, policy: RoutingPolicy | None = None, max_replicas: int = 256) -> None:
        """Bound retained replica metadata while keeping the pure policy separately testable."""
        if max_replicas < 1:
            raise ValueError("max_replicas must be positive")
        self.policy = policy or RoutingPolicy()
        self.max_replicas = max_replicas
        self._snapshots: dict[str, ReplicaSnapshot] = {}
        self._leases: dict[str, RoutingLease] = {}
        self._replica_leases: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._owner = uuid4().hex
        self._sequence = 0

    def update_snapshot(self, snapshot: ReplicaSnapshot) -> None:
        """Reject out-of-order telemetry so an older low-load view cannot overwrite newer work."""
        with self._lock:
            previous = self._snapshots.get(snapshot.replica_id)
            if previous is not None and snapshot.received_at < previous.received_at:
                raise ValueError("snapshot arrival timestamp moved backward")
            if previous is None and len(self._snapshots) >= self.max_replicas:
                raise ValueError("replica snapshot limit reached")
            self._snapshots[snapshot.replica_id] = snapshot

    def _candidate(self, snapshot: ReplicaSnapshot) -> Candidate:
        """Add leases missing from telemetry without double-counting acknowledged leases."""
        unreflected = len(
            self._replica_leases.get(snapshot.replica_id, set()) - snapshot.reflected_lease_ids
        )
        return Candidate(
            snapshot, snapshot.ongoing_requests + snapshot.queued_requests + unreflected
        )

    def reserve(self, request: RoutingRequest, now: float | None = None) -> RoutingLease:
        """Check and reserve atomically, using stable ID tie-breaking for reproducibility."""
        timestamp = time.monotonic() if now is None else now
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("router clock must be finite and nonnegative")
        with self._lock:
            candidates = [self._candidate(snapshot) for snapshot in self._snapshots.values()]
            eligible = [
                candidate
                for candidate in candidates
                if is_eligible(candidate, request, self.policy, timestamp)
            ]
            if not eligible:
                raise NoReplicaAvailable("No eligible replica has available capacity")
            minimum_load = min(candidate.load for candidate in eligible)
            decisions = [
                score_candidate(candidate, request, self.policy, minimum_load, timestamp)
                for candidate in eligible
            ]
            decision = min(decisions, key=lambda item: (item.score, item.replica_id))
            self._sequence += 1
            lease = RoutingLease(f"{self._owner}:{self._sequence}", decision)
            self._leases[lease.lease_id] = lease
            self._replica_leases.setdefault(decision.replica_id, set()).add(lease.lease_id)
            return lease

    def release(self, lease: RoutingLease) -> None:
        """Require exact ownership and reject duplicate cleanup without underflowing."""
        with self._lock:
            if self._leases.get(lease.lease_id) != lease:
                raise RuntimeError("routing lease is unknown, altered, or already released")
            del self._leases[lease.lease_id]
            self._replica_leases[lease.decision.replica_id].remove(lease.lease_id)

    @property
    def snapshots(self) -> tuple[ReplicaSnapshot, ...]:
        """Expose immutable evidence without granting access to mutable reservation state."""
        with self._lock:
            return tuple(self._snapshots.values())

    @property
    def active_reservations(self) -> int:
        """Expose outstanding ownership for shutdown and cancellation verification."""
        with self._lock:
            return len(self._leases)

    def forget_replica(self, replica_id: str) -> None:
        """Retire metadata only after active leases drain, preserving cleanup accountability."""
        with self._lock:
            if self._replica_leases.get(replica_id):
                raise RuntimeError("cannot forget a replica with active reservations")
            self._snapshots.pop(replica_id, None)
            self._replica_leases.pop(replica_id, None)
