"""Persist bounded synthetic health probes and automatically signal exact-generation rollback."""

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from functools import partial
from typing import Literal

from pydantic import Field, model_validator

from finserve.contracts.deployment import (
    HealthObservation,
    ImmutableModel,
    RegressionSignal,
    RollbackRecord,
)
from finserve.http_ownership import HTTPClosureError
from finserve.registry.artifacts import ArtifactRef
from finserve.registry.model_assets import owned_disk
from finserve.registry.producer_stages import ProducerStages, StageState
from finserve.registry.producer_tasks import declare_input, start_owned
from finserve.reliability.rollback import DeploymentStore, RollbackController
from finserve.reliability.warm_routes import WarmRouteAdapter


class MonitorPolicy(ImmutableModel):
    """Freeze synthetic probe thresholds separately from user-traffic SLA or release quality."""

    monitor_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    deployment_id: str = Field(min_length=1, max_length=128)
    revision_id: str = Field(min_length=1, max_length=128)
    revision_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation: int = Field(ge=0, strict=True)
    consecutive_regressions: int = Field(default=3, ge=1, le=32, strict=True)
    maximum_probes: int = Field(default=60, ge=1, le=256, strict=True)
    recovery_attempts: int = Field(default=3, ge=1, le=10, strict=True)
    interval_seconds: float = Field(default=5.0, ge=0.1, le=60)
    slow_probe_seconds: float = Field(default=2.0, gt=0, le=30)
    probe_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    @model_validator(mode="after")
    def reachable_policy(self) -> "MonitorPolicy":
        """Reject configurations whose declared threshold cannot be reached in this bounded run."""
        if (
            self.maximum_probes < self.consecutive_regressions
            or self.slow_probe_seconds > self.probe_timeout_seconds
        ):
            raise ValueError("monitor thresholds exceed collection bounds")
        return self

    def digest(self) -> str:
        """Normalize nested defaults so caller JSON round trips cannot change policy identity."""
        normalized = MonitorPolicy.model_validate_json(self.model_dump_json())
        data = json.dumps(normalized.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode()).hexdigest()


class ProbeObservation(ImmutableModel):
    """A receipt retains failed probes and the route fence surrounding their actual HTTP work."""

    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=0, le=255, strict=True)
    previous: ArtifactRef | None
    finished_at: float = Field(gt=0)
    elapsed_seconds: float = Field(ge=0)
    route_matches: bool = Field(strict=True)
    health: HealthObservation | None
    error_code: str | None = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


class MonitorResult(ImmutableModel):
    """The caller can inspect the exact evidence and signal without granting quality approval."""

    observation: ProbeObservation
    reference: ArtifactRef
    signal: RegressionSignal | None = None
    rollback: RollbackRecord | None = None
    finished: bool


class ProbeMonitor:
    """One immutable candidate window; journal CAS prevents overlapping probe attempts."""

    def __init__(
        self,
        policy: MonitorPolicy,
        journal: ProducerStages,
        control: DeploymentStore,
        adapter: WarmRouteAdapter,
        *,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Use trusted stores and real health; the caller owns authenticated client lifetime."""
        self.policy = MonitorPolicy.model_validate_json(policy.model_dump_json())
        self.journal, self.control, self.adapter = journal, control, adapter
        self.clock, self.monotonic = clock, monotonic

    def _matches(self) -> bool:
        """Both traffic truth and deployment intent must still name this exact candidate."""
        policy = self.policy
        state = self.control.deployment(policy.deployment_id)
        route = self.adapter.store.snapshot(policy.deployment_id)
        revision = self.control.revision(policy.revision_id)
        return (
            state.active_revision == route.revision_id == policy.revision_id
            and state.generation == route.generation == policy.generation
            and revision.digest() == route.revision_digest == policy.revision_digest
            and state.rollback_id is None
            and state.active_revision != state.known_good_revision
        )

    def _classification(
        self, item: ProbeObservation
    ) -> Literal["healthy", "failed", "slow", "stale"]:
        """Latency is complete synthetic health-probe duration; it is never labeled model TTFT."""
        if not item.route_matches:
            return "stale"
        health = item.health
        if (
            health is None
            or not health.ready
            or not health.smoke_passed
            or health.revision_id != self.policy.revision_id
            or health.revision_digest != self.policy.revision_digest
        ):
            return "failed"
        return "slow" if item.elapsed_seconds > self.policy.slow_probe_seconds else "healthy"

    def _result(
        self, observations: list[ProbeObservation], references: list[ArtifactRef]
    ) -> MonitorResult:
        """Recompute a deterministic signal from retained consecutive outcomes on every replay."""
        last, reference = observations[-1], references[-1]
        classifications = [self._classification(item) for item in observations]
        count = self.policy.consecutive_regressions
        recent = classifications[-count:]
        reason = classifications[-1]
        signal = None
        if len(recent) == count and all(value in {"failed", "slow"} for value in recent):
            identity = f"{self.policy.digest()}:{last.sequence}"
            signal = RegressionSignal(
                signal_id="probe-" + hashlib.sha256(identity.encode()).hexdigest(),
                deployment_id=self.policy.deployment_id,
                observed_revision=self.policy.revision_id,
                observed_generation=self.policy.generation,
                detected_at=last.finished_at,
                detector="synthetic-stream-health-v1",
                reason=f"{count} consecutive failed/slow probes; final artifact {reference.sha256}",
            )
        return MonitorResult(
            observation=last,
            reference=reference,
            signal=signal,
            finished=signal is not None
            or reason == "stale"
            or len(observations) == self.policy.maximum_probes,
        )

    def _history(self) -> tuple[list[ProbeObservation], list[ArtifactRef], StageState | None]:
        """Chain completed verified CAS rows and stop at the first unfinished or terminal probe."""
        policy, journal = self.policy, self.journal
        declare_input(
            journal,
            policy.monitor_id + ":policy",
            {
                "policy": policy.model_dump(mode="json"),
                "traffic_url": self.adapter.traffic_url,
                "control_store": str(self.control.path),
                "route_store": str(self.adapter.store.path),
            },
        )
        observations: list[ProbeObservation] = []
        references: list[ArtifactRef] = []
        for sequence in range(policy.maximum_probes):
            previous = references[-1] if references else None
            state = declare_input(
                journal,
                f"{policy.monitor_id}:probe-{sequence:03d}",
                {
                    "policy": policy.digest(),
                    "sequence": sequence,
                    "previous": previous.model_dump() if previous else None,
                },
            )
            if state.status != "completed":
                return observations, references, state
            assert state.output is not None
            item = ProbeObservation.model_validate_json(journal.artifacts.get(state.output))
            if (item.policy_sha256, item.sequence, item.previous) != (
                policy.digest(),
                sequence,
                previous,
            ):
                raise ValueError("monitor probe chain differs from frozen inputs")
            if observations and item.finished_at < observations[-1].finished_at:
                raise ValueError("monitor observation clock moved backward")
            observations.append(item)
            references.append(state.output)
            if self._result(observations, references).finished:
                return observations, references, None
        return observations, references, None

    async def observe(self, deployment_id: str) -> MonitorResult:
        """Issue at most one new probe; unresolved work never retries just because time elapsed."""
        if deployment_id != self.policy.deployment_id:
            raise ValueError("monitor deployment differs from frozen policy")
        observations, references, state = await owned_disk(self._history)
        if state is None:
            return self._result(observations, references)
        # Starting a running attempt fails its durable CAS before health can issue a request.
        running = await start_owned(self.journal, state.stage_id)
        assert running.attempt_id is not None
        before = await owned_disk(self._matches)
        health, error = None, None
        started = self.monotonic()
        if before:
            try:
                async with asyncio.timeout(self.policy.probe_timeout_seconds):
                    health = await self.adapter.health(deployment_id)
            except HTTPClosureError:
                raise
            except Exception as failure:
                error = type(failure).__name__
        elapsed = self.monotonic() - started
        matches = before and await owned_disk(self._matches)
        item = ProbeObservation(
            policy_sha256=self.policy.digest(),
            sequence=len(observations),
            previous=references[-1] if references else None,
            finished_at=self.clock(),
            elapsed_seconds=elapsed,
            route_matches=matches,
            health=health,
            error_code=error,
        )
        if observations and item.finished_at < observations[-1].finished_at:
            raise ValueError("monitor observation clock moved backward")

        def publish() -> ArtifactRef:
            """Persist raw probes before signal creation or external rollback."""
            reference = self.journal.artifacts.put(item.model_dump_json().encode())
            assert running.attempt_id is not None
            self.journal.finish(state.stage_id, running.attempt_id, reference)
            return reference

        reference = await owned_disk(publish)
        return self._result([*observations, item], [*references, reference])

    async def run(self) -> MonitorResult:
        """Bound the probe window and automatically resume rollback on detection."""
        while True:
            result = await self.observe(self.policy.deployment_id)
            if result.signal is not None:
                await owned_disk(partial(self.control.detect, result.signal))
                controller = RollbackController(self.control, self.policy.probe_timeout_seconds)
                rollback = None
                for attempt in range(self.policy.recovery_attempts):
                    rollback = await controller.resume(result.signal.signal_id, self.adapter)
                    if rollback.status in {"restored", "needs_reconciliation"}:
                        break
                    if attempt + 1 < self.policy.recovery_attempts:
                        await asyncio.sleep(self.policy.interval_seconds)
                assert rollback is not None
                return result.model_copy(update={"rollback": rollback})
            if result.finished:
                return result
            await asyncio.sleep(self.policy.interval_seconds)
