"""Task entry points join approved producer evidence to warm deployment and probation."""

import asyncio
import os
from typing import Literal, get_args

import httpx

from finserve.contracts.rollout import RolloutSettings
from finserve.registry.lifecycle import LifecycleService, LifecycleSpec
from finserve.registry.produced_release import register_produced_release
from finserve.registry.producer_pipeline import execution_input, journal_runtime
from finserve.registry.producer_runtime import existing_baseline
from finserve.registry.release_activation import (
    acknowledge_release,
    complete_probation,
    prepare_release,
)
from finserve.registry.release_gate import evaluate_gate
from finserve.reliability.monitor import MonitorPolicy, ProbeMonitor
from finserve.reliability.warm_routes import WarmRouteAdapter

RolloutStep = Literal["prepare", "deploy", "acknowledge", "probation"]


def traffic_client(settings: RolloutSettings) -> httpx.AsyncClient:
    """Authentication comes from the trusted worker environment and is never written to CAS."""
    key = os.environ.get("FINSERVE_API_KEY", "")
    if not 16 <= len(key) <= 4096 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError("FINSERVE_API_KEY must be a bounded printable runtime secret")
    return httpx.AsyncClient(
        trust_env=False,
        follow_redirects=False,
        headers={"Authorization": "Bearer " + key, "Accept-Encoding": "identity"},
        timeout=settings.probe_timeout_seconds,
    )


def rollout_stage(job_id: str, step: RolloutStep) -> str:
    """Use only frozen settings and canonical producer identities for every rollout task."""
    if step not in get_args(RolloutStep):
        raise ValueError("unsupported rollout action")
    with journal_runtime() as journal:
        frozen = execution_input(journal, job_id)
        settings = frozen.execution.rollout
        if settings is None:
            raise ValueError("rollout settings must be frozen before producer collection")
        routes, control = frozen.stores()
        # Reconstruct producer receipts; an unrelated canonical job is not a producer release.
        register_produced_release(journal, job_id + ":release-plan")
        spec = LifecycleSpec.model_validate_json(journal.registry.specification(job_id))
        producer = frozen.execution.producer
        baseline_id = (
            existing_baseline(journal, producer).revision.revision_id
            if producer.existing_baseline_stage is not None
            else job_id + "-baseline"
        )
        if (
            spec.deployment_id != producer.deployment_id
            or spec.expected_generation != producer.expected_generation
            or spec.expected_revision != baseline_id
            or spec.target.revision_id != job_id + "-candidate"
            or spec.policy != producer.policy
        ):
            raise ValueError("rollout differs from frozen producer lifecycle")
        if evaluate_gate(journal.registry, journal.artifacts, job_id).status != "approved":
            raise ValueError("rollout requires approved canonical evidence")

        async def run() -> None:
            """Own authenticated transport through health, cutover and any rollback work."""
            async with traffic_client(settings) as client:
                adapter = WarmRouteAdapter(
                    routes,
                    settings.traffic_url,
                    client,
                    settings.probe_timeout_seconds,
                )
                if step == "prepare":
                    await prepare_release(
                        journal.registry, journal.artifacts, job_id, control, adapter
                    )
                elif step == "deploy":
                    state = await LifecycleService(journal.registry, journal.artifacts).run(
                        spec, adapter
                    )
                    if state.status != "promoted" or state.last_error is not None:
                        raise RuntimeError("producer deployment requires reconciliation")
                elif step == "acknowledge":
                    await acknowledge_release(
                        journal.registry, journal.artifacts, job_id, control, adapter
                    )
                else:
                    policy = MonitorPolicy(
                        monitor_id=job_id + "-probation",
                        deployment_id=spec.deployment_id,
                        revision_id=spec.target.revision_id,
                        revision_digest=spec.target.digest(),
                        generation=spec.expected_generation + 1,
                        **settings.model_dump(exclude={"traffic_url"}),
                    )
                    monitor = ProbeMonitor(policy, journal, control, adapter)
                    await monitor.run()
                    await complete_probation(job_id, monitor)

        asyncio.run(run())
        return job_id
