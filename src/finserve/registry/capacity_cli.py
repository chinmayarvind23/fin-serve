"""Run a frozen local capacity plan through actual managed model lifecycle executors."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from finserve.contracts.capacity import CapacityPlan, CapacityState
from finserve.registry.local_capacity import CapacityController, freeze_capacity
from finserve.registry.managed_runtime import DockerRuntime
from finserve.registry.model_assets import owned_disk
from finserve.registry.pipeline import runtime
from finserve.registry.producer_stages import ProducerStages
from finserve.reliability.rollback import DeploymentStore
from finserve.reliability.warm_routes import WarmRouteStore


async def run_plan(
    journal: ProducerStages,
    routes: WarmRouteStore,
    control: DeploymentStore,
    plan: CapacityPlan,
) -> CapacityState:
    """The real adapter owns all Docker calls; frozen sample bounds also cap blocked retries."""
    with routes.transaction() as connection:
        exists = connection.execute(
            "SELECT 1 FROM warm_capacity_plans WHERE id=?", (plan.plan_id,)
        ).fetchone()
    if not exists:
        await owned_disk(lambda: freeze_capacity(journal, routes, control, plan))
    async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
        controller = CapacityController(journal, routes, control, plan, DockerRuntime(), client)
        try:
            for _ in range(plan.max_samples):
                state = await controller.tick()
                if state.phase == "closed":
                    return state
                await asyncio.sleep(plan.policy.sample_seconds)
        finally:
            # This is an explicit drain attempt, not a TTL-based declaration of absence.
            state = await controller.close()
    return state


def run_cli(arguments: list[str] | None = None) -> int:
    """Storage and frozen plan are trusted local inputs; no request can choose a runtime image."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(arguments)
    repository = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    if output == repository or repository in output.parents:
        raise ValueError("capacity evidence must be outside source")
    if not args.routes.is_file() or not args.control.is_file():
        raise ValueError("capacity runner requires preexisting route/control stores")
    if args.plan.stat().st_size > 2 * 1024**2:
        raise ValueError("capacity input exceeds byte limit")
    plan = CapacityPlan.model_validate_json(args.plan.read_bytes())
    routes, control = WarmRouteStore(args.routes), DeploymentStore(args.control)
    registry, artifacts = runtime()
    started = time.time()
    try:
        state = asyncio.run(run_plan(ProducerStages(registry, artifacts), routes, control, plan))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as destination:
            json.dump(
                {
                    "plan_id": plan.plan_id,
                    "started_at": started,
                    "finished_at": time.time(),
                    "state": state.model_dump(),
                    "route_store": routes.identity,
                    "control_store": control.identity,
                },
                destination,
                sort_keys=True,
                indent=2,
            )
        return 0 if state.phase == "closed" and state.error is None else 2
    finally:
        registry.close()


if __name__ == "__main__":
    raise SystemExit(run_cli())
