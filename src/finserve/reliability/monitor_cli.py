"""Run a finite automatic warm-route monitor against trusted existing local deployment stores."""

import argparse
import asyncio
import os
from pathlib import Path

import httpx

from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.metadata import Registry
from finserve.registry.model_assets import owned_disk
from finserve.registry.producer_stages import ProducerStages
from finserve.reliability.monitor import MonitorPolicy, ProbeMonitor
from finserve.reliability.rollback import DeploymentStore
from finserve.reliability.warm_routes import WarmRouteAdapter, WarmRouteStore


def existing_file(value: str) -> Path:
    """Reject missing control inputs instead of silently initializing a different deployment."""
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise argparse.ArgumentTypeError("existing regular file required")
    return path


def read_policy(arguments: argparse.Namespace) -> MonitorPolicy:
    """Bound input reads and reject source-checkout output paths before acquiring stores."""
    policy_path: Path = arguments.policy
    with policy_path.open("rb") as source:
        data = source.read(16385)
    if len(data) > 16384:
        raise ValueError("monitor policy exceeds 16 KiB")
    repository = Path(__file__).resolve().parents[3]
    for value in (arguments.journal, arguments.artifacts):
        target = Path(value).resolve()
        if target == repository or repository in target.parents:
            raise ValueError("monitor evidence must remain outside the source repository")
    return MonitorPolicy.model_validate_json(data)


async def run(arguments: argparse.Namespace) -> int:
    """Keep the ingress key only in memory and print a bounded non-secret result receipt."""
    policy = await owned_disk(lambda: read_policy(arguments))
    key = os.environ.get("FINSERVE_API_KEY", "")
    if not 16 <= len(key) <= 4096 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError("FINSERVE_API_KEY must be a bounded printable runtime secret")
    registry = Registry("sqlite:///" + str(arguments.journal))
    try:
        journal = ProducerStages(registry, LocalArtifactStore(arguments.artifacts))
        control, routes = DeploymentStore(arguments.control), WarmRouteStore(arguments.routes)
        async with httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            headers={"Authorization": "Bearer " + key, "Accept-Encoding": "identity"},
            timeout=policy.probe_timeout_seconds,
        ) as client:
            adapter = WarmRouteAdapter(
                routes, arguments.traffic_url, client, policy.probe_timeout_seconds
            )
            result = await ProbeMonitor(policy, journal, control, adapter).run()
            print(result.model_dump_json(), flush=True)
            return 2 if result.rollback is not None and result.rollback.status != "restored" else 0
    finally:
        registry.close()


def main() -> None:
    """Explicit files and endpoint scope the monitor without provisioning a backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=existing_file, required=True)
    parser.add_argument("--control", type=existing_file, required=True)
    parser.add_argument("--routes", type=existing_file, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--traffic-url", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
