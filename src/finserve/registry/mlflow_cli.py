"""Mirror a persisted registry run and decision to an explicitly selected MLflow experiment."""

import argparse
from pathlib import Path

from finserve.registry.artifacts import LocalArtifactStore
from finserve.registry.metadata import Registry
from finserve.registry.mlflow import client_for_uri, mirror_run


def run_cli(arguments: list[str] | None = None) -> None:
    """Export existing verified evidence without changing deployment or promotion authority."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-url", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--decision-digest", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(arguments)
    registry = Registry(args.registry_url)
    try:
        artifacts = LocalArtifactStore(args.artifact_root)
        decision = registry.decision(args.decision_digest)
        identity = mirror_run(
            registry,
            artifacts,
            args.run_id,
            decision,
            client_for_uri(args.tracking_uri),
            args.experiment_id,
        )
        print(identity)
    finally:
        registry.close()


if __name__ == "__main__":
    run_cli()
