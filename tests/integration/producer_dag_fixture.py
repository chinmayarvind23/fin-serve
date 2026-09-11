"""Execute the real Airflow DAG with explicit task fixtures, including gate-failure cleanup."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


def freeze(*, require_rollout: bool = False) -> str:
    """Identify this synthetic scheduler test without fetching a model or starting Docker."""
    assert require_rollout
    return "scheduler-fixture"


def action(job_id: str, step: str = "") -> str:
    """Record task execution order; optionally model a failed canonical gate."""
    output = Path(os.environ["FINSERVE_DAG_FIXTURE_TRACE"])
    with output.open("a") as stream:
        stream.write(step + "\n")
    if os.environ["FINSERVE_DAG_FIXTURE_FAULT"] == step:
        raise RuntimeError("explicit scheduler fixture failure: " + step)
    return job_id


def register(plan_id: str) -> str:
    """Retain the producer plan handoff convention in the scheduler fixture."""
    return action(plan_id.removesuffix(":release-plan"), "register")


def evaluate(job_id: str) -> str:
    """A gate fixture checks dependency semantics, not performance or model quality."""
    return action(job_id, "evaluate")


def cleanup(job_id: str) -> dict[str, str]:
    """Confirm that all-done cleanup executes after a failed upstream gate."""
    action(job_id, "cleanup")
    return {"baseline": "not_launched", "candidate": "not_launched"}


def main() -> None:
    """Use an isolated actual Airflow metadata database and retain both complete run results."""
    workspace = Path(sys.argv[1]).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    os.environ.update(
        {
            "AIRFLOW_HOME": str(workspace / "airflow"),
            "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
            "AIRFLOW__CORE__DAGS_FOLDER": str(repository / "pipelines/airflow_dags"),
        }
    )
    sys.path.insert(0, str(repository / "pipelines/airflow_dags"))
    subprocess.run([sys.executable, "-m", "airflow", "db", "migrate"], check=True)
    module = importlib.import_module("finserve_producer")
    module.__dict__.update(
        {
            "freeze_stage": freeze,
            "collection_stage": action,
            "register_produced_stage": register,
            "evaluate_release_stage": evaluate,
            "rollout_stage": action,
            "cleanup_stage": cleanup,
        }
    )
    results: list[dict[str, object]] = []
    for fault in ("none", "evaluate"):
        trace = workspace / (fault + ".trace")
        os.environ["FINSERVE_DAG_FIXTURE_TRACE"] = str(trace)
        os.environ["FINSERVE_DAG_FIXTURE_FAULT"] = fault
        result = module.dag.test()
        events = trace.read_text().splitlines()
        record: dict[str, object] = {"fault": fault, "state": str(result.state), "events": events}
        results.append(record)
        (workspace / "results.json").write_text(json.dumps(results, indent=2))
        assert str(result.state) == ("success" if fault == "none" else "failed")
        assert events[-1] == "cleanup"
        assert ("prepare" in events) == (fault == "none")
    print(json.dumps({"scope": "actual Airflow; synthetic task callbacks", "results": results}))


if __name__ == "__main__":
    main()
