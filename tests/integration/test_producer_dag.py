"""The actual public Airflow DAG retains failure state after all-done runtime cleanup."""

import importlib.util
from pathlib import Path

import pytest


def test_producer_dag_dependency_boundaries() -> None:
    """Check parsed SDK tasks: collection precedes traffic and cleanup is not the success leaf."""
    pytest.importorskip("airflow.sdk")
    path = Path(__file__).resolve().parents[2] / "pipelines/airflow_dags/finserve_producer.py"
    spec = importlib.util.spec_from_file_location("producer_dag_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dag = module.dag
    assert len(dag.task_ids) == 18
    assert dag.get_task("cleanup_unused").trigger_rule == "all_done"
    assert dag.get_task("release_complete").trigger_rule == "all_success"
    assert dag.get_task("release_complete").upstream_task_ids == {
        "release_probation",
        "cleanup_unused",
    }
    assert {item.task_id for item in dag.leaves} == {"release_complete"}
    assert dag.get_task("release_prepare").upstream_task_ids == {"evaluate_gates"}
    assert dag.get_task("produce_candidate_launch").upstream_task_ids == {
        "produce_baseline_performance"
    }
    assert dag.get_task("cleanup_unused").upstream_task_ids >= {
        "freeze_job",
        "produce_fetch",
        "produce_candidate_performance",
        "evaluate_gates",
        "release_prepare",
        "release_deploy",
        "release_acknowledge",
        "release_probation",
    }
