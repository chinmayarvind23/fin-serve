"""Airflow 3 public-SDK DAG for offline evidence and verified candidate activation."""

from datetime import UTC, datetime

from airflow.sdk import DAG, task

from finserve.registry.pipeline import deploy_stage, evaluate_stage, register_stage

with DAG(
    dag_id="finserve_evidence_lifecycle",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["finserve", "offline-evidence"],
) as dag:

    @task(task_id="register_evidence", retries=1)
    def register_evidence() -> str:
        """Freeze and register input artifacts; task retries preserve the original specification."""
        return register_stage()

    @task(task_id="evaluate_gates", retries=0)
    def evaluate_gates(job_id: str) -> str:
        """A failed quality or performance gate prevents downstream activation."""
        return evaluate_stage(job_id)

    @task(task_id="deploy_verified_candidate", retries=1)
    def deploy_verified_candidate(job_id: str) -> str:
        """A retry reconciles the durable operation before any possible external action replay."""
        return deploy_stage(job_id)

    deploy_verified_candidate(evaluate_gates(register_evidence()))
