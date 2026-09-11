"""Manual local producer workflow with canonical release gates and failure-preserving cleanup."""

from datetime import UTC, datetime

from airflow.sdk import DAG, task

from finserve.registry.pipeline import evaluate_release_stage, register_produced_stage
from finserve.registry.producer_pipeline import cleanup_stage, collection_stage, freeze_stage
from finserve.registry.producer_rollout import RolloutStep, rollout_stage
from finserve.registry.producer_runtime import ProducerStep

with DAG(
    dag_id="finserve_producer_lifecycle",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["finserve", "local-producer"],
) as dag:

    @task(task_id="freeze_job", retries=0)
    def freeze_job() -> str:
        """Freeze the full request before any model fetch or Docker work can begin."""
        return freeze_stage(require_rollout=True)

    @task(retries=0)
    def collect(job_id: str, step: ProducerStep) -> str:
        """A failed owned action needs its recorded reconciliation rules, not a blind retry."""
        return collection_stage(job_id, step)

    @task(task_id="register_evidence", retries=1)
    def register(job_id: str) -> str:
        """Reconstruct actual paired receipts under the previously frozen release plan."""
        return register_produced_stage(job_id + ":release-plan")

    @task(task_id="evaluate_gates", retries=0)
    def evaluate(job_id: str) -> str:
        """Rejected quality or performance prevents every traffic mutation downstream."""
        return evaluate_release_stage(job_id)

    @task(retries=0)
    def rollout(job_id: str, step: RolloutStep) -> str:
        """Keep route preparation, deployment, acknowledgment and probation separately visible."""
        return rollout_stage(job_id, step)

    @task(task_id="cleanup_unused", trigger_rule="all_done", retries=0)
    def cleanup(job_id: str) -> str:
        """Stop only exact never-served runtimes after every producer/release task has settled."""
        outcomes = cleanup_stage(job_id)
        if "needs_reconciliation" in outcomes.values():
            raise RuntimeError("producer cleanup requires launch reconciliation")
        return job_id

    @task(task_id="release_complete", retries=0)
    def finish(probation_job: str, cleanup_job: str) -> str:
        """An all-success terminal task prevents successful cleanup masking a failed release."""
        if probation_job != cleanup_job:
            raise ValueError("release completion job mismatch")
        return probation_job

    job = freeze_job()
    previous = job
    pending = [job]
    for action in (
        "fetch",
        "build",
        "freeze",
        "baseline_launch",
        "baseline_quality",
        "baseline_performance",
        "candidate_launch",
        "candidate_quality",
        "candidate_performance",
    ):
        previous = collect.override(task_id="produce_" + action)(previous, action)
        pending.append(previous)
    previous = evaluate(register(previous))
    # Transitive dependencies settle registration and evaluation before all-done cleanup.
    pending.append(previous)
    for action in ("prepare", "deploy", "acknowledge", "probation"):
        previous = rollout.override(task_id="release_" + action)(previous, action)
        pending.append(previous)
    cleaned = cleanup(job)
    for upstream in pending:
        upstream >> cleaned
    finish(previous, cleaned)
