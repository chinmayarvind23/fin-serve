"""Actual optional SDK contracts; AWS requests remain stubbed and MLflow uses local SQLite only."""

import base64
import hashlib
import importlib
import importlib.util
import io
from pathlib import Path
from typing import cast

import pytest
from test_lifecycle import registered
from test_promotion import artifact

from finserve.registry.artifacts import S3ArtifactStore, S3Client
from finserve.registry.lifecycle import LifecycleService
from finserve.registry.mlflow import client_for_uri, mirror_run


def test_boto3_s3_conditional_write_contract() -> None:
    """Use real botocore request validation to verify IfNoneMatch/checksum API compatibility."""
    boto3 = pytest.importorskip("boto3")
    sdk = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="fixture", aws_secret_access_key="fixture"
    )
    stubber = importlib.import_module("botocore.stub").Stubber(sdk)
    response_module = importlib.import_module("botocore.response")
    data = b"verified SDK contract fixture"
    digest = hashlib.sha256(data)
    key = f"evidence/sha256/{digest.hexdigest()[:2]}/{digest.hexdigest()}"
    expected = {
        "Bucket": "fixture-bucket",
        "Key": key,
        "Body": data,
        "IfNoneMatch": "*",
        "ChecksumSHA256": base64.b64encode(digest.digest()).decode("ascii"),
    }
    stubber.add_response("put_object", {}, expected)
    stubber.add_response(
        "get_object",
        {"Body": response_module.StreamingBody(io.BytesIO(data), len(data))},
        {"Bucket": "fixture-bucket", "Key": key},
    )
    stubber.add_client_error(
        "put_object",
        service_error_code="PreconditionFailed",
        http_status_code=412,
        expected_params=expected,
    )
    stubber.add_response(
        "get_object",
        {"Body": response_module.StreamingBody(io.BytesIO(data), len(data))},
        {"Bucket": "fixture-bucket", "Key": key},
    )
    with stubber:
        store = S3ArtifactStore("fixture-bucket", "evidence", cast(S3Client, sdk))
        reference = store.put(data)
        assert store.put(data) == reference
        stubber.assert_no_pending_responses()


async def test_real_local_mlflow_evidence_mirror(tmp_path: Path) -> None:
    """Exercise actual MLflow APIs with explicitly synthetic metric fixtures."""
    module = pytest.importorskip("mlflow")
    registry, artifacts, specification, _ = registered(tmp_path)
    try:
        state = await LifecycleService(registry, artifacts).run(specification, evaluate_only=True)
        assert state.decision_digest is not None
        uri = "sqlite:///" + str(tmp_path / "mlflow.db")
        sdk = module.MlflowClient(tracking_uri=uri)
        experiment = sdk.create_experiment(
            "FinServe local integration fixture",
            artifact_location=(tmp_path / "mlflow-artifacts").as_uri(),
        )
        run_id = mirror_run(
            registry,
            artifacts,
            specification.candidate_run_id,
            registry.decision(state.decision_digest),
            client_for_uri(uri),
            experiment,
        )
        actual = sdk.get_run(run_id)
        assert actual.info.status == "FINISHED"
        assert actual.data.metrics["candidate_accuracy"] == 1
        assert actual.data.tags["finserve.run_id"] == specification.candidate_run_id
        assert len(sdk.list_artifacts(run_id, "verified_evidence")) == 4
        artifact(tmp_path / "unrelated", "unrelated-run", 2, 0.2)
        unrelated = registry.register_run(tmp_path / "unrelated", artifacts, specification.target)
        for wrong_run in (unrelated.run_id, specification.baseline_run_id):
            with pytest.raises(ValueError):
                mirror_run(
                    registry,
                    artifacts,
                    wrong_run,
                    registry.decision(state.decision_digest),
                    client_for_uri(uri),
                    experiment,
                )
        assert len(sdk.search_runs([experiment])) == 1
    finally:
        registry.close()


def test_airflow_public_sdk_dag_structure() -> None:
    """Import the actual Airflow3 DAG and check that evaluation is upstream of deployment."""
    pytest.importorskip("airflow.sdk")
    directory = Path(__file__).resolve().parents[2] / "pipelines/airflow_dags"
    specification = importlib.util.spec_from_file_location(
        "finserve_lifecycle_check", directory / "finserve_lifecycle.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    dag = module.dag
    assert set(dag.task_ids) == {"register_evidence", "evaluate_gates", "deploy_verified_candidate"}
    assert dag.get_task("register_evidence").downstream_task_ids == {"evaluate_gates"}
    assert dag.get_task("evaluate_gates").downstream_task_ids == {"deploy_verified_candidate"}
