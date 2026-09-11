"""Registry/artifact correctness is tested without relying on AWS availability or credentials."""

import hashlib
import io
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from finserve.contracts.deployment import Revision
from finserve.registry.artifacts import LocalArtifactStore, S3ArtifactStore
from finserve.registry.metadata import Registry, RegistryConflict, schema


class ExistingObject(Exception):
    """Simulate the SDK's conditional-write failure for an already published immutable object."""

    response = {"Error": {"Code": "PreconditionFailed"}}


class MemoryS3:
    """An injected transport fixture asserts conditional writes and closes response bodies."""

    def __init__(self) -> None:
        """Track bytes and request headers independently of the content-addressed wrapper."""
        self.objects: dict[str, bytes] = {}
        self.calls: list[dict[str, object]] = []
        self.body = io.BytesIO()

    def put_object(self, **kwargs: object) -> dict[str, object]:
        """Require IfNoneMatch so a retry cannot overwrite an object whose digest already exists."""
        self.calls.append(kwargs)
        assert kwargs["IfNoneMatch"] == "*"
        key, value = str(kwargs["Key"]), kwargs["Body"]
        assert isinstance(value, bytes)
        if key in self.objects:
            raise ExistingObject
        self.objects[key] = value
        return {}

    def get_object(self, **kwargs: object) -> dict[str, object]:
        """Return a new closeable body to test connection cleanup on success and corruption."""
        self.body = io.BytesIO(self.objects[str(kwargs["Key"])])
        return {"Body": self.body}


def test_local_store_idempotency_corruption_and_namespace(tmp_path: Path) -> None:
    """A named artifact is accepted only when its bytes still match the immutable reference."""
    store = LocalArtifactStore(tmp_path / "objects", maximum_bytes=100)
    reference = store.put(b"verified bytes")
    assert store.put(b"verified bytes") == reference
    assert store.get(reference) == b"verified bytes"
    with pytest.raises(ValueError):
        store.get(reference.model_copy(update={"namespace": "file:///another-store"}))
    target = tmp_path / "objects/sha256" / reference.sha256[:2] / reference.sha256
    target.write_bytes(b"corruption")
    with pytest.raises(ValueError):
        store.get(reference)
    with pytest.raises(ValueError):
        store.put(b"x" * 101)


def test_s3_conditional_put_and_download_checksum() -> None:
    """S3 ETags are irrelevant; the independently downloaded bytes must match their SHA256."""
    client = MemoryS3()
    store = S3ArtifactStore("fixture-bucket", "evidence", client)
    reference = store.put(b"content")
    assert reference.sha256 == hashlib.sha256(b"content").hexdigest()
    assert store.put(b"content") == reference
    assert client.body.closed
    key = str(client.calls[0]["Key"])
    client.objects[key] = b"wrong"
    with pytest.raises(ValueError):
        store.get(reference)
    assert client.body.closed


def test_portable_postgresql_schema_compiles_without_claiming_database_execution() -> None:
    """Compile real PostgreSQL DDL; actual PostgreSQL transaction tests remain a separate tier."""
    statements = [
        str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for table in schema.sorted_tables
    ]
    assert len(statements) >= 7
    assert any("FOREIGN KEY" in statement for statement in statements)
    assert any("UNIQUE (locked_deployment)" in statement for statement in statements)


def test_immutable_jobs_and_compare_and_swap(tmp_path: Path) -> None:
    """Changed job inputs and stale worker versions cannot overwrite durable lifecycle truth."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.db"))
    try:
        state = registry.create_job("job", '{"version":1}')
        assert registry.create_job("job", '{"version":1}') == state
        with pytest.raises(RegistryConflict):
            registry.create_job("job", '{"version":2}')
        state = registry.claim("job", "deployment", "worker-1", 30)
        with pytest.raises(RegistryConflict):
            registry.claim("job", "deployment", "worker-2", 30)
        updated = registry.advance(state, "worker-1", status="evaluated", decision_digest="f" * 64)
        assert updated.version == 1
        with pytest.raises(RegistryConflict):
            registry.advance(state, "worker-1", status="deploying")
        assert registry.history("job") == [updated]
        registry.release("job", "worker-1")
    finally:
        registry.close()


def test_identity_mutation_missing_records_and_illegal_stage_are_rejected(tmp_path: Path) -> None:
    """Registry truth forbids revision mutation and skipping registration-to-deployment gates."""
    registry = Registry("sqlite:///" + str(tmp_path / "registry.db"))
    try:
        value = Revision(
            revision_id="fixture",
            model_revision="weights-v1",
            tokenizer_revision="tokenizer-v1",
            source_revision="source-v1",
            image_digest="sha256:" + "a" * 64,
            config_digest="b" * 64,
            engine="fixture",
            engine_config="fixture",
        )
        registry.register_revision(value)
        assert registry.revision(value.revision_id) == value
        with pytest.raises(RegistryConflict):
            registry.register_revision(value.model_copy(update={"engine_config": "changed"}))
        with pytest.raises(KeyError):
            registry.run("missing")
        with pytest.raises(ValueError):
            registry.register_model("", "tokenizer")
        registry.create_job("job", "{}")
        state = registry.claim("job", "deployment", "worker", 30)
        with pytest.raises(RegistryConflict):
            registry.advance(state, "worker", status="promoted")
        with pytest.raises(RegistryConflict):
            registry.advance(state, "worker", status="evaluated")
        forged = state.model_copy(update={"status": "verifying", "decision_digest": "a" * 64})
        with pytest.raises(RegistryConflict):
            registry.advance(forged, "worker", status="promoted")
        with pytest.raises(RegistryConflict):
            registry.advance(state, "worker", job_id="different")
        assert registry.job("job") == state
        with pytest.raises(ValueError):
            registry.claim("job", "deployment", "worker", float("nan"))
        registry.release("job", "worker")
    finally:
        registry.close()
