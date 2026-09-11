"""Explicit request-shape maps cannot silently change provenance or select by evaluator metadata."""

import inspect
import json
from pathlib import Path

import pytest

from finserve.benchmark.constraint_mapping import (
    RequestConstraintBinding,
    RequestConstraintMap,
    prompt_digest,
    read_constraint_map,
)
from finserve.benchmark.request_mapping import END, SEPARATOR, START
from finserve.benchmark.runner import RunConfig, request_payload
from finserve.benchmark.workload import WorkItem
from finserve.contracts.output_constraint import OutputConstraint
from finserve.registry.engine_entrypoint import VLLMParameters


def binding(prompt: str, kind: str | None = "integer") -> RequestConstraintBinding:
    """Test declarations name only prompt and requested syntax, never expected answers."""
    return RequestConstraintBinding(
        prompt_sha256=prompt_digest(prompt),
        constraint=None if kind is None else OutputConstraint.model_validate({"kind": kind}),
    )


def config(mapping: RequestConstraintMap, **changes: object) -> RunConfig:
    """Use declared synthetic runtime identity while exercising the pinned backend requirement."""
    values: dict[str, object] = dict(
        engine="vllm",
        engine_config=VLLMParameters(structured_output_backend="xgrammar").model_dump_json(),
        output_constraints=mapping,
        constraint_transport="native_vllm",
    )
    values.update(changes)
    return RunConfig.model_validate(values)


def test_original_bytes_and_reporting_independence() -> None:
    """Hash before role extraction; equivalent-looking whitespace and Unicode remain different."""
    prompt = START + "system" + SEPARATOR + "Return an integer." + END
    mapping = RequestConstraintMap(entries=(binding(prompt),))
    run = config(
        mapping, request_api="chat", prompt_mapping="chatml_roles_v1", chat_template_sha256="a" * 64
    )
    item = WorkItem(case_id="one", family="label", prompt=prompt)
    payload = request_payload(item, run)
    assert payload["structured_outputs"] == OutputConstraint(kind="integer").vllm_parameters()
    assert (
        request_payload(
            item.model_copy(update={"case_id": "unrelated", "family": "different"}), run
        )
        == payload
    )
    for other in (prompt + " ", "Return an integer.", prompt.replace("\n", "\r\n")):
        with pytest.raises(ValueError, match="missing"):
            mapping.resolve(other)
    assert prompt_digest("é") != prompt_digest("e\u0301")
    assert list(inspect.signature(RequestConstraintMap.resolve).parameters) == ["self", "prompt"]


def test_canonical_map_and_explicit_unconstrained_request() -> None:
    """Entry order is immaterial; an explicit None is different from a missing binding."""
    entries = (binding("one"), binding("two", None))
    mapping = RequestConstraintMap(entries=entries)
    assert mapping.digest() == RequestConstraintMap(entries=entries[::-1]).digest()
    assert mapping.resolve("two") is None
    assert "structured_outputs" not in request_payload(
        WorkItem(case_id="x", prompt="two"), config(mapping)
    )
    changed = RequestConstraintMap(entries=(entries[0], binding("two", "decimal")))
    assert changed.digest() != mapping.digest()
    assert config(changed).request_mapping_digest() != config(mapping).request_mapping_digest()
    assert (
        config(mapping, constraint_transport="finserve").request_mapping_digest()
        != config(mapping).request_mapping_digest()
    )
    with pytest.raises(ValueError, match="duplicate"):
        RequestConstraintMap(entries=(entries[0], entries[0]))
    with pytest.raises(ValueError, match="duplicate"):
        RequestConstraintMap(entries=(entries[0], binding("one", "decimal")))


@pytest.mark.parametrize(
    "changes",
    [
        {"constraint_transport": None},
        {"output_constraints": None},
        {"engine": "sglang"},
        {"engine_config": "{}"},
        {"engine_config": "not-json"},
        {"engine_config": '{"structured_output_backend":"auto"}'},
    ],
)
def test_missing_or_unsupported_configuration_is_rejected(changes: dict[str, object]) -> None:
    """Selecting a new request mapping requires a declared transport and pinned native backend."""
    with pytest.raises(ValueError):
        config(RequestConstraintMap(entries=(binding("one"),)), **changes)


def test_sidecar_rejects_duplicates_unbounded_bytes_and_metadata(tmp_path: Path) -> None:
    """Validate original sidecar bytes before canonicalization can hide duplicate JSON fields."""
    path = tmp_path / "constraints.json"
    mapping = RequestConstraintMap(entries=(binding("one"),))
    path.write_text(mapping.model_dump_json(), encoding="utf-8")
    assert read_constraint_map(path) == mapping
    for content in ('{"entries": [], "entries": []}', " " * 262145):
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            read_constraint_map(path)
    for field in ("expected", "case_id", "family", "evaluator_kind"):
        data = {**binding("one").model_dump(), field: "forbidden"}
        with pytest.raises(ValueError):
            RequestConstraintBinding.model_validate(data)
    with pytest.raises(ValueError):
        RequestConstraintBinding.model_validate({"prompt_sha256": prompt_digest("one")})
    with pytest.raises(ValueError):
        RequestConstraintMap.model_validate(
            {"entries": [binding(str(index)).model_dump() for index in range(257)]}
        )
    for prompt in ("", "é" * 65537):
        with pytest.raises(ValueError):
            prompt_digest(prompt)
    assert json.loads(mapping.model_dump_json())["schema_version"] == "prompt-output-contracts-v1"


def test_deferred_producer_template_cannot_execute() -> None:
    """Templates can freeze shape before an image exists, but payloads require resolved identity."""
    mapping = RequestConstraintMap(entries=(binding("one"),))
    deferred = RunConfig(output_constraints=mapping, constraint_transport="native_vllm")
    assert deferred.engine == deferred.engine_config == "undeclared"
    assert deferred.request_mapping_digest() == config(mapping).request_mapping_digest()
    with pytest.raises(ValueError, match="xgrammar"):
        request_payload(WorkItem(case_id="one", prompt="one"), deferred)
