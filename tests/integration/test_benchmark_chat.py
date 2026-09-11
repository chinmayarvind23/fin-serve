"""Chat evidence preserves raw prompts while freezing a distinct tokenizer/API cohort."""

import json
import time
from pathlib import Path

import httpx
import pytest

from finserve.benchmark.runner import (
    RunConfig,
    StreamState,
    request_one,
    request_payload,
    run_benchmark,
    validate_comparison,
    validate_evidence,
)
from finserve.benchmark.workload import default_workload
from finserve.evaluation.quality import default_suite
from finserve.reliability.promotion import QualityEvidence, check_quality_identity


def config(**changes: object) -> RunConfig:
    """Use explicit fixture identities without representing them as executed model evidence."""
    values: dict[str, object] = dict(
        requests=2,
        warmup=0,
        concurrency=1,
        hardware="fixture",
        revision="fixture",
        model_revision="fixture",
        tokenizer_revision="fixture",
        engine="fixture",
        engine_config="fixture",
        request_api="chat",
        system_prompt="Only the answer.",
        chat_template_sha256="a" * 64,
    )
    values.update(changes)
    return RunConfig.model_validate(values)


def response(request: httpx.Request) -> httpx.Response:
    """One content delta carries three authoritative tokens; role-only events carry none."""
    return httpx.Response(
        200,
        request=request,
        content=(
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"usage":{"completion_tokens":3}}\n\n'
            "data: [DONE]\n\n"
        ),
    )


def test_mapping_and_legacy_encoding() -> None:
    """Old configuration JSON stays unchanged; new chat mapping is explicit and hash-bound."""
    legacy = RunConfig().model_dump()
    assert not {"request_api", "system_prompt", "chat_template_sha256"} & legacy.keys()
    assert RunConfig.model_validate(legacy).model_dump() == legacy
    assert RunConfig().model_dump(include={"requests"}) == {"requests": 100}
    assert "request_api" not in RunConfig().model_dump(exclude={"request_api"})
    item = default_workload().items[0]
    frozen = config()
    payload = request_payload(item, frozen)
    assert payload["messages"] == [
        {"role": "system", "content": frozen.system_prompt},
        {"role": "user", "content": item.prompt},
    ]
    assert payload["stream_options"] == {"include_usage": True}
    assert "prompt" not in payload
    assert request_payload(item, config(system_prompt=""))["messages"] == [
        {"role": "user", "content": item.prompt}
    ]
    assert (
        config(system_prompt="changed").request_mapping_digest() != frozen.request_mapping_digest()
    )
    with pytest.raises(ValueError, match="template digest"):
        RunConfig(request_api="chat")
    with pytest.raises(ValueError, match="chat-only"):
        RunConfig(system_prompt="unused")


@pytest.mark.parametrize(
    "api,choice",
    [
        ("chat", {"text": "wrong protocol"}),
        ("completions", {"delta": {"content": "wrong protocol"}}),
    ],
)
def test_response_protocol_cannot_change(api: str, choice: dict[str, object]) -> None:
    """An endpoint cannot silently substitute completion content for the declared chat cohort."""
    state = StreamState.model_validate({"request_api": api})
    with pytest.raises(ValueError, match="response_api"):
        state.consume(json.dumps({"choices": [choice]}))


async def test_chat_timing_and_raw_wire_mapping() -> None:
    """Role events do not count as content, and parsed usage is independent of event count."""
    state = StreamState(request_api="chat")
    state.consume('{"choices":[{"delta":{"role":"assistant"}}]}')
    assert state.first_content_s is None
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Retain the actual serialized HTTP request before returning fixture events."""
        seen.append(json.loads(request.content))
        return response(request)

    item = default_workload().items[0]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        record = await request_one(
            client,
            "http://test/v1/chat/completions",
            item,
            0,
            time.perf_counter(),
            config(),
            "measured",
        )
    assert record.success and record.output == "answer" and record.generated_tokens == 3
    assert record.first_content_s is not None
    assert seen == [request_payload(item, config())]


async def test_chat_comparison_and_quality_bind_mapping(tmp_path: Path) -> None:
    """Changed chat inputs fail comparison; missing or mismatched quality mapping fails closed."""
    left, right = tmp_path / "left", tmp_path / "right"
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        for output in (left, right):
            await run_benchmark(
                client, "http://test/v1/chat/completions", default_workload(), config(), output
            )
    validate_comparison(left, right)
    first, second = validate_evidence(left), validate_evidence(right)
    suite = default_suite()
    evidence = QualityEvidence(
        suite_hash=suite.digest(),
        evaluator_version=suite.evaluator_version,
        baseline_run_id=json.loads((left / "manifest.json").read_text())["run_id"],
        candidate_run_id=json.loads((right / "manifest.json").read_text())["run_id"],
        reference_model_revision="fixture",
        candidate_model_revision="fixture",
        reference={},
        candidate={},
    )
    assert "request_mapping_sha256" not in evidence.model_dump()
    assert evidence.model_dump(include={"suite_hash"}) == {"suite_hash": suite.digest()}
    with pytest.raises(ValueError, match="mapping"):
        check_quality_identity(evidence, suite, left, right, first, second)
    bound = evidence.model_copy(
        update={"request_mapping_sha256": config().request_mapping_digest()}
    )
    check_quality_identity(bound, suite, left, right, first, second)
    completion = RunConfig.model_validate(
        {
            key: value
            for key, value in config().model_dump().items()
            if key not in {"request_api", "system_prompt", "chat_template_sha256"}
        }
    )
    for before, after in (
        (first.model_copy(update={"configuration": completion}), second),
        (first, second.model_copy(update={"configuration": completion})),
    ):
        with pytest.raises(ValueError, match="mapping"):
            check_quality_identity(evidence, suite, left, right, before, after)
        with pytest.raises(ValueError, match="mapping"):
            check_quality_identity(bound, suite, left, right, before, after)
    document = json.loads((right / "manifest.json").read_text())
    for field, value in [("system_prompt", "changed"), ("chat_template_sha256", "b" * 64)]:
        altered = json.loads(json.dumps(document))
        altered["configuration"][field] = value
        (right / "manifest.json").write_text(json.dumps(altered))
        with pytest.raises(ValueError, match=field):
            validate_comparison(left, right)
