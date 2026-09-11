"""Strict offline role extraction preserves input facts and existing request identities."""

import hashlib
import inspect

import pytest

from finserve.benchmark.request_mapping import END, SEPARATOR, START, chatml_roles
from finserve.benchmark.runner import RunConfig, request_payload
from finserve.benchmark.workload import WorkItem


def test_roles_round_trip_and_explicit_instruction() -> None:
    """Whitespace and Unicode survive extraction; only the configured instruction is appended."""
    system, user = "Keep facts. \n", "\n  café\t€ -12.50\n"
    prompt = START + system + SEPARATOR + user + END
    roles = chatml_roles(prompt, "")
    assert roles == [{"role": "system", "content": system}, {"role": "user", "content": user}]
    assert START + roles[0]["content"] + SEPARATOR + roles[1]["content"] + END == prompt
    configured = chatml_roles(prompt, "Preserve requested precision.")
    assert configured[0]["content"] == system + "\n\nPreserve requested precision."
    assert configured[1] == roles[1]
    assert list(inspect.signature(chatml_roles).parameters) == ["prompt", "instruction"]


def test_plain_input_and_mapping_are_explicit() -> None:
    """Plain text remains literal; legacy chat never interprets embedded role delimiters."""
    plain = " \nReturn 3 decimal places.\t "
    assert chatml_roles(plain, "") == [{"role": "user", "content": plain}]
    assert chatml_roles(plain, "policy") == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": plain},
    ]
    prompt = START + "context" + SEPARATOR + "question" + END
    item = WorkItem(case_id="irrelevant", family="unrelated", prompt=prompt)
    legacy = RunConfig(request_api="chat", chat_template_sha256="a" * 64)
    mapped = legacy.model_copy(update={"prompt_mapping": "chatml_roles_v1"})
    assert request_payload(item, legacy)["messages"] == [{"role": "user", "content": prompt}]
    assert request_payload(item, mapped)["messages"] == chatml_roles(prompt, "")
    assert request_payload(
        item.model_copy(update={"case_id": "other", "family": "other"}), mapped
    ) == request_payload(item, mapped)
    assert legacy.request_mapping_digest() != mapped.request_mapping_digest()
    with pytest.raises(ValueError, match="chat-only"):
        RunConfig(prompt_mapping="chatml_roles_v1")


@pytest.mark.parametrize(
    "prompt",
    [
        "<|im_start",
        "<|im_end|>",
        "<|endoftext|>",
        START + "system" + SEPARATOR + "user" + END + "answer",
        START + "system" + SEPARATOR + "user" + SEPARATOR + "duplicate" + END,
        START + "<|im_start|>assistant\n" + SEPARATOR + "user" + END,
        (START + "system" + SEPARATOR + "user" + END).replace("\n", "\r\n"),
        "<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n",
    ],
)
def test_malformed_or_ambiguous_roles_are_rejected(prompt: str) -> None:
    """The mapper never guesses a privilege boundary from partial or nested control sequences."""
    with pytest.raises(ValueError):
        chatml_roles(prompt, "")


def test_mapping_input_bounds() -> None:
    """Bounds count prompt UTF-8 bytes and reject instruction control tokens before HTTP."""
    with pytest.raises(ValueError, match="bounded"):
        chatml_roles("é" * 65537, "")
    with pytest.raises(ValueError, match="bounded"):
        chatml_roles("plain", "x" * 16385)
    with pytest.raises(ValueError, match="control"):
        chatml_roles("plain", "<|im_start|>system")


@pytest.mark.parametrize(
    "chat,config_hash,mapping_hash",
    [
        (
            False,
            "c14a41eacf9853bc1512314c39f24c70225aa179da767015bfe4077ddeee4dab",
            "92bc78199bb4ae70b22c9edb46ce4096d063e31555de20b558b937962cd0ae3c",
        ),
        (
            True,
            "0921a2ecd3ba4b6c8fa1eed0f1ea71a41393b9b042c2999d02c8f6fe24cf9a91",
            "2eb0da3d5bcf47a09fbda66f0c98a2f3d4ffcf2a04a1812971dbad812ee4995c",
        ),
    ],
)
def test_prechange_config_and_mapping_digests(
    chat: bool, config_hash: str, mapping_hash: str
) -> None:
    """Hashes captured before implementation protect historical completion and native-chat bytes."""
    config = (
        RunConfig(request_api="chat", system_prompt="format", chat_template_sha256="a" * 64)
        if chat
        else RunConfig()
    )
    assert hashlib.sha256(config.model_dump_json().encode()).hexdigest() == config_hash
    assert config.request_mapping_digest() == mapping_hash
    assert RunConfig.model_validate_json(config.model_dump_json()) == config
