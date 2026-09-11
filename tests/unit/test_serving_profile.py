"""Canonical configuration must survive task handoffs without endpoint-only reinterpretation."""

import pytest

from finserve.contracts.deployment import Revision
from finserve.contracts.serving_profile import ServingProfileV1
from finserve.reliability.warm_routes import BackendConfiguration, WarmBackend


def profile(endpoint: str = "http://127.0.0.1:9000/v1", parameters: str = "{}") -> ServingProfileV1:
    """Synthetic identities test contracts without claiming model downloads or image builds."""
    return ServingProfileV1(
        engine="fixture",
        engine_version="0.0.0",
        engine_parameters_json=parameters,
        model_revision="a" * 40,
        tokenizer_revision="b" * 40,
        model_manifest_sha256="c" * 64,
        tokenizer_manifest_sha256="d" * 64,
        base_url=endpoint,
        served_model="reference",
    )


def revision(value: ServingProfileV1, name: str = "candidate") -> Revision:
    """The image remains a separate identity; configuration points at the full serving profile."""
    return Revision(
        revision_id=name,
        model_revision=value.model_revision,
        tokenizer_revision=value.tokenizer_revision,
        source_revision="e" * 40,
        image_digest="sha256:" + "f" * 64,
        config_digest=value.digest(),
        engine=value.engine,
        engine_config=value.engine_parameters_json,
    )


def test_profile_canonicalization_and_warm_binding() -> None:
    """Ordering does not change identity; endpoint or parameter changes do."""
    first = profile(parameters='{"max_sequences":8,"dtype":"float16"}')
    reordered = profile(parameters='{"dtype":"float16", "max_sequences":8}')
    assert first.digest() == reordered.digest()
    assert ServingProfileV1.model_validate_json(first.model_dump_json()) == first
    configuration = BackendConfiguration(base_url=first.base_url, model=first.served_model)
    assert WarmBackend(revision=revision(first), configuration=configuration, serving_profile=first)
    for changed in (
        profile("http://127.0.0.1:9001/v1", first.engine_parameters_json),
        profile(parameters='{"max_sequences":4}'),
    ):
        with pytest.raises(ValueError):
            changed.verify_revision(revision(first))
    with pytest.raises(ValueError):
        WarmBackend(
            revision=revision(first),
            configuration=configuration.model_copy(update={"model": "other"}),
            serving_profile=first,
        )


@pytest.mark.parametrize(
    "parameters",
    [
        '{"x":1,"x":2}',
        '{"nested":{"x":1,"x":2}}',
        '{"x":NaN}',
        "[1,2]",
        '{"api_key":"secret"}',
        '{"nested":{"password":"secret"}}',
        '{"x":Infinity}',
        '{"x":"' + "é" * 12000 + '"}',
    ],
)
def test_invalid_or_secret_parameter_configuration_is_rejected(parameters: str) -> None:
    """Duplicate keys, nonfinite values and literal credentials cannot enter canonical identity."""
    with pytest.raises(ValueError):
        profile(parameters=parameters)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://user:secret@host/v1",
        "http://host/v1?key=secret",
        "http://host/v1/",
        "file:///v1",
    ],
)
def test_profile_endpoint_requires_fixed_server_owned_url(endpoint: str) -> None:
    """The shared profile cannot smuggle destination overrides or serialized credentials."""
    with pytest.raises(ValueError):
        profile(endpoint)
