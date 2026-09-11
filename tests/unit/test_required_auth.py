"""An explicit deployment mode fails before creating unauthenticated serving resources."""

import pytest

from finserve.auth import require_credentials
from finserve.engines.ray_backends import OpenAIEngineReplica, RoutedBackends, build_application
from finserve.gateway.app import from_env


@pytest.mark.parametrize(
    "value", ["", "short", "x" * 4097, "x" * 16 + "\n", "x" * 16 + " ", "é" * 32]
)
def test_required_credential_bounds(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """An existing Secret is insufficient if its value cannot provide the declared auth boundary."""
    monkeypatch.setenv("FINSERVE_REQUIRE_AUTH", "1")
    monkeypatch.setenv("FINSERVE_API_KEY", value)
    with pytest.raises(ValueError, match="FINSERVE_API_KEY"):
        require_credentials("FINSERVE_API_KEY")


def test_required_auth_is_explicit_and_validates_the_entire_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixtures can omit credentials, while a production configuration needs every selected key."""
    monkeypatch.delenv("FINSERVE_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("FINSERVE_API_KEY", raising=False)
    monkeypatch.delenv("FINSERVE_RAY_API_KEY", raising=False)
    require_credentials("FINSERVE_API_KEY")
    monkeypatch.setenv("FINSERVE_REQUIRE_AUTH", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        require_credentials("FINSERVE_API_KEY")
    monkeypatch.setenv("FINSERVE_REQUIRE_AUTH", "1")
    monkeypatch.setenv("FINSERVE_API_KEY", "fixture-ingress-credential")
    with pytest.raises(ValueError, match="FINSERVE_RAY_API_KEY"):
        require_credentials("FINSERVE_API_KEY", "FINSERVE_RAY_API_KEY")
    monkeypatch.setenv("FINSERVE_RAY_API_KEY", "fixture-private-hop-credential")
    require_credentials("FINSERVE_API_KEY", "FINSERVE_RAY_API_KEY")


@pytest.mark.parametrize(
    "backend, missing",
    [
        ("fixture", "FINSERVE_API_KEY"),
        ("ray-http", "FINSERVE_RAY_API_KEY"),
        ("vllm", "FINSERVE_ENGINE_API_KEY"),
        ("sglang", "FINSERVE_ENGINE_API_KEY"),
    ],
)
def test_gateway_auth_precedes_client_creation(
    monkeypatch: pytest.MonkeyPatch, backend: str, missing: str
) -> None:
    """Missing credentials fail before endpoint lookup, pooled clients or exporter allocation."""
    monkeypatch.setenv("FINSERVE_REQUIRE_AUTH", "1")
    monkeypatch.setenv("FINSERVE_ENGINE", backend)
    monkeypatch.setenv("FINSERVE_API_KEY", "fixture-ingress-credential")
    monkeypatch.delenv(missing, raising=False)
    monkeypatch.delenv("FINSERVE_ENGINE_URL", raising=False)
    with pytest.raises(ValueError, match=missing):
        from_env()


def test_ray_auth_precedes_actor_or_client_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each actor checks its own environment, so a valid head cannot hide an empty worker Secret."""
    monkeypatch.setenv("FINSERVE_REQUIRE_AUTH", "1")
    monkeypatch.delenv("FINSERVE_ENGINE_API_KEY", raising=False)
    monkeypatch.delenv("FINSERVE_RAY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FINSERVE_ENGINE_API_KEY"):
        build_application({})
    with pytest.raises(ValueError, match="FINSERVE_ENGINE_API_KEY"):
        OpenAIEngineReplica("fixture", 1, "fixture", "invalid-url")
    with pytest.raises(ValueError, match="FINSERVE_RAY_API_KEY"):
        RoutedBackends({}, "fixture", "least_load")
