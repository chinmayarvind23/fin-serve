"""Operator CLI rejects partial inputs and closes its registry on lookup failure."""

from pathlib import Path

import pytest

from finserve.registry.metadata import Registry
from finserve.registry.mlflow_cli import run_cli


def test_required_options() -> None:
    """An incomplete invocation cannot create an implicit tracking destination."""
    with pytest.raises(SystemExit) as error:
        run_cli([])
    assert error.value.code == 2


def test_registry_closed_when_decision_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed persisted lookup closes storage and prints no fabricated tracking identity."""
    closed: list[Registry] = []
    original = Registry.close

    def close(registry: Registry) -> None:
        """Retain ownership evidence while actually disposing the database resources."""
        closed.append(registry)
        original(registry)

    monkeypatch.setattr(Registry, "close", close)
    with pytest.raises(KeyError, match="registry record not found"):
        run_cli(
            [
                "--registry-url",
                "sqlite:///" + str(tmp_path / "registry.db"),
                "--artifact-root",
                str(tmp_path / "objects"),
                "--run-id",
                "missing",
                "--decision-digest",
                "0" * 64,
                "--tracking-uri",
                "sqlite:///" + str(tmp_path / "tracking.db"),
                "--experiment-id",
                "1",
            ]
        )
    assert len(closed) == 1
    assert capsys.readouterr().out == ""
