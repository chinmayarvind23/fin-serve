"""Post-shutdown verification must not leave a malformed trace marked as a completed cohort."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from finserve.benchmark.runner import RunConfig


@pytest.mark.parametrize("raw", [b"malformed-json\n", b'{"unexpected":"shape"}\n'])
def test_bad_trace_marks_terminal_receipt_failed(tmp_path: Path, raw: bytes) -> None:
    """An actual malformed artifact exercises the failure path after the owned server stopped."""
    source = Path(__file__).resolve().parents[2] / "scripts/measure_trace_overhead.py"
    spec = importlib.util.spec_from_file_location("trace_overhead_review", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "traces.jsonl").write_bytes(raw)
    receipt: dict[str, Any] = {"status": "complete", "summary": {}}
    with pytest.raises((ValueError, KeyError)):
        module.verify_receipt(receipt, tmp_path, "full", RunConfig())
    recorded = json.loads((tmp_path / "process.json").read_bytes())
    assert recorded["status"] == "failed"
    assert recorded["validation_error"] in {"JSONDecodeError", "KeyError"}
