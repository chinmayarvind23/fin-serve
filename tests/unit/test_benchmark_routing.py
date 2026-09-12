"""Observed attribution must survive request failure without fabricating an executing replica."""

import json
import time

import httpx
import pytest

from finserve.benchmark.metrics import RequestRecord
from finserve.benchmark.routing import observed_routing
from finserve.benchmark.runner import RunConfig, request_one
from finserve.benchmark.workload import WorkItem

PHYSICAL = {
    "x-finserve-revision": "extra-1",
    "x-finserve-revision-digest": "b" * 64,
    "x-finserve-route-generation": "3",
}
ANCHOR = {
    "x-finserve-anchor-revision": "primary",
    "x-finserve-anchor-digest": "a" * 64,
    "x-finserve-anchor-generation": "3",
}


async def collect(status: int, headers: dict[str, str], body: str) -> RequestRecord:
    """Exercise the ordinary collector, including its offered-failure and timeout accounting."""

    def respond(request: httpx.Request) -> httpx.Response:
        """The transport owns only this explicit response; it does not replace collector parsing."""
        return httpx.Response(status, headers=headers, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        return await request_one(
            client,
            "http://localhost/v1/completions",
            WorkItem(case_id="routing", prompt="hello", max_tokens=1),
            0,
            time.perf_counter(),
            RunConfig(requests=1),
            "measured",
        )


async def test_success_and_later_failure_keep_actual_physical_binding() -> None:
    """The selected extra's identity remains distinct from its approved logical anchor."""
    headers = {**PHYSICAL, **ANCHOR, "x-finserve-pool-generation": "7"}
    body = (
        'data: {"choices":[{"text":"ok","finish_reason":"stop"}],'
        '"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
    )
    row = await collect(200, headers, body)
    assert row.success and row.routing is not None
    assert row.routing.physical is not None and row.routing.physical.revision_id == "extra-1"
    assert row.routing.anchor is not None and row.routing.anchor.revision_id == "primary"
    assert row.routing.pool_generation == 7
    failed = await collect(200, headers, body.split("data: [DONE]")[0])
    assert not failed.success and failed.output == "ok" and failed.routing == row.routing
    assert RequestRecord.model_validate_json(row.model_dump_json()) == row


async def test_predispatch_failure_keeps_anchor_without_physical_claim() -> None:
    """An unauthorized request names only the route observed before any backend dispatch."""
    row = await collect(401, ANCHOR, "denied")
    assert not row.success and row.status_code == 401 and row.routing is not None
    assert row.routing.physical is None and row.routing.anchor is not None
    invalid = await collect(200, ANCHOR, "")
    assert not invalid.success and invalid.error == "ValueError"


async def test_legacy_absence_preserves_original_serialized_keys() -> None:
    """Reconstructing older direct-engine evidence must not insert a new null field."""
    row = await collect(503, {}, "unavailable")
    assert row.routing is None and "routing" not in row.model_dump()
    old = json.loads(row.model_dump_json())
    assert RequestRecord.model_validate(old).model_dump() == old
    legacy = observed_routing(httpx.Response(200, headers=PHYSICAL))
    assert legacy is not None and legacy.physical is not None and legacy.anchor is None


@pytest.mark.parametrize(
    "headers",
    [
        {"x-finserve-revision": "partial"},
        {**PHYSICAL, "x-finserve-route-generation": "03"},
        {**PHYSICAL, "x-finserve-pool-generation": "1"},
        {**PHYSICAL, **ANCHOR},
        {**PHYSICAL, "x-finserve-revision-digest": "z" * 64},
    ],
)
async def test_incomplete_or_invalid_claim_retains_failed_request(headers: dict[str, str]) -> None:
    """Invalid attribution cannot become a successful benchmark row or disappear from load."""
    row = await collect(200, headers, "")
    assert (
        row.offered
        and not row.success
        and row.status_code == 200
        and row.error in {"ValueError", "ValidationError"}
    )


def test_duplicate_header_is_not_silently_selected() -> None:
    """A proxy combining contradictory upstream identities cannot choose one by accident."""
    headers = [*PHYSICAL.items(), ("x-finserve-revision", "other")]
    with pytest.raises(ValueError, match="ambiguous"):
        observed_routing(httpx.Response(200, headers=headers))
