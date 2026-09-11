"""Collect real quality responses with the same frozen wire mapper used for performance."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Literal, Self

import httpx
from pydantic import Field, model_validator

from finserve.benchmark.metrics import RequestRecord
from finserve.benchmark.runner import prepare_output, request_one, request_payload, write_json
from finserve.benchmark.workload import WorkItem
from finserve.contracts.deployment import ImmutableModel
from finserve.contracts.producer import QualityCollectionSpec
from finserve.evaluation.quality import unique_object


class QualityCollectionResult(ImmutableModel):
    """Missing outputs are failed cases; a result never contains a caller-supplied passed flag."""

    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requests_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["completed", "interrupted", "failed"]
    planned: int = Field(ge=1, le=1024)
    recorded: int = Field(ge=0, le=1024)
    successful: int = Field(ge=0, le=1024)
    outputs: dict[str, str]

    @model_validator(mode="after")
    def counts_match(self) -> Self:
        """Incomplete or failed requests cannot disappear into an apparently complete output set."""
        if not 0 <= self.successful <= self.recorded <= self.planned:
            raise ValueError("quality collection counts disagree")
        if len(self.outputs) != self.successful:
            raise ValueError("quality output count differs from successful requests")
        if self.status == "completed" and self.recorded != self.planned:
            raise ValueError("completed quality collection is missing requests")
        return self


async def collect_quality(
    client: httpx.AsyncClient, specification: QualityCollectionSpec, output: Path
) -> QualityCollectionResult:
    """Record each response before inspecting it; cancellation preserves partial raw evidence."""
    specification = QualityCollectionSpec.model_validate_json(specification.model_dump_json())
    output = prepare_output(output)
    (output / "specification.json").write_text(specification.canonical(), encoding="utf-8")
    write_json(
        output / "status.json", {"status": "running", "specification": specification.digest()}
    )
    outputs: dict[str, str] = {}
    count = 0
    status: Literal["completed", "interrupted", "failed"] = "failed"
    raw_digest = hashlib.sha256()
    try:
        with (output / "requests.jsonl").open("xb") as raw:
            for index, case in enumerate(specification.suite.cases):
                item = WorkItem(
                    case_id=case.case_id,
                    family=case.family,
                    prompt=case.prompt,
                    max_tokens=specification.max_tokens,
                )
                row = await request_one(
                    client,
                    specification.endpoint(),
                    item,
                    index,
                    time.perf_counter(),
                    specification.configuration,
                    "quality",
                )
                encoded = (
                    json.dumps(
                        {
                            "request": request_payload(item, specification.configuration),
                            "response": row.model_dump(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                ).encode()
                raw.write(encoded)
                raw.flush()
                raw_digest.update(encoded)
                count += 1
                if row.success:
                    outputs[case.case_id] = row.output
                # Retain the crossing record, then stop offering work. Overshoot is bounded
                # by one recorder event/output limit; an oversized collection cannot qualify.
                if raw.tell() > specification.maximum_raw_bytes:
                    raise ValueError("quality collection exceeded raw evidence byte budget")
                if row.error == "CancelledError":
                    raise asyncio.CancelledError
        status = "completed"
    except asyncio.CancelledError:
        status = "interrupted"
        raise
    finally:
        result = QualityCollectionResult(
            specification_sha256=specification.digest(),
            suite_hash=specification.suite.digest(),
            request_mapping_sha256=specification.configuration.request_mapping_digest(),
            requests_sha256=raw_digest.hexdigest(),
            status=status,
            planned=len(specification.suite.cases),
            recorded=count,
            successful=len(outputs),
            outputs=outputs,
        )
        write_json(output / "result.json", result.model_dump())
        write_json(
            output / "status.json", {"status": status, "specification": specification.digest()}
        )
    return result


def load_quality_collection(
    directory: Path,
) -> tuple[QualityCollectionSpec, QualityCollectionResult]:
    """Recompute completed outputs from raw records before a gate or registry consumes them."""
    specification = QualityCollectionSpec.model_validate_json(
        bounded_file(directory / "specification.json", 2 * 1024**2)
    )
    result = QualityCollectionResult.model_validate_json(
        bounded_file(directory / "result.json", specification.maximum_raw_bytes)
    )
    if (directory / "requests.jsonl").stat().st_size > specification.maximum_raw_bytes:
        raise ValueError("quality raw evidence exceeds declared byte budget")
    if (
        result.status != "completed"
        or result.specification_sha256 != specification.digest()
        or result.suite_hash != specification.suite.digest()
        or result.request_mapping_sha256 != specification.configuration.request_mapping_digest()
    ):
        raise ValueError("quality collection identity or completion mismatch")
    digest = hashlib.sha256()
    outputs: dict[str, str] = {}
    count = 0
    with (directory / "requests.jsonl").open("rb") as raw:
        for count, case in enumerate(specification.suite.cases, 1):
            encoded = raw.readline(32 * 1024**2)
            if not encoded.endswith(b"\n"):
                raise ValueError("missing or oversized quality request record")
            digest.update(encoded)
            document = json.loads(encoded, object_pairs_hook=unique_object)
            row = RequestRecord.model_validate(document["response"])
            item = WorkItem(
                case_id=case.case_id,
                family=case.family,
                prompt=case.prompt,
                max_tokens=specification.max_tokens,
            )
            if (
                document["request"] != request_payload(item, specification.configuration)
                or row.logical_id != count - 1
                or row.case_id != case.case_id
                or row.family != case.family
                or row.phase != "quality"
                or not row.offered
                or row.send_s is None
                or (row.output and row.first_content_s is None)
                or (row.generated_tokens is not None and row.generated_tokens > item.max_tokens)
                or (row.success and row.status_code != 200)
                or (row.success and row.generated_tokens and row.first_content_s is None)
            ):
                raise ValueError("quality raw request differs from frozen input")
            if row.success:
                outputs[row.case_id] = row.output
        if raw.read(1):
            raise ValueError("extra quality request records")
    if (
        result.requests_sha256 != digest.hexdigest()
        or result.recorded != count
        or result.planned != len(specification.suite.cases)
        or result.outputs != outputs
    ):
        raise ValueError("quality collection result differs from raw evidence")
    return specification, result


def bounded_file(path: Path, maximum_bytes: int) -> bytes:
    """Bound small manifest/result reads before JSON parsing and retained-output reconstruction."""
    with path.open("rb") as source:
        data = source.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError("quality evidence file exceeds byte budget")
    return data
