# Finance-Shaped Benchmark Suite

## Purpose

FinServe remains a general-purpose inference and ML-systems platform.

Finance is included as one representative workload family because it creates useful serving stressors:

- long documents,
- structured and numerical outputs,
- repeated report templates,
- multimodal chart interpretation,
- quality requirements that make silent degradation unacceptable.

The serving architecture is not specialized for finance. The same scheduler, engine, deployment, and observability paths must also work for non-financial workloads.

## Workload classes

### `SEC_QA`

Question answering over public filing excerpts or synthetic filing-like documents.

```text
long prompt
short/medium output
prefill-heavy
```

Serving questions:

- Does long prefill dominate TTFT?
- Does prefix caching help repeated filing templates?
- Does long-document traffic hurt interactive short requests?

Quality checks:

- answer correctness,
- evidence support,
- structured-output validity where applicable.

### `EARNINGS_SUMMARY`

Summarization over public or synthetic earnings-call-like transcripts.

```text
medium/long prompt
long output
decode-heavy after prefill
```

Serving questions:

- Does speculative decoding improve inter-token latency?
- How does long generation affect continuous batching?
- Which concurrency level maximizes throughput while respecting p95?

Quality checks:

- key-point coverage,
- unsupported-claim rate,
- output-length adherence.

### `FINANCIAL_TABLE_EXTRACTION`

Structured extraction from financial tables represented as text, HTML, or document input.

```text
medium input
short structured output
schema-constrained generation
```

Serving questions:

- Does an optimized or quantized configuration preserve exact fields?
- Is schema validity stable across engines?
- How much request throughput is available for short generations?

Quality checks:

- exact field accuracy,
- numeric accuracy,
- JSON/schema validity.

### `CHART_REASONING`

Multimodal reasoning over public or synthetic financial charts.

```text
image + text
vision encoding
short/medium output
heterogeneous multimodal stages
```

Serving questions:

- What share of latency comes from vision encoding versus decode?
- Does resolution create GPU-memory or compilation pressure?
- Does modality-aware routing improve utilization?
- Is stage disaggregation worth the transfer overhead?

Quality checks:

- chart QA accuracy,
- numeric extraction accuracy,
- visual grounding.

## Systems stress matrix

| Workload                       | Modality      | Input       | Output           | Main serving stress                      |
| ------------------------------ | ------------- | ----------- | ---------------- | ---------------------------------------- |
| `SEC_QA`                     | text          | long        | short/medium     | prefill and prefix/cache behavior        |
| `EARNINGS_SUMMARY`           | text          | medium/long | long             | decode, speculation, continuous batching |
| `FINANCIAL_TABLE_EXTRACTION` | text/document | medium      | short structured | TTFT and exact output fidelity           |
| `CHART_REASONING`            | image + text  | multimodal  | short/medium     | vision stage, routing, GPU memory        |

## Data policy

Use public-domain, appropriately licensed, or synthetic inputs. Preserve provenance, license information, hashes, and dataset versions.

Do not commit proprietary analyst reports, private financial records, credentials, or non-redistributable data.

## Architecture isolation

The finance label exists for evaluation and reporting, not for a special execution path.

Bad:

```text
if domain == finance:
    route to a hardcoded finance server
```

Good:

```text
modality
input length
expected output length
SLO class
cache affinity
replica load
GPU capacity
    |
    v
generic scheduler and engine policy
```

## Reporting rule

Version reports show both overall performance and slices:

```text
overall
finance-shaped aggregate
SEC_QA
EARNINGS_SUMMARY
FINANCIAL_TABLE_EXTRACTION
CHART_REASONING
non-financial workload families
```

A strong result on the finance slice cannot substitute for the declared platform-wide result.
