# Benchmark populations

FinServe uses original synthetic finance-shaped and general prompts to exercise the same serving path. These are small systems workloads, not a benchmark of financial expertise or investment advice. No public filings or proprietary reports were used in the recorded sustained comparison.

## Frozen sustained text workload

[`text-release-v1.json`](../benchmarks/configs/text-release-v1.json) contains 64 cases, with 16 in each family:

| Family | Task shape | What the serving measurement exercises |
| --- | --- | --- |
| `SEC_QA` | Compute a margin from stated revenue and income | Short numerical explanation with varying context |
| `FINANCIAL_TABLE_EXTRACTION` | Extract supplied company, revenue and capex fields | Short structured output |
| `EARNINGS_SUMMARY` | Summarize supplied synthetic business facts | Longer generated responses |
| `GENERAL` | Original nonfinancial instructions | A control family using the same engine and scheduler |

The canonical workload hash is `ec5b6d6b04fff5ad77d8ad3205e0c9c5e9f90dcc0aed5380792d7d4754789e54`. The native sustained runs each cycled through this population for 3,072 measured requests after 64 separate warmup requests. Each family therefore contributed 768 measured requests per run. Repetitions increase the load population; they do not create 3,072 independent task types.

The workload freezes prompt bytes, family, case identity and output budget. The original completion payload includes explicit Qwen chat delimiters. New chat-API experiments also bind the system prompt, request mapping and chat-template digest; changing that mapping creates a new comparison envelope. Historical completion results retain their original representation.

## Quality is a separate population

[`correctness-32-v1.json`](../evals/golden/correctness-32-v1.json) contains 32 original deterministic cases, eight per family. Its hash is `998b1f5dd448c2dffe247bc6fe89b5251699fb1b3fc170dc0fb427662706b463`. Exact answers and typed JSON expectations test task correctness independently of agreement between two serving configurations. The [evaluation guide](evaluation.md) specifies the grader and retained failures.

The three-case `default_suite()` is a development smoke. Its successful execution cannot replace the frozen 32-case release requirement. Likewise, comparing output strings from repeated load requests measures serving agreement, not truth.

## Separate routing and visual experiments

The frozen routing workload has 64 mechanical cases balanced across short/long context and 16/128-token output budgets, with declared latency SLOs. Four ordered cohorts compare least-load and adaptive routing across two actual engines sharing one physical GPU. This population tests routing behavior; it is not a new finance quality suite.

Image understanding uses separate pinned-model probes. The retained uniform-color local/HTTP normalization experiment had 36 successful responses and zero correct answers. Three counterfactual bar charts passed a separate functional probe. Neither population is included in the sustained text denominator, and neither establishes general chart-reasoning quality.

## Reporting and provenance

Report the overall population and its actual workload slices. Do not add a `CHART_REASONING` slice to text-only runs. Failed requests stay in the denominator; failed or rejected experiments remain available. Raw evidence lives outside the checkout, while workload definitions, deterministic graders and measurement code are versioned here.

Scheduling depends on configured model/GPU compatibility, current load and declared affinity. A finance family label does not route to a special finance server. See [methodology](benchmark-methodology.md) and [recorded results](results.md) for timing definitions, artifact identities and limitations.
