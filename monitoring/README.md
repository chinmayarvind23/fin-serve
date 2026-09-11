# Text gateway monitoring

The optional Compose monitoring override provisions Prometheus, a Grafana datasource and the **FinServe ? Text gateway operations** dashboard. The dashboard is stored in `dashboards/gateway.json`; provisioning uses the stable datasource UID `finserve-prometheus`. Prometheus scrapes every five seconds and evaluates four alert rules. Both images are pinned by digest.

Follow the [container guide](../infra/docker/README.md) to start the override with a private Grafana password. Default ports are loopback-only: Prometheus `9097`, Grafana `3007`. Sign in to Grafana as `admin` with the configured private password. The dashboard appears in the FinServe folder. The local stack is ephemeral; it does not configure backups or a production monitoring retention service. Prometheus retention is bounded to 24 hours and 256 MB.

## Metric populations

These metrics cover the text gateway. Vision requests, visual jobs and the explorer use separate paths and are not represented. Authentication, invalid-body and model-rejection responses are not counted text outcomes. Quota/admission rejection and cancellation have explicit outcome labels where the gateway records them.

The duration histogram covers requests whose text stream iteration started, including nonstream response collection. It excludes quota/admission rejection and cancellation before stream iteration. TTFT covers observed visible content. Histogram percentiles are approximate and differ from exact benchmark percentiles and populations. Reported generated-token totals may include failed streams when their engine reports usage.

Current stat panels use instant queries, mask non-up targets and display No data for missing samples. Time series preserve gaps. No request ID, prompt or credential is a metric label. This dashboard does not infer physical GPU utilization, model accuracy, billed cost or production availability.

## Alert defaults

| Rule | Condition | Sustained interval |
| --- | --- | --- |
| Gateway scrape unavailable | Configured target has `up=0` | 30 seconds |
| Engine failure ratio | Counted engine failure, timeout or quota-store unavailability exceeds 5% | 1 minute |
| Admission rejected | Overload outcomes exceed 10% of counted requests | 1 minute |
| High TTFT p95 | Approximate server p95 exceeds 1 second | 2 minutes |

Ratio and latency alerts require at least 20 relevant observations in the two-minute window. They evaluate per instance, so a busy healthy gateway cannot hide a failing one. These are operational defaults, not frozen release thresholds. Alertmanager and notification receivers are not configured; no email or chat notification is sent.

`alerts.test.yml` tests pending/firing/recovery, instance isolation, sparse traffic, distinct admission/quota outcomes and sustained histogram latency. The Quality workflow runs pinned `promtool check config` and `promtool test rules` with network access disabled inside each test container.

## Verified local integration

A separate owned local stack ran a gateway image built from exact source `31fc8a7fd2e06d42dbf3260830f9daca00f269b3`. It completed 32 fixture requests and exposed 160 generated character tokens. Prometheus scraped those counters; Grafana loaded the nine-panel dashboard and queried its provisioned datasource through its HTTP API. An actual gateway stop reached the scrape alert's firing state, masked current stats returned no data, and restart restored the scrape and cleared the alert. All three test containers stopped with exit code 0. Evidence is retained privately under `monitoring-live-01`.

This proves local provisioning, query and alert behavior. Browser rendering, notification delivery and hosted operation are separate checks. Local Langfuse ingestion has its own [verified stack and limitations](../infra/docker/langfuse.md). CloudWatch remains unconfigured.

The configuration follows [Prometheus alert rules](https://prometheus.io/docs/prometheus/3.5/configuration/alerting_rules/), [Promtool rule testing](https://prometheus.io/docs/prometheus/3.5/configuration/unit_testing_rules/) and [Grafana file provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/).
