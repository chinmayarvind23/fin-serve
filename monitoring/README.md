# Monitoring

`prometheus.yml` scrapes the gateway's isolated metric registry every five seconds.
`grafana-datasources.yaml` provisions that datasource with the optional Docker
monitoring override. See `infra/docker/README.md` for startup.

Metrics use bounded outcome labels and report active requests, generated tokens,
TTFT and duration. They exclude prompts and credentials. `/healthz` is liveness;
actual inference and exact revision checks remain separate promotion requirements.
Optional local JSON traces use bounded SDK buffering. Their filesystem exporter
does not provide a hard I/O completion deadline. CloudWatch and Langfuse are not
configured by these local files.
