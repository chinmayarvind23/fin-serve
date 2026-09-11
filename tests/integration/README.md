# Cross-module verification

This suite exercises actual application composition, SQL/CAS persistence, producer stages, lifecycle decisions, stream cancellation and optional runtime adapters. Some tests use real local libraries or services; others inject HTTP/Docker fixtures. Each test's scope identifies that boundary.

Run `uv run --no-sync pytest tests/integration` with the dependencies in [the command guide](../../docs/commands.md). Optional Ray/Airflow and hardware-specific executions have separate environments and retained logs. A fixture lifecycle test does not establish deployed AWS health or pretrained model quality.
