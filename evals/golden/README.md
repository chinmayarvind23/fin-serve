# Frozen quality inputs

`correctness-32-v1.json` contains the unchanged synthetic release suite with exact-string and typed-JSON targets. Its SHA256 is `998b1f5dd448c2dffe247bc6fe89b5251699fb1b3fc170dc0fb427662706b463`. The benchmark load workload is separate. Freeze the suite, grader and request mapping before collection; retain baseline failures and raw outputs.

`finance_workloads.yaml` contains earlier synthetic examples and a schema description. It is not the measured 32-case release suite and does not establish external finance-dataset coverage. [Guide](../../docs/quality-gates.md) defines grading and [Guide](../../docs/quality-gates.md) records the failed gates.
