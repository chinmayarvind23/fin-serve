# Results

This file is generated from benchmarks.

| Metric                         | Evidence                                    |
| ------------------------------ | ------------------------------------------- |
| 38 -> 94 requests/s            | baseline/optimized manifests + raw requests |
| 2.4x token throughput          | generated tokens + windows                  |
| 690 -> 295 ms median TTFT      | raw request timing                          |
| 3.8 -> 1.9 s p95               | raw request timing                          |
| 37% lower GPU cost / 1M tokens | pricing + GPU hours + token counts          |
| 99.2% quality parity           | frozen eval suite/results                   |
| 99.95% successful requests     | load failure taxonomy                       |
| 81% GPU utilization            | raw GPU telemetry + aggregation             |
| rollback <= 94 s               | induced regression event                    |
