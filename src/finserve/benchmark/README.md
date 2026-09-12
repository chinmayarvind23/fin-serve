# Measurement and evidence

`runner.py` freezes inputs and records each offered HTTP request, including partial output and failure. `metrics.py` defines populations and percentiles; `experiment.py` combines the request run with source/environment and physical GPU observations. `gpu.py` integrates bounded sample-held observations with coverage, while `cost.py` requires explicit price and billed/modelled time.

The routing modules use a separate frozen mechanical workload and owned process/session controls. Their populations do not replace the text release workload. Native image identity stays undeclared when it was not observed. New runs require fresh output directories outside the repository; failed attempts remain retained.

Use [measurement commands](../../../docs/commands.md), [metric definitions](../../../docs/LLD.md) and [Guide](../../../docs/quality-gates.md). Successful HTTP completion, semantic correctness and exact output agreement are separate measurements.
