# Runbook: Engine failure or GPU OOM

A gateway `engine_failed` outcome does not establish GPU OOM. Preserve the failed stream and inspect the actual owned engine log, process exit and GPU observation before classifying the failure. Quota-store unavailability, protocol errors and network loss have different remedies.

1. Identify the exact engine process/container, model revision and configuration. Check native running/waiting requests, KV occupancy, physical free memory and whether multiple engines share that device.
2. Stop new routing to an unhealthy endpoint. Preserve existing ownership until cancellation or retirement is acknowledged; do not free a lease on an ambiguous remote timeout.
3. Check the failed request's context/output limits and image dimensions against the configured model and memory envelope. Retain attempted requests, including partial outputs.
4. For a confirmed OOM, drain or stop only the owned runtime, then restart a known-good configuration or move to independently available capacity. A shorter context, lower concurrency, new precision or different memory fraction is a new experiment configuration.
5. Verify exact-model readiness and real inference through the traffic route before re-enabling the endpoint. Record restart and recovery times separately from a warm route switch.

Do not alter the frozen workload or remove OOM records from a benchmark denominator. The local controller uses PID descriptors to avoid signalling unrelated processes. Cloud node replacement and automatic OOM recovery remain separate deployment checks.
