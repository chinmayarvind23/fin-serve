"""Each app owns its registry, preventing duplicate collectors in tests and replicas."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    """Counters cover terminal outcomes, not just successful streams."""

    def __init__(self) -> None:
        """Fixed label sets avoid per-request time-series cardinality growth."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "finserve_requests", "Terminal requests", ["outcome"], registry=self.registry
        )
        self.active = Gauge("finserve_active", "Admitted active requests", registry=self.registry)
        self.tokens = Counter(
            "finserve_generated_tokens", "Engine generated tokens", registry=self.registry
        )
        self.ttft = Histogram(
            "finserve_server_ttft_seconds", "Receive to first content", registry=self.registry
        )
        self.duration = Histogram(
            "finserve_request_seconds", "Receive to terminal outcome", registry=self.registry
        )
