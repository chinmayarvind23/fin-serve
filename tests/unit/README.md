# Deterministic core checks

These tests exercise routing decisions, lease ownership, request/metric contracts, quality grading, immutable identities, producer journals, protocol parsing and failure handling. Subprocess, Docker and HTTP fixtures check orchestration branches without claiming that a real engine or cloud service ran.

Run `uv run --no-sync pytest tests/unit` after installing the development dependencies and required optional runtimes from [the command guide](../../docs/commands.md). CI adds strict typing, coverage and semantic mutation checks. GPU performance and real-browser behavior require separate evidence.
