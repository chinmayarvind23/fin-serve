# Security regression locations

Security checks run beside their boundary tests in `tests/contract`, `tests/unit` and `tests/integration`. They cover authentication before body receive, actual byte limits, query/response budgets, cancellation ownership, artifact identity, path aliases and required production credentials. This directory does not contain a separate executable suite.

Use the full [verification commands](../../docs/commands.md). [Security boundaries](../../docs/security.md) distinguishes tested application behavior from unverified cluster enforcement, TLS and cloud identity configuration.
