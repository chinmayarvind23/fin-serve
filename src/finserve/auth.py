"""Explicit deployment authentication checks run before allocating serving resources."""

import os


def require_credentials(*names: str) -> None:
    """Production mode rejects empty Secrets; local fixtures retain unauthenticated mode."""
    setting = os.getenv("FINSERVE_REQUIRE_AUTH", "0")
    if setting not in {"0", "1"}:
        raise ValueError("FINSERVE_REQUIRE_AUTH must be 0 or 1")
    if setting == "0":
        return
    for name in names:
        value = os.getenv(name, "")
        if not 16 <= len(value) <= 4096 or any(not 33 <= ord(char) <= 126 for char in value):
            raise ValueError(f"A bounded nonempty deployment credential is required: {name}")
