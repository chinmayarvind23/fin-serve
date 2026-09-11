"""Reject immediately when full rather than building an unbounded waiting queue."""

from dataclasses import dataclass


@dataclass
class Admission:
    """Single-event-loop counter; use one limiter per process and budget replicas externally."""

    limit: int
    active: int = 0

    def __post_init__(self) -> None:
        """Invalid capacity is a startup error, not a silent permanent overload."""
        if self.limit < 1:
            raise ValueError("admission limit must be positive")

    def acquire(self) -> bool:
        """No await separates check and increment, making admission atomic on one loop."""
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        """An underflow indicates duplicate cleanup and must never be hidden."""
        if self.active < 1:
            raise RuntimeError("admission lease released twice")
        self.active -= 1
