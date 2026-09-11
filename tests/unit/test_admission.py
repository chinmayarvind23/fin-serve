"""Small deterministic admission invariants merit direct boundary tests."""

import pytest

from finserve.gateway.admission import Admission


def test_capacity_and_underflow() -> None:
    """Overload cannot increase active work, and double release must be visible."""
    admission = Admission(1)
    assert admission.acquire()
    assert not admission.acquire()
    admission.release()
    with pytest.raises(RuntimeError):
        admission.release()
    with pytest.raises(ValueError):
        Admission(0)
