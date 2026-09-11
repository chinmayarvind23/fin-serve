"""Real Linux socket lifecycle checks for local experiment restart preflight."""

import socket
import sys

import pytest

from finserve.benchmark.twin_control import require_available_port

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux engine process boundary")


def test_live_listener_is_never_reused() -> None:
    """An unrelated listener still blocks startup even if it enabled address reuse."""
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(OSError):
            require_available_port(listener.getsockname()[1])


def test_time_wait_does_not_block_restart_preflight() -> None:
    """Reproduce the original failed bind, then allow a closed server's retained TCP state."""
    with socket.socket() as listener, socket.socket() as client:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        client.connect(("127.0.0.1", port))
        connection, _ = listener.accept()
        connection.close()
        assert client.recv(1) == b""
    with socket.socket() as original_probe, pytest.raises(OSError):
        original_probe.bind(("127.0.0.1", port))
    require_available_port(port)
