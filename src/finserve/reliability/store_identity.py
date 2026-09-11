"""Persistent local database identity prevents retrying against silently recreated state."""

import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4


def initialize_identity(connection: sqlite3.Connection) -> str:
    """Only explicit store initialization may create an identity; ordinary opens never do."""
    connection.execute(
        "CREATE TABLE IF NOT EXISTS store_identity("
        "singleton INTEGER PRIMARY KEY CHECK(singleton=1),"
        "identity TEXT NOT NULL)"
    )
    connection.execute("INSERT OR IGNORE INTO store_identity VALUES(1,?)", (uuid4().hex,))
    return read_identity(connection)


def read_identity(connection: sqlite3.Connection) -> str:
    """A missing identity is lost or unmigrated state, not permission to initialize it."""
    row: tuple[str] | None = connection.execute(
        "SELECT identity FROM store_identity WHERE singleton=1"
    ).fetchone()
    if row is None or len(row[0]) != 32 or any(char not in "0123456789abcdef" for char in row[0]):
        raise ValueError("invalid persistent store identity")
    return row[0]


def open_existing(path: Path) -> sqlite3.Connection:
    """SQLite mode=rw prevents a check/open race from creating an empty replacement database."""
    return sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=5)


def existing_identity(path: Path) -> str:
    """Read the identity of an existing database without migrating or creating any state."""
    with closing(open_existing(path)) as connection:
        return read_identity(connection)


def verify_identity(connection: sqlite3.Connection, expected: str) -> None:
    """Call inside the write transaction before examining or modifying protected state."""
    if read_identity(connection) != expected:
        raise ValueError("persistent store identity changed")
