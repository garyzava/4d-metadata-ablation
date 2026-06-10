"""Read-only Postgres connector using SQLAlchemy."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_connection_url() -> str:
    """Return the DATABASE_URL from the environment."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return url


def create_readonly_engine(url: str | None = None) -> Engine:
    """Create a read-only SQLAlchemy engine.

    Sets the default transaction to READ ONLY so no writes can leak through.
    """
    url = url or get_connection_url()
    engine = create_engine(
        url,
        execution_options={"postgresql_readonly": True},
        pool_pre_ping=True,
    )
    return engine


@contextmanager
def readonly_connection(engine: Engine) -> Iterator[Any]:
    """Yield a read-only connection with statement timeout."""
    with engine.connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
        conn.execute(text("SET statement_timeout = '30s'"))
        yield conn
