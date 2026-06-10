"""Database connectors with engine factory.

Trimmed for the metadata-ablation experiment: postgres only. The upstream
gyzasql also re-exports sqlite + file-converter; those modules were dropped
from this vendor (see src/gyzasql/_VENDORED.md).
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from gyzasql.db.connectors.postgres import create_readonly_engine

__all__ = [
    "create_engine_for_dataset",
    "create_readonly_engine",
]


def create_engine_for_dataset(connection_ref: str, db_type: str) -> Engine:
    """Create the appropriate SQLAlchemy engine based on dataset type."""
    if db_type == "postgres":
        return create_readonly_engine(connection_ref)
    raise ValueError(f"Unsupported db_type: {db_type!r} (this vendored gyzasql is postgres-only)")
