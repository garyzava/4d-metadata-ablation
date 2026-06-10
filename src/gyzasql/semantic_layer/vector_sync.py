"""SQLite write helpers for semantic context entries.

Historically these were dual-write helpers (SQLite + a Qdrant vector index).
The Qdrant indexing path was removed in this vendored copy: it was disabled at
runtime for every run (``GYZASQL_QDRANT_PATH`` was never set) and depended on RAG
modules already dropped from the vendor. SQLite (``MetadataStore``) is the sole
source of truth; these functions remain the write path that
``semantic_layer.context.populate_*`` delegates to for the L1+ ablation
dimensions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gyzasql.semantic_layer.store import MetadataStore


# ── Dataset-level upserts ────────────────────────────────────


def sync_upsert_business_term(
    store: MetadataStore,
    dataset_id: int,
    term: str,
    definition: str = "",
    synonyms: list[str] | None = None,
) -> None:
    """Upsert a business term in the SQLite metadata store."""
    store.upsert_business_term(dataset_id, term, definition, synonyms)


def sync_upsert_domain_knowledge(
    store: MetadataStore,
    dataset_id: int,
    topic: str,
    content: str,
) -> None:
    """Upsert a domain-knowledge entry in the SQLite metadata store."""
    store.upsert_domain_knowledge(dataset_id, topic, content)


def sync_upsert_query_pattern(
    store: MetadataStore,
    dataset_id: int,
    pattern_type: str,
    content: str,
) -> None:
    """Upsert a query pattern in the SQLite metadata store."""
    store.upsert_query_pattern(dataset_id, pattern_type, content)


# ── Workspace-level upserts ──────────────────────────────────


def sync_upsert_ws_business_term(
    store: MetadataStore,
    workspace_id: int,
    term: str,
    definition: str = "",
    synonyms: list[str] | None = None,
) -> None:
    """Upsert a workspace business term in the SQLite metadata store."""
    store.upsert_ws_business_term(workspace_id, term, definition, synonyms)


def sync_upsert_ws_domain_knowledge(
    store: MetadataStore,
    workspace_id: int,
    topic: str,
    content: str,
) -> None:
    """Upsert a workspace domain-knowledge entry in the SQLite metadata store."""
    store.upsert_ws_domain_knowledge(workspace_id, topic, content)


def sync_upsert_ws_query_pattern(
    store: MetadataStore,
    workspace_id: int,
    pattern_type: str,
    content: str,
) -> None:
    """Upsert a workspace query pattern in the SQLite metadata store."""
    store.upsert_ws_query_pattern(workspace_id, pattern_type, content)


# ── Dataset-level deletes ────────────────────────────────────


def sync_delete_business_term(store: MetadataStore, dataset_id: int, term: str) -> bool:
    """Delete a business term from the SQLite metadata store."""
    return store.delete_business_term(dataset_id, term)


def sync_delete_domain_knowledge(store: MetadataStore, dataset_id: int, topic: str) -> bool:
    """Delete a domain-knowledge entry from the SQLite metadata store."""
    return store.delete_domain_knowledge(dataset_id, topic)


def sync_delete_query_pattern(
    store: MetadataStore, dataset_id: int, pattern_type: str, content: str
) -> bool:
    """Delete a query pattern from the SQLite metadata store."""
    return store.delete_query_pattern(dataset_id, pattern_type, content)


# ── Workspace-level deletes ──────────────────────────────────


def sync_delete_ws_business_term(store: MetadataStore, workspace_id: int, term: str) -> bool:
    """Delete a workspace business term from the SQLite metadata store."""
    return store.delete_ws_business_term(workspace_id, term)


def sync_delete_ws_domain_knowledge(store: MetadataStore, workspace_id: int, topic: str) -> bool:
    """Delete a workspace domain-knowledge entry from the SQLite metadata store."""
    return store.delete_ws_domain_knowledge(workspace_id, topic)


def sync_delete_ws_query_pattern(
    store: MetadataStore, workspace_id: int, pattern_type: str, content: str
) -> bool:
    """Delete a workspace query pattern from the SQLite metadata store."""
    return store.delete_ws_query_pattern(workspace_id, pattern_type, content)
