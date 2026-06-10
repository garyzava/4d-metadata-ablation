"""SQLite metadata store for the semantic layer.

Creates tables on first use (no migration framework needed for MVP).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("gyzasql_metadata.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

INSERT OR IGNORE INTO workspaces (id, name, description, created_at)
VALUES (1, 'default', 'Default workspace', datetime('now'));

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    db_type TEXT NOT NULL DEFAULT 'postgres',
    connection_ref TEXT NOT NULL,
    workspace_id INTEGER NOT NULL DEFAULT 1 REFERENCES workspaces(id),
    created_at TEXT NOT NULL,
    UNIQUE(workspace_id, name)
);

CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    UNIQUE(dataset_id, name)
);

CREATE TABLE IF NOT EXISTS columns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id INTEGER NOT NULL REFERENCES tables(id),
    name TEXT NOT NULL,
    data_type TEXT NOT NULL,
    nullable INTEGER NOT NULL DEFAULT 1,
    is_primary_key INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    UNIQUE(table_id, name)
);

CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    from_table TEXT NOT NULL,
    from_column TEXT NOT NULL,
    to_table TEXT NOT NULL,
    to_column TEXT NOT NULL,
    UNIQUE(dataset_id, from_table, from_column, to_table, to_column)
);

CREATE TABLE IF NOT EXISTS query_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    pattern_type TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(dataset_id, pattern_type, content)
);

CREATE TABLE IF NOT EXISTS business_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    term TEXT NOT NULL,
    definition TEXT NOT NULL DEFAULT '',
    synonyms_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE(dataset_id, term)
);

CREATE TABLE IF NOT EXISTS domain_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(dataset_id, topic)
);

CREATE TABLE IF NOT EXISTS completeness_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    score REAL NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    dataset_id INTEGER NOT NULL DEFAULT 0,
    workspace_id INTEGER NOT NULL DEFAULT 0,
    doc_type TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    collection TEXT NOT NULL DEFAULT 'gyzasql_docs',
    ingested_at TEXT NOT NULL,
    chunking_strategy TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    confidence_json TEXT NOT NULL DEFAULT '',
    UNIQUE(source, dataset_id, workspace_id)
);

CREATE TABLE IF NOT EXISTS ws_business_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
    term TEXT NOT NULL,
    definition TEXT NOT NULL DEFAULT '',
    synonyms_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE(workspace_id, term)
);

CREATE TABLE IF NOT EXISTS ws_domain_knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
    topic TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(workspace_id, topic)
);

CREATE TABLE IF NOT EXISTS ws_query_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
    pattern_type TEXT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE(workspace_id, pattern_type, content)
);
"""


@dataclass
class MetadataStore:
    """Thin wrapper around a SQLite database for metadata storage."""

    db_path: Path = DEFAULT_DB_PATH

    def __post_init__(self) -> None:
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate_to_workspaces()
            self._migrate_dataset_unique_constraint()
            self._conn.executescript(SCHEMA_SQL)
            self._migrate_add_ws_context_tables()
            self._migrate_documents_to_fk()
            self._migrate_documents_add_pipeline_metadata()
        return self._conn

    def _migrate_to_workspaces(self) -> None:
        """Add workspaces table and workspace_id to datasets if upgrading an existing DB."""
        # Check if datasets table exists and is missing workspace_id
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'"
        ).fetchone()
        if existing is None:
            return  # Fresh DB — SCHEMA_SQL will create everything

        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(datasets)").fetchall()]
        if "workspace_id" in cols:
            return  # Already migrated

        # Note: ALTER TABLE ADD COLUMN in SQLite cannot use REFERENCES, so we
        # omit it here. The FK constraint exists in SCHEMA_SQL for fresh databases.
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            INSERT OR IGNORE INTO workspaces (id, name, description, created_at)
            VALUES (1, 'default', 'Default workspace', datetime('now'));
            ALTER TABLE datasets ADD COLUMN workspace_id INTEGER NOT NULL DEFAULT 1;
        """)

    def _migrate_dataset_unique_constraint(self) -> None:
        """Migrate datasets UNIQUE(name) → UNIQUE(workspace_id, name) for workspace scoping."""
        # Check if datasets table exists
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'"
        ).fetchone()
        if existing is None:
            return  # Fresh DB — SCHEMA_SQL will create with correct constraint

        # Check current index: if there's a unique index on (name) alone, we need to migrate
        indexes = self._conn.execute("PRAGMA index_list(datasets)").fetchall()
        needs_migration = False
        for idx in indexes:
            if idx[2]:  # unique flag
                cols = self._conn.execute(f"PRAGMA index_info({idx[1]})").fetchall()
                col_names = [c[2] for c in cols]
                if col_names == ["name"]:
                    needs_migration = True
                    break

        if not needs_migration:
            return

        # SQLite doesn't support ALTER TABLE DROP CONSTRAINT, so rebuild the table.
        # Must disable FK checks temporarily since other tables reference datasets.
        self._conn.executescript("""
            PRAGMA foreign_keys=OFF;
            CREATE TABLE IF NOT EXISTS datasets_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                db_type TEXT NOT NULL DEFAULT 'postgres',
                connection_ref TEXT NOT NULL,
                workspace_id INTEGER NOT NULL DEFAULT 1 REFERENCES workspaces(id),
                created_at TEXT NOT NULL,
                UNIQUE(workspace_id, name)
            );
            INSERT OR IGNORE INTO datasets_new (id, name, db_type, connection_ref, workspace_id, created_at)
                SELECT id, name, db_type, connection_ref, workspace_id, created_at FROM datasets;
            DROP TABLE datasets;
            ALTER TABLE datasets_new RENAME TO datasets;
            PRAGMA foreign_keys=ON;
        """)

    def _migrate_add_ws_context_tables(self) -> None:
        """Add workspace-level context tables if they don't exist (for existing DBs)."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ws_business_context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
                term TEXT NOT NULL,
                definition TEXT NOT NULL DEFAULT '',
                synonyms_json TEXT NOT NULL DEFAULT '[]',
                UNIQUE(workspace_id, term)
            );
            CREATE TABLE IF NOT EXISTS ws_domain_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(workspace_id, topic)
            );
            CREATE TABLE IF NOT EXISTS ws_query_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id INTEGER NOT NULL REFERENCES workspaces(id),
                pattern_type TEXT NOT NULL,
                content TEXT NOT NULL,
                UNIQUE(workspace_id, pattern_type, content)
            );
        """)

    def _migrate_documents_to_fk(self) -> None:
        """Migrate documents table from text dataset/workspace to FK-based scoping."""
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
        if existing is None:
            return  # Fresh DB — SCHEMA_SQL already creates the FK version

        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(documents)").fetchall()]
        if "dataset_id" in cols:
            return  # Already migrated

        if "dataset" not in cols:
            return  # Fresh DB without old schema

        # Rebuild the table with FK columns (use 0 sentinel for unscoped docs)
        self._conn.executescript("""
            PRAGMA foreign_keys=OFF;
            CREATE TABLE IF NOT EXISTS documents_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                dataset_id INTEGER NOT NULL DEFAULT 0,
                workspace_id INTEGER NOT NULL DEFAULT 0,
                doc_type TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                collection TEXT NOT NULL DEFAULT 'gyzasql_docs',
                ingested_at TEXT NOT NULL,
                UNIQUE(source, dataset_id, workspace_id)
            );
            DROP TABLE documents;
            ALTER TABLE documents_new RENAME TO documents;
            PRAGMA foreign_keys=ON;
        """)

    def _migrate_documents_add_pipeline_metadata(self) -> None:
        """Add pipeline metadata columns to documents table if missing."""
        existing = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
        if existing is None:
            return  # Fresh DB — SCHEMA_SQL already creates with new columns

        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(documents)").fetchall()]
        if "chunking_strategy" not in cols:
            self._conn.execute("ALTER TABLE documents ADD COLUMN chunking_strategy TEXT NOT NULL DEFAULT ''")
        if "embedding_model" not in cols:
            self._conn.execute("ALTER TABLE documents ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''")
        if "confidence_json" not in cols:
            self._conn.execute("ALTER TABLE documents ADD COLUMN confidence_json TEXT NOT NULL DEFAULT ''")
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Workspaces ────────────────────────────────────────────

    def upsert_workspace(self, name: str, description: str = "") -> int:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            "INSERT INTO workspaces (name, description, created_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET description=excluded.description",
            (name, description, now),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM workspaces WHERE name=?", (name,)).fetchone()
        return row["id"]

    def get_workspace_id(self, name: str) -> int | None:
        row = self.conn.execute("SELECT id FROM workspaces WHERE name=?", (name,)).fetchone()
        return row["id"] if row else None

    def get_workspace_datasets(self, workspace_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, name, db_type, connection_ref, workspace_id, created_at FROM datasets WHERE workspace_id=?",
            (workspace_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_workspaces(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM workspaces ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    # ── Datasets ──────────────────────────────────────────────

    def upsert_dataset(
        self, name: str, connection_ref: str, db_type: str = "postgres", workspace: str = "default"
    ) -> int:
        workspace_id = self.get_workspace_id(workspace)
        if workspace_id is None:
            workspace_id = self.upsert_workspace(workspace)
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            "INSERT INTO datasets (name, db_type, connection_ref, workspace_id, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(workspace_id, name) DO UPDATE SET "
            "connection_ref=excluded.connection_ref",
            (name, db_type, connection_ref, workspace_id, now),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM datasets WHERE workspace_id=? AND name=?", (workspace_id, name)
        ).fetchone()
        return row["id"]

    def get_dataset_id(self, name: str, workspace: str | None = None) -> int | None:
        if workspace is not None:
            ws_id = self.get_workspace_id(workspace)
            if ws_id is None:
                return None
            row = self.conn.execute("SELECT id FROM datasets WHERE workspace_id=? AND name=?", (ws_id, name)).fetchone()
        else:
            row = self.conn.execute("SELECT id FROM datasets WHERE name=?", (name,)).fetchone()
        return row["id"] if row else None

    def get_dataset(self, name: str, workspace: str | None = None) -> dict[str, Any] | None:
        if workspace is not None:
            ws_id = self.get_workspace_id(workspace)
            if ws_id is None:
                return None
            row = self.conn.execute(
                "SELECT id, name, db_type, connection_ref, workspace_id, created_at "
                "FROM datasets WHERE workspace_id=? AND name=?",
                (ws_id, name),
            ).fetchone()
        else:
            rows = self.conn.execute(
                "SELECT id, name, db_type, connection_ref, workspace_id, created_at "
                "FROM datasets WHERE name=? ORDER BY workspace_id, id",
                (name,),
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise ValueError(f"Dataset '{name}' is ambiguous across workspaces. Pass a workspace name explicitly.")
            row = rows[0]
        return dict(row) if row else None

    def get_db_type(self, dataset_id: int) -> str:
        row = self.conn.execute("SELECT db_type FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return row["db_type"] if row else "postgres"

    def get_dataset_workspace_id(self, dataset_id: int) -> int | None:
        row = self.conn.execute("SELECT workspace_id FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return row["workspace_id"] if row else None

    # ── Tables & Columns ──────────────────────────────────────

    def upsert_table(self, dataset_id: int, name: str, description: str = "") -> int:
        self.conn.execute(
            "INSERT INTO tables (dataset_id, name, description) VALUES (?, ?, ?) "
            "ON CONFLICT(dataset_id, name) DO UPDATE SET description=excluded.description",
            (dataset_id, name, description),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM tables WHERE dataset_id=? AND name=?", (dataset_id, name)).fetchone()
        return row["id"]

    def upsert_column(
        self,
        table_id: int,
        name: str,
        data_type: str,
        nullable: bool = True,
        is_primary_key: bool = False,
        description: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT INTO columns (table_id, name, data_type, nullable, is_primary_key, description) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(table_id, name) DO UPDATE SET "
            "data_type=excluded.data_type, nullable=excluded.nullable, "
            "is_primary_key=excluded.is_primary_key, description=excluded.description",
            (table_id, name, data_type, int(nullable), int(is_primary_key), description),
        )
        self.conn.commit()

    # ── Relationships ─────────────────────────────────────────

    def upsert_relationship(
        self, dataset_id: int, from_table: str, from_column: str, to_table: str, to_column: str
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO relationships "
            "(dataset_id, from_table, from_column, to_table, to_column) "
            "VALUES (?, ?, ?, ?, ?)",
            (dataset_id, from_table, from_column, to_table, to_column),
        )
        self.conn.commit()

    # ── Query Patterns ────────────────────────────────────────

    def upsert_query_pattern(self, dataset_id: int, pattern_type: str, content: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO query_patterns (dataset_id, pattern_type, content) VALUES (?, ?, ?)",
            (dataset_id, pattern_type, content),
        )
        self.conn.commit()

    # ── Business Context ──────────────────────────────────────

    def upsert_business_term(
        self, dataset_id: int, term: str, definition: str = "", synonyms: list[str] | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO business_context (dataset_id, term, definition, synonyms_json) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(dataset_id, term) DO UPDATE SET "
            "definition=excluded.definition, synonyms_json=excluded.synonyms_json",
            (dataset_id, term, definition, json.dumps(synonyms or [])),
        )
        self.conn.commit()

    # ── Domain Knowledge ──────────────────────────────────────

    def upsert_domain_knowledge(self, dataset_id: int, topic: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO domain_knowledge (dataset_id, topic, content) VALUES (?, ?, ?) "
            "ON CONFLICT(dataset_id, topic) DO UPDATE SET content=excluded.content",
            (dataset_id, topic, content),
        )
        self.conn.commit()

    # ── Workspace-level Business Context ─────────────────────

    def upsert_ws_business_term(
        self, workspace_id: int, term: str, definition: str = "", synonyms: list[str] | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO ws_business_context (workspace_id, term, definition, synonyms_json) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(workspace_id, term) DO UPDATE SET "
            "definition=excluded.definition, synonyms_json=excluded.synonyms_json",
            (workspace_id, term, definition, json.dumps(synonyms or [])),
        )
        self.conn.commit()

    # ── Workspace-level Domain Knowledge ──────────────────

    def upsert_ws_domain_knowledge(self, workspace_id: int, topic: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO ws_domain_knowledge (workspace_id, topic, content) VALUES (?, ?, ?) "
            "ON CONFLICT(workspace_id, topic) DO UPDATE SET content=excluded.content",
            (workspace_id, topic, content),
        )
        self.conn.commit()

    # ── Workspace-level Query Patterns ────────────────────

    def upsert_ws_query_pattern(self, workspace_id: int, pattern_type: str, content: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO ws_query_patterns (workspace_id, pattern_type, content) VALUES (?, ?, ?)",
            (workspace_id, pattern_type, content),
        )
        self.conn.commit()

    # ── Delete Semantic Context ────────────────────────────────

    def delete_business_term(self, dataset_id: int, term: str) -> bool:
        cur = self.conn.execute("DELETE FROM business_context WHERE dataset_id=? AND term=?", (dataset_id, term))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_domain_knowledge(self, dataset_id: int, topic: str) -> bool:
        cur = self.conn.execute("DELETE FROM domain_knowledge WHERE dataset_id=? AND topic=?", (dataset_id, topic))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_query_pattern(self, dataset_id: int, pattern_type: str, content: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM query_patterns WHERE dataset_id=? AND pattern_type=? AND content=?",
            (dataset_id, pattern_type, content),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_ws_business_term(self, workspace_id: int, term: str) -> bool:
        cur = self.conn.execute("DELETE FROM ws_business_context WHERE workspace_id=? AND term=?", (workspace_id, term))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_ws_domain_knowledge(self, workspace_id: int, topic: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM ws_domain_knowledge WHERE workspace_id=? AND topic=?", (workspace_id, topic)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete_ws_query_pattern(self, workspace_id: int, pattern_type: str, content: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM ws_query_patterns WHERE workspace_id=? AND pattern_type=? AND content=?",
            (workspace_id, pattern_type, content),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ── Completeness Snapshots ────────────────────────────────

    def save_snapshot(self, dataset_id: int, score: float, metrics: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            "INSERT INTO completeness_snapshots (dataset_id, score, metrics_json, created_at) VALUES (?, ?, ?, ?)",
            (dataset_id, score, json.dumps(metrics), now),
        )
        self.conn.commit()

    def get_snapshots(self, dataset_id: int, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT score, metrics_json, created_at FROM completeness_snapshots "
            "WHERE dataset_id=? ORDER BY created_at DESC LIMIT ?",
            (dataset_id, limit),
        ).fetchall()
        result = []
        for r in rows:
            entry = {"score": r["score"], "created_at": r["created_at"]}
            entry["dimensions"] = json.loads(r["metrics_json"])
            result.append(entry)
        return result

    # ── Read helpers ──────────────────────────────────────────

    def get_tables(self, dataset_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM tables WHERE dataset_id=?", (dataset_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_columns(self, table_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM columns WHERE table_id=?", (table_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_relationships(self, dataset_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM relationships WHERE dataset_id=?", (dataset_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_query_patterns(self, dataset_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM query_patterns WHERE dataset_id=?", (dataset_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_business_terms(self, dataset_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM business_context WHERE dataset_id=?", (dataset_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_domain_knowledge(self, dataset_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM domain_knowledge WHERE dataset_id=?", (dataset_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_ws_business_terms(self, workspace_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM ws_business_context WHERE workspace_id=?", (workspace_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_ws_domain_knowledge(self, workspace_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM ws_domain_knowledge WHERE workspace_id=?", (workspace_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_ws_query_patterns(self, workspace_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM ws_query_patterns WHERE workspace_id=?", (workspace_id,)).fetchall()
        return [dict(r) for r in rows]

    # ── Document Lifecycle ──────────────────────────────────────

    def _resolve_doc_ids(self, dataset: str | None, workspace: str | None) -> tuple[int, int]:
        """Resolve dataset/workspace names to IDs for document operations.

        Returns 0 for unresolved or unspecified names (sentinel for 'unscoped').
        """
        ds_id = (self.get_dataset_id(dataset) if dataset else None) or 0
        ws_id = (self.get_workspace_id(workspace) if workspace else None) or 0
        return ds_id, ws_id

    def upsert_document(
        self,
        source: str,
        chunk_count: int,
        collection: str = "gyzasql_docs",
        dataset: str | None = None,
        workspace: str | None = None,
        doc_type: str = "",
        title: str = "",
        chunking_strategy: str = "",
        embedding_model: str = "",
        confidence_json: str = "",
    ) -> int:
        """Track an ingested document for lifecycle management."""
        now = datetime.now(UTC).isoformat()
        ds_id, ws_id = self._resolve_doc_ids(dataset, workspace)
        self.conn.execute(
            "INSERT INTO documents "
            "(source, dataset_id, workspace_id, doc_type, title, chunk_count, collection, ingested_at, "
            "chunking_strategy, embedding_model, confidence_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source, dataset_id, workspace_id) DO UPDATE SET "
            "doc_type=excluded.doc_type, title=excluded.title, "
            "chunk_count=excluded.chunk_count, ingested_at=excluded.ingested_at, "
            "chunking_strategy=excluded.chunking_strategy, embedding_model=excluded.embedding_model, "
            "confidence_json=excluded.confidence_json",
            (
                source,
                ds_id,
                ws_id,
                doc_type,
                title,
                chunk_count,
                collection,
                now,
                chunking_strategy,
                embedding_model,
                confidence_json,
            ),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT id FROM documents WHERE source=? AND dataset_id=? AND workspace_id=?",
            (source, ds_id, ws_id),
        ).fetchone()
        return row["id"]

    def list_documents(self, dataset: str | None = None, workspace: str | None = None) -> list[dict[str, Any]]:
        """List ingested documents, optionally filtered by dataset/workspace."""
        query = (
            "SELECT d.*, ds.name as dataset, w.name as workspace "
            "FROM documents d "
            "LEFT JOIN datasets ds ON d.dataset_id = ds.id "
            "LEFT JOIN workspaces w ON d.workspace_id = w.id "
            "WHERE 1=1"
        )
        params: list[Any] = []
        if dataset is not None:
            query += " AND ds.name=?"
            params.append(dataset)
        if workspace is not None:
            query += " AND w.name=?"
            params.append(workspace)
        query += " ORDER BY d.ingested_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_document(self, source: str, dataset: str | None = None, workspace: str | None = None) -> dict | None:
        """Get a specific document record by source path."""
        ds_id, ws_id = self._resolve_doc_ids(dataset, workspace)
        row = self.conn.execute(
            "SELECT * FROM documents WHERE source=? AND dataset_id=? AND workspace_id=?",
            (source, ds_id, ws_id),
        ).fetchone()
        return dict(row) if row else None

    def delete_document_record(self, source: str, dataset: str | None = None, workspace: str | None = None) -> bool:
        """Delete a document record. Returns True if a row was deleted."""
        ds_id, ws_id = self._resolve_doc_ids(dataset, workspace)
        cur = self.conn.execute(
            "DELETE FROM documents WHERE source=? AND dataset_id=? AND workspace_id=?",
            (source, ds_id, ws_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def count_described_tables(self, dataset_id: int) -> tuple[int, int]:
        """Return (described, total) table counts."""
        rows = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN description != '' THEN 1 ELSE 0 END) as described "
            "FROM tables WHERE dataset_id=?",
            (dataset_id,),
        ).fetchone()
        return rows["described"] or 0, rows["total"] or 0

    def count_described_columns(self, dataset_id: int) -> tuple[int, int]:
        """Return (described, total) column counts."""
        rows = self.conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN c.description != '' THEN 1 ELSE 0 END) as described "
            "FROM columns c JOIN tables t ON c.table_id = t.id "
            "WHERE t.dataset_id=?",
            (dataset_id,),
        ).fetchone()
        return rows["described"] or 0, rows["total"] or 0
