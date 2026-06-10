"""Context builder: stores introspection results and populates semantic dimensions."""

from __future__ import annotations

from gyzasql.db.introspect import SchemaInfo
from gyzasql.semantic_layer.store import MetadataStore


def store_introspection(store: MetadataStore, dataset_id: int, schema: SchemaInfo) -> None:
    """Persist introspection results (tables, columns, relationships) into the metadata DB."""
    for table in schema.tables:
        table_id = store.upsert_table(dataset_id, table.name)
        for col in table.columns:
            store.upsert_column(
                table_id,
                name=col.name,
                data_type=col.data_type,
                nullable=col.nullable,
                is_primary_key=col.is_primary_key,
            )

    for fk in schema.foreign_keys:
        store.upsert_relationship(
            dataset_id,
            from_table=fk.from_table,
            from_column=fk.from_column,
            to_table=fk.to_table,
            to_column=fk.to_column,
        )


def populate_data_dictionary(
    store: MetadataStore,
    dataset_id: int,
    descriptions: dict[str, str | dict[str, str]],
) -> None:
    """Populate table and column descriptions.

    ``descriptions`` maps table names to either a string (table description)
    or a dict with an optional ``"_table"`` key for the table description
    and column-name keys for column descriptions.

    Example::

        {
            "Album": {
                "_table": "Music albums",
                "AlbumId": "Primary key",
                "Title": "Album title",
            },
            "Artist": "Musical artists",
        }
    """
    for table_name, value in descriptions.items():
        if isinstance(value, str):
            store.upsert_table(dataset_id, table_name, description=value)
            continue

        table_desc = value.pop("_table", "")
        table_id = store.upsert_table(dataset_id, table_name, description=table_desc)
        for col_name, col_desc in value.items():
            # Update only the description; preserve existing type info by re-upserting
            existing = store.conn.execute(
                "SELECT data_type, nullable, is_primary_key FROM columns WHERE table_id=? AND name=?",
                (table_id, col_name),
            ).fetchone()
            if existing:
                store.upsert_column(
                    table_id,
                    name=col_name,
                    data_type=existing["data_type"],
                    nullable=bool(existing["nullable"]),
                    is_primary_key=bool(existing["is_primary_key"]),
                    description=col_desc,
                )


def populate_query_patterns(
    store: MetadataStore,
    dataset_id: int,
    patterns: dict[str, list[str]],
) -> None:
    """Add common query patterns grouped by type.

    ``patterns`` maps a pattern type (e.g. ``"join_path"``, ``"gotcha"``,
    ``"business_rule"``) to a list of content strings.
    """
    from gyzasql.semantic_layer.vector_sync import sync_upsert_query_pattern

    for pattern_type, items in patterns.items():
        for content in items:
            sync_upsert_query_pattern(store, dataset_id, pattern_type, content)


def populate_business_context(
    store: MetadataStore,
    dataset_id: int,
    terms: dict[str, dict[str, str | list[str]]],
) -> None:
    """Add business terminology.

    ``terms`` maps a term to ``{"definition": ..., "synonyms": [...]}``.
    """
    from gyzasql.semantic_layer.vector_sync import sync_upsert_business_term

    for term, info in terms.items():
        sync_upsert_business_term(
            store,
            dataset_id,
            term=term,
            definition=str(info.get("definition", "")),
            synonyms=info.get("synonyms", []),  # type: ignore[arg-type]
        )


def populate_domain_knowledge(
    store: MetadataStore,
    dataset_id: int,
    knowledge: dict[str, str],
) -> None:
    """Add domain knowledge entries (topic → content)."""
    from gyzasql.semantic_layer.vector_sync import sync_upsert_domain_knowledge

    for topic, content in knowledge.items():
        sync_upsert_domain_knowledge(store, dataset_id, topic, content)


def _merge_with_workspace_context(store: MetadataStore, workspace_id: int, context: dict) -> dict:
    """Merge workspace-level context into dataset context.

    Dataset-level entries take precedence: if a business term or domain knowledge
    topic exists at both levels, the dataset version wins.  Query patterns are
    additive (deduplicated by ``(pattern_type, content)`` pair).
    """
    # Business terms: dataset overrides workspace for same term name
    ws_terms = store.get_ws_business_terms(workspace_id)
    ds_term_names = {t["term"] for t in context.get("business_terms", [])}
    inherited_terms = [t for t in ws_terms if t["term"] not in ds_term_names]
    context["business_terms"] = context.get("business_terms", []) + inherited_terms

    # Domain knowledge: dataset overrides workspace for same topic
    ws_dk = store.get_ws_domain_knowledge(workspace_id)
    ds_topics = {dk["topic"] for dk in context.get("domain_knowledge", [])}
    inherited_dk = [dk for dk in ws_dk if dk["topic"] not in ds_topics]
    context["domain_knowledge"] = context.get("domain_knowledge", []) + inherited_dk

    # Query patterns: additive, deduplicate by (pattern_type, content)
    ws_patterns = store.get_ws_query_patterns(workspace_id)
    ds_pattern_keys = {(p["pattern_type"], p["content"]) for p in context.get("query_patterns", [])}
    inherited_patterns = [p for p in ws_patterns if (p["pattern_type"], p["content"]) not in ds_pattern_keys]
    context["query_patterns"] = context.get("query_patterns", []) + inherited_patterns

    return context


def get_context(store: MetadataStore, dataset_id: int) -> dict:
    """Build the full context payload for a dataset, including inherited workspace context."""
    tables = store.get_tables(dataset_id)
    for t in tables:
        t["columns"] = store.get_columns(t["id"])

    context = {
        "tables": tables,
        "relationships": store.get_relationships(dataset_id),
        "query_patterns": store.get_query_patterns(dataset_id),
        "business_terms": store.get_business_terms(dataset_id),
        "domain_knowledge": store.get_domain_knowledge(dataset_id),
    }

    # Merge workspace-level context if dataset belongs to a workspace
    workspace_id = store.get_dataset_workspace_id(dataset_id)
    if workspace_id is not None:
        context = _merge_with_workspace_context(store, workspace_id, context)

    return context


def get_workspace_context(store: MetadataStore, workspace_id: int, primary_dataset_id: int) -> dict:
    """Build merged context from all datasets in a workspace.

    The primary dataset's context is returned in full (tables with columns,
    relationships, etc.). Sibling datasets contribute summary-level context:
    table names/descriptions, business terms, and domain knowledge.
    """
    primary = get_context(store, primary_dataset_id)
    siblings = store.get_workspace_datasets(workspace_id)

    related_datasets: list[dict] = []
    for ds in siblings:
        if ds["id"] == primary_dataset_id:
            continue

        ds_tables = store.get_tables(ds["id"])
        related_datasets.append(
            {
                "name": ds["name"],
                "db_type": ds["db_type"],
                "tables": [{"name": t["name"], "description": t.get("description", "")} for t in ds_tables],
                "business_terms": store.get_business_terms(ds["id"]),
                "domain_knowledge": store.get_domain_knowledge(ds["id"]),
            }
        )

    primary["related_datasets"] = related_datasets
    return primary
