"""Text2SQL tool: generate SQL from natural language using schema context and an LLM."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from gyzasql.llm.client import chat_completion, get_config
from gyzasql.llm.prompt_loader import load_template
from gyzasql.semantic_layer.context import get_context
from gyzasql.semantic_layer.schema_filter import filter_context
from gyzasql.semantic_layer.store import MetadataStore

# This vendored copy drops semantic_layer.scoring.compute_completeness (gyzasql IP, not a paper
# metric for the metadata-ablation experiment). See src/gyzasql/_VENDORED.md.

_DIALECT_CONFIG: dict[str, dict[str, str]] = {
    "postgres": {
        "name": "PostgreSQL",
        "quoting_rule": ('Always double-quote all table and column identifiers (e.g. SELECT "col" FROM "table").'),
        "quote": '"',
    },
    "mysql": {
        "name": "MySQL",
        "quoting_rule": ("Always backtick-quote all table and column identifiers (e.g. SELECT `col` FROM `table`)."),
        "quote": "`",
    },
    "sqlite": {
        "name": "SQLite",
        "quoting_rule": ('Always double-quote all table and column identifiers (e.g. SELECT "col" FROM "table").'),
        "quote": '"',
    },
    "bigquery": {
        "name": "BigQuery",
        "quoting_rule": ("Always backtick-quote all table and column identifiers (e.g. SELECT `col` FROM `table`)."),
        "quote": "`",
    },
    "mssql": {
        "name": "SQL Server",
        "quoting_rule": ("Always bracket-quote all table and column identifiers (e.g. SELECT [col] FROM [table])."),
        "quote_open": "[",
        "quote_close": "]",
    },
}

_DEFAULT_DIALECT = _DIALECT_CONFIG["postgres"]


def _get_dialect_config(db_type: str) -> dict[str, str]:
    return _DIALECT_CONFIG.get(db_type, _DEFAULT_DIALECT)


def _build_system_prompt(db_type: str = "postgres") -> str:
    cfg = _get_dialect_config(db_type)
    return load_template("text2sql", dialect_name=cfg["name"], quoting_rule=cfg["quoting_rule"])


class Text2SQLResult(BaseModel):
    question: str
    sql: str
    schema_context_used: bool = True
    warnings: list[str] = []
    thinking: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _quote_ident(name: str, db_type: str) -> str:
    """Wrap an identifier in the dialect-appropriate quote characters."""
    cfg = _get_dialect_config(db_type)
    open_q = cfg.get("quote_open", cfg.get("quote", '"'))
    close_q = cfg.get("quote_close", cfg.get("quote", '"'))
    return f"{open_q}{name}{close_q}"


def _build_schema_prompt(context: dict, db_type: str = "postgres") -> str:
    """Format schema context into a compact prompt section."""
    lines = ["## Database Schema\n"]
    for table in context.get("tables", []):
        q_table = _quote_ident(table["name"], db_type)
        cols = ", ".join(
            f"{_quote_ident(c['name'], db_type)} ({c['data_type']}{'*' if c.get('is_primary_key') else ''})"
            for c in table.get("columns", [])
        )
        desc = f" -- {table['description']}" if table.get("description") else ""
        lines.append(f"- {q_table}({cols}){desc}")

    rels = context.get("relationships", [])
    if rels:
        lines.append("\n## Foreign Keys")
        for r in rels:
            lines.append(
                f"- {_quote_ident(r['from_table'], db_type)}.{_quote_ident(r['from_column'], db_type)}"
                f" -> {_quote_ident(r['to_table'], db_type)}.{_quote_ident(r['to_column'], db_type)}"
            )

    patterns = context.get("query_patterns", [])
    if patterns:
        lines.append("\n## Query Patterns / Notes")
        for p in patterns:
            lines.append(f"- [{p.get('pattern_type', '')}] {p.get('content', '')}")

    terms = context.get("business_terms", [])
    if terms:
        lines.append("\n## Business Terminology")
        for t in terms:
            raw = t.get("synonyms_json", [])
            synonyms = json.loads(raw) if isinstance(raw, str) else raw
            syn = f" (aka: {', '.join(synonyms)})" if synonyms else ""
            lines.append(f"- {t['term']}: {t.get('definition', '')}{syn}")

    knowledge = context.get("domain_knowledge", [])
    if knowledge:
        lines.append("\n## Domain Knowledge")
        for dk in knowledge:
            topic = dk.get("topic", "")
            content = dk.get("content", "")
            lines.append(f"- {topic}: {content}" if topic else f"- {content}")

    related = context.get("related_datasets", [])
    if related:
        lines.append("\n## Related Data Sources (context only, not queryable)")
        for ds in related:
            table_names = ", ".join(t["name"] for t in ds.get("tables", []))
            lines.append(f'- Dataset "{ds["name"]}" ({ds.get("db_type", "unknown")}): {table_names}')
            for bt in ds.get("business_terms", []):
                raw = bt.get("synonyms_json", [])
                syns = json.loads(raw) if isinstance(raw, str) else raw
                syn_str = f" (aka: {', '.join(syns)})" if syns else ""
                lines.append(f"  - {bt['term']}: {bt.get('definition', '')}{syn_str}")

    rag_chunks = context.get("rag_chunks", [])
    if rag_chunks:
        lines.append("\n## Relevant Document Context")
        for chunk in rag_chunks:
            source = chunk.get("title") or chunk.get("source", "")
            lines.append(f"### From: {source}")
            lines.append(chunk.get("text", ""))

    return "\n".join(lines)


# Regex compiled at import time. Ordered preference:
#   1. ```sql ... ``` or ``` ... ``` fenced blocks (last fence wins — Gemma 4
#      may include earlier fences as worked-out examples in its thinking).
#   2. Last line-anchored SELECT/WITH statement to end-of-response (handles
#      Gemma 4's "...thinking...\nSELECT ..." pattern with no fence).
import re as _re  # local alias to avoid module-level import pollution

_SQL_FENCE_RE = _re.compile(r"```(?:sql\s*\n?|\s*\n?)(.*?)```", _re.DOTALL | _re.IGNORECASE)
# Column-0 anchor (no leading whitespace) so nested subquery SELECTs at indent
# level > 0 are NOT treated as a separate statement-start.
_SQL_TAIL_RE = _re.compile(r"^(SELECT|WITH)\b", _re.MULTILINE | _re.IGNORECASE)


def _extract_sql(text: str) -> str:
    """Extract SQL from a model response that may contain prose, thinking, or fences.

    Added 2026-05-08 to recover Gemma 4 outputs (gyzasql's `_chat_gemini` adapter
    uses the legacy `google-generativeai` SDK, which doesn't separate thinking
    from final answer; Gemma 4 emits ~15 KB of inline reasoning prose followed
    by SQL). For models that already return clean SQL (Qwen 9B/27B/30B/3.6, Opus
    4.7), the fallback returns the response unchanged. See src/gyzasql/_VENDORED.md.

    Heuristic, in order:
      1. ```sql ... ``` or ``` ... ``` fenced block (last fence wins; Gemma may
         include earlier fences as worked-out examples in its thinking).
      2. Last column-0 SELECT/WITH statement to end-of-response. Column-0 anchor
         is critical: indented SELECTs are usually subqueries, not top-level.
      3. Backward-compatible whole-response fence strip.
    """
    if not text:
        return ""
    s = text.strip()

    # 1. Fenced code block — take the LAST fence (Gemma's earlier fences may be examples)
    fences = _SQL_FENCE_RE.findall(s)
    if fences:
        candidate = fences[-1].strip()
        if _re.search(r"\b(SELECT|WITH)\b", candidate, _re.IGNORECASE):
            return candidate

    # 2. Last column-0 SELECT/WITH to end-of-string
    last_match = None
    for m in _SQL_TAIL_RE.finditer(s):
        last_match = m
    if last_match:
        return s[last_match.start():].strip()

    # 3. No structure detected. Backward-compatible: strip whole-response fences.
    if s.startswith("```"):
        s = "\n".join(s.split("\n")[1:])
    if s.endswith("```"):
        s = "\n".join(s.split("\n")[:-1])
    return s.strip()


def generate_sql(
    store: MetadataStore,
    dataset_id: int,
    question: str,
    model: str | None = None,
) -> Text2SQLResult:
    """Generate SQL for a question using schema context and an LLM."""
    config = get_config(model=model) if model else get_config()

    db_type = store.get_db_type(dataset_id)
    context = get_context(store, dataset_id)
    context = filter_context(context, question)
    schema_prompt = _build_schema_prompt(context, db_type=db_type)

    try:
        result = chat_completion(
            messages=[
                {"role": "system", "content": _build_system_prompt(db_type)},
                {"role": "user", "content": f"{schema_prompt}\n\n## Question\n{question}"},
            ],
            config=config,
        )
        sql = _extract_sql(result.text)

        if not sql.strip():
            return Text2SQLResult(
                question=question,
                sql="",
                thinking=result.thinking,
                error="LLM returned empty response (token budget exhausted in reasoning?)",
            )

        return Text2SQLResult(
            question=question,
            sql=sql,
            thinking=result.thinking,
        )
    except Exception as exc:
        return Text2SQLResult(
            question=question,
            sql="",
            error=str(exc),
        )
