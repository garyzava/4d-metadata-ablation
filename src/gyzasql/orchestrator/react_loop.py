"""Agentic orchestrator: coordinates schema lookup, text2sql, guardrails, and execution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy.engine import Engine

from gyzasql.db.connectors import create_engine_for_dataset
from gyzasql.observability.tracing import get_tracer
from gyzasql.semantic_layer.context import get_context, get_workspace_context
from gyzasql.semantic_layer.schema_filter import filter_context
from gyzasql.semantic_layer.store import MetadataStore
from gyzasql.tools.execute_sql import QueryResult, execute_sql
from gyzasql.tools.text2sql import Text2SQLResult, generate_sql

# This vendored copy drops semantic_layer.scoring.compute_completeness (gyzasql IP, not a paper
# metric) and tools.retrieve_docs (RAG, Qdrant disabled in this experiment). See _VENDORED.md.


class Step(BaseModel):
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]


class OrchestratorResult(BaseModel):
    question: str
    answer: str = ""
    sql: str = ""
    rows: list[dict[str, Any]] = []
    row_count: int = 0
    tables_used: list[str] = []
    warnings: list[str] = []
    steps: list[Step] = []
    needs_clarification: bool = False
    clarification_message: str = ""
    needs_approval: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# Questions that are too vague to answer without clarification
_VAGUE_INDICATORS = [
    lambda q: len(q.split()) <= 2,
]


def _needs_clarification(question: str) -> str | None:
    """Return a clarification message if the question is too vague, else None.

    Vendored-copy note: upstream gyzasql also returns a clarification when the
    completeness_score is below GYZASQL_MIN_COMPLETENESS. That branch was
    removed here because compute_completeness is not vendored (gyzasql IP, not
    a paper metric). The L0 ablation is intentionally bare-schema, so a low
    completeness score is the *experimental condition*, not a clarification
    trigger.
    """
    stripped = question.strip()
    if not stripped:
        return "Please provide a question about the database."

    if any(check(stripped) for check in _VAGUE_INDICATORS):
        return (
            f'Your question "{stripped}" is quite short. '
            "Could you provide more detail about what you'd like to know? "
            "For example, specify which tables, metrics, or filters you're interested in."
        )

    return None


def ask(
    question: str,
    dataset: str = "chinook",
    workspace: str | None = None,
    metadata_db: str | None = None,
    connection_url: str | None = None,
    hitl: bool = False,
    approved_sql: str | None = None,
) -> OrchestratorResult:
    """Full orchestration: question → context → text2sql → guardrails → execute → answer.

    This is the main entry point for the agentic loop.

    Args:
        workspace: Optional workspace name. When provided, merges context from
            all datasets in the workspace for richer LLM prompts.
        hitl: When True, return generated SQL for approval without executing.
        approved_sql: Pre-approved SQL to execute directly (skips text2sql).
    """
    db_path = metadata_db or os.environ.get("GYZASQL_METADATA_DB", "gyzasql_metadata.db")
    store = MetadataStore(db_path=Path(db_path))
    result = OrchestratorResult(question=question)
    tracer = get_tracer()

    with tracer.start_as_current_span("ask") as span:
        span.set_attribute("gyzasql.question", question)
        span.set_attribute("gyzasql.dataset", dataset)

        try:
            # Step 1: Resolve dataset
            dataset_id = store.get_dataset_id(dataset, workspace=workspace)
            if dataset_id is None:
                result.error = f"Dataset '{dataset}' not found. Run introspect first."
                return result

            # Step 2: Get context, filter by relevance.
            # (Vendored-copy note: upstream also runs compute_completeness here and
            # populates result.completeness_score / result.warnings. Removed because
            # compute_completeness is gyzasql IP and not a paper metric. Also removed:
            # the GYZASQL_QDRANT_PATH/retrieve_docs RAG-augmentation block — Qdrant is
            # disabled in this experiment. See src/gyzasql/_VENDORED.md.)
            with tracer.start_as_current_span("get_context"):
                if workspace:
                    workspace_id = store.get_workspace_id(workspace)
                    if workspace_id is None:
                        result.error = f"Workspace '{workspace}' not found."
                        return result
                    context = get_workspace_context(store, workspace_id, dataset_id)
                else:
                    context = get_context(store, dataset_id)
                context = filter_context(context, question)
            result.tables_used = [t["name"] for t in context.get("tables", [])]

            result.steps.append(
                Step(
                    tool="get_context",
                    input={"dataset": dataset},
                    output={"table_count": len(result.tables_used)},
                )
            )

            # Step 3: Check if clarification is needed
            clarification = _needs_clarification(question)
            if clarification:
                result.needs_clarification = True
                result.clarification_message = clarification
                result.steps.append(
                    Step(
                        tool="clarify",
                        input={"question": question},
                        output={"message": clarification},
                    )
                )
                return result

            # Step 4: Generate SQL (or use approved_sql)
            if approved_sql:
                result.sql = approved_sql
                result.steps.append(
                    Step(
                        tool="hitl_approve",
                        input={"approved_sql": approved_sql},
                        output={"sql": approved_sql},
                    )
                )
            else:
                with tracer.start_as_current_span("text2sql"):
                    text2sql_result: Text2SQLResult = generate_sql(store, dataset_id, question)
                result.steps.append(
                    Step(
                        tool="text2sql",
                        input={"question": question},
                        output=text2sql_result.to_dict(),
                    )
                )

                if text2sql_result.error:
                    result.error = f"text2sql failed: {text2sql_result.error}"
                    return result

                result.sql = text2sql_result.sql

            # Step 4b: HITL — return SQL for approval without executing
            if hitl and not approved_sql:
                result.needs_approval = True
                return result

            # Step 5: Execute SQL
            row = store.conn.execute("SELECT connection_ref FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            db_type = store.get_db_type(dataset_id)
            conn_ref = connection_url or (row["connection_ref"] if row else None)
            engine: Engine = create_engine_for_dataset(conn_ref, db_type)

            with tracer.start_as_current_span("execute_sql"):
                query_result: QueryResult = execute_sql(engine, result.sql, db_type=db_type)
            result.steps.append(
                Step(
                    tool="execute_sql",
                    input={"sql": result.sql},
                    output=query_result.to_dict(),
                )
            )

            if query_result.error:
                result.error = f"Execution failed: {query_result.error}"
                if query_result.guardrail_violations:
                    result.warnings.extend(query_result.guardrail_violations)
                return result

            result.rows = query_result.rows
            result.row_count = query_result.row_count
            result.sql = query_result.sql  # may have LIMIT injected

            # Step 6: Build answer summary
            if result.row_count == 0:
                result.answer = "The query returned no results."
            elif result.row_count == 1 and len(query_result.columns) == 1:
                val = list(result.rows[0].values())[0]
                result.answer = f"{val}"
            else:
                result.answer = f"Returned {result.row_count} row(s) with columns: {', '.join(query_result.columns)}."

            span.set_attribute("gyzasql.row_count", result.row_count)
            return result

        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            result.error = str(exc)
            return result
        finally:
            store.close()
