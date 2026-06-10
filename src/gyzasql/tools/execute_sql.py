"""Execute SQL tool: runs guarded, read-only queries and returns structured results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from gyzasql.db.connectors.postgres import readonly_connection
from gyzasql.governance.guardrails import GuardrailResult, check_sql


class QueryResult(BaseModel):
    sql: str
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    row_count: int = 0
    error: str | None = None
    guardrail_violations: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def execute_sql(engine: Engine, sql: str, max_rows: int = 500, db_type: str = "postgres") -> QueryResult:
    """Check guardrails, execute SQL, and return structured results."""
    check: GuardrailResult = check_sql(sql, max_rows=max_rows)

    if not check.allowed:
        # Empty SQL is a model failure, not a security block — surface it as such.
        if check.violations == ["Empty SQL statement."]:
            error = "text2sql produced empty SQL — model likely exhausted its token budget in reasoning."
        else:
            error = "Query blocked by guardrails."
        return QueryResult(
            sql=sql,
            error=error,
            guardrail_violations=check.violations,
        )

    # Escape literal colons so SQLAlchemy's text() does not misread them as bind
    # parameters: SQL like `LIKE '_:%:__.___'` contains `:__`, which text() would
    # treat as a bind param named `__` and fail with "value required for bind
    # parameter". `\:` renders back to a literal `:` (and `::` casts survive), and
    # generated SQL never uses real bind params, so escaping every colon is safe.
    # This keeps the candidate path symmetric with precompute_gold.py. NOTE: this
    # diverges from the upstream vendored engine (which lacked the escape), and is
    # intentional, so a valid colon-in-literal candidate executes instead of
    # spuriously failing to run.
    sql_to_run = check.sql.replace(":", r"\:")
    try:
        if db_type == "postgres":
            with readonly_connection(engine) as conn:
                result = conn.execute(text(sql_to_run))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
        else:
            with engine.connect() as conn:
                result = conn.execute(text(sql_to_run))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]

        return QueryResult(
            sql=check.sql,
            rows=rows,
            columns=columns,
            row_count=len(rows),
        )
    except Exception as exc:
        return QueryResult(sql=check.sql, error=str(exc))
