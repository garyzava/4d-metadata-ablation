"""SQL guardrails: read-only enforcement, LIMIT injection, and statement validation."""

from __future__ import annotations

import re

from pydantic import BaseModel

# Patterns that indicate non-read-only operations
_FORBIDDEN_PATTERNS = [
    re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b", re.IGNORECASE),
    re.compile(r"\bINTO\s+\w+\s*\(", re.IGNORECASE),  # SELECT INTO
]

_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)

DEFAULT_MAX_ROWS = 500


class GuardrailResult(BaseModel):
    allowed: bool
    sql: str
    violations: list[str]


def check_sql(sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> GuardrailResult:
    """Validate SQL and enforce guardrails.

    Returns a GuardrailResult with the (possibly modified) SQL and any violations.
    """
    sql = sql.strip().rstrip(";")
    violations: list[str] = []

    if not sql:
        return GuardrailResult(allowed=False, sql=sql, violations=["Empty SQL statement."])

    for pattern in _FORBIDDEN_PATTERNS:
        match = pattern.search(sql)
        if match:
            violations.append(f"Forbidden keyword: {match.group(0).upper()}")

    if violations:
        return GuardrailResult(allowed=False, sql=sql, violations=violations)

    # Enforce LIMIT if not present
    if not _LIMIT_RE.search(sql):
        sql = f"{sql}\nLIMIT {max_rows}"

    return GuardrailResult(allowed=True, sql=sql, violations=[])
