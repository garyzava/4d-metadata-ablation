#!/usr/bin/env python3
"""Pre-compute gold SQL results for BIRD mini-dev benchmark.

Executes all gold SQL statements against the bird_minidev PostgreSQL database
and caches results to JSON for use by the experiment runner.

Usage:
    python precompute_gold.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

# This script lives at the public-repo root; engine is at src/ (overridable via env).
PROJECT_ROOT = Path(os.environ.get("GYZASQL_REPO_ROOT") or Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gyzasql.eval.generate import serialize_value  # noqa: E402

# Default matches notebook 01's dataset output (overridable by env).
BIRD_DATA_DIR = Path(os.environ.get("GYZASQL_BIRD_DATA_DIR", str(PROJECT_ROOT / "bird_mini_dev" / "minidev" / "MINIDEV")))
BIRD_CASES_PATH = BIRD_DATA_DIR / "mini_dev_postgresql.json"
GOLD_RESULTS_PATH = BIRD_DATA_DIR / "mini_dev_gold_results.json"
BIRD_CONNECTION_URL = os.environ.get("GYZASQL_BIRD_DATABASE_URL", "postgresql+psycopg://localhost/bird_minidev")


def precompute_gold() -> None:
    """Execute all BIRD gold SQL and cache results."""
    with open(BIRD_CASES_PATH) as f:
        cases = json.load(f)

    engine = create_engine(BIRD_CONNECTION_URL)
    gold_results: dict[str, dict] = {}
    errors = 0
    total = len(cases)

    with engine.connect() as conn:
        # Set 30s statement timeout (matching BIRD's meta_time_out)
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY"))
        conn.execute(text("SET statement_timeout = '30s'"))

        for i, case in enumerate(cases, 1):
            qid = str(case["question_id"])
            db_id = case["db_id"]
            gold_sql = case["SQL"]

            try:
                # Escape literal colons so SQLAlchemy's text() does not misread
                # them as bind parameters: gold SQL like `LIKE '_:%:__.___'`
                # contains `:__`, which text() would treat as a bind param named
                # `__` and fail with "value required for bind parameter". No gold
                # SQL uses real bind params, so escaping every colon is safe.
                # (exec_driver_sql is NOT usable here — psycopg then treats the
                # `%` in LIKE patterns as parameter markers.)
                result = conn.execute(text(gold_sql.replace(":", r"\:")))
                columns = list(result.keys())
                rows = [
                    {k: serialize_value(v) for k, v in zip(columns, row)}
                    for row in result.fetchall()
                ]
                gold_results[qid] = {
                    "db_id": db_id,
                    "rows": rows,
                    "row_count": len(rows),
                }
                print(f"  [{i}/{total}] q{qid} ({db_id}) -> {len(rows)} rows")

            except Exception as e:
                gold_results[qid] = {
                    "db_id": db_id,
                    "rows": None,
                    "error": str(e),
                }
                errors += 1
                print(f"  [{i}/{total}] q{qid} ({db_id}) -> ERROR: {e}", file=sys.stderr)

    GOLD_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLD_RESULTS_PATH, "w") as f:
        json.dump(gold_results, f, indent=2, default=str)

    print(f"\nDone: {total - errors}/{total} succeeded, {errors} errors")
    print(f"Saved to {GOLD_RESULTS_PATH}")


if __name__ == "__main__":
    precompute_gold()
