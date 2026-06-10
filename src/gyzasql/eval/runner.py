"""Eval runner: executes evaluation cases through the orchestrator and collects results."""

from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gyzasql.orchestrator.react_loop import ask


class EvalCase(BaseModel):
    question: str
    expected_sql: str | None = None
    expected_columns: list[str] | None = None
    expected_row_count: int | None = None
    expected_values: list[dict[str, Any]] | None = None
    expected_value_tolerance: float | None = None
    ignore_order: bool = False
    difficulty: str = "basic"
    tags: list[str] = []


class EvalResult(BaseModel):
    case: EvalCase
    actual_sql: str = ""
    actual_columns: list[str] = []
    actual_row_count: int = 0
    actual_rows: list[dict[str, Any]] = []
    execution_success: bool = False
    error: str | None = None
    metrics: dict[str, Any] = {}


def _values_equal(a: Any, b: Any, tolerance: float) -> bool:
    """Compare two values with fuzzy numeric matching (inspired by Spider 2.0)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), abs_tol=tolerance)
    return a == b


def _vectors_match(v1: list[Any], v2: list[Any], tolerance: float, ignore_order: bool) -> bool:
    """Compare two column vectors element-wise (mirrors Spider 2.0's vectors_match)."""
    if len(v1) != len(v2):
        return False

    if ignore_order:
        v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
        v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))

    return all(_values_equal(a, b, tolerance) for a, b in zip(v1, v2))


def _compare_values(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    tolerance: float = 1e-2,
    ignore_order: bool = False,
) -> bool:
    """Compare expected vs actual result rows using column-as-vector matching.

    Mirrors Spider 2.0's compare_pandas_table() approach:
    1. Transpose dict-rows into column vectors
    2. Each expected column vector must match at least one actual column vector
    3. Extra actual columns are ignored
    """
    if not expected or not actual:
        return not expected and not actual

    # Extract column vectors from expected rows
    expected_cols = list(expected[0].keys())
    expected_vectors: dict[str, list[Any]] = {col: [] for col in expected_cols}
    for row in expected:
        for col in expected_cols:
            expected_vectors[col].append(row.get(col))

    # Extract column vectors from actual rows
    if not actual:
        return False
    actual_cols = list(actual[0].keys())
    actual_vectors: dict[str, list[Any]] = {col: [] for col in actual_cols}
    for row in actual:
        for col in actual_cols:
            actual_vectors[col].append(row.get(col))

    # Each expected column vector must match at least one actual column vector
    actual_vector_list = list(actual_vectors.values())
    for exp_col in expected_cols:
        exp_vector = expected_vectors[exp_col]
        matched = any(
            _vectors_match(exp_vector, act_vector, tolerance, ignore_order) for act_vector in actual_vector_list
        )
        if not matched:
            return False

    return True


def _bird_ex_match(expected_rows: list[dict[str, Any]], actual_rows: list[dict[str, Any]]) -> bool:
    """BIRD-standard EX: unordered set comparison of result tuples.

    Matches official BIRD evaluation: set(predicted_res) == set(ground_truth_res).
    All values normalized to (float, str, None) to handle type coercion across
    database drivers (e.g. Decimal vs float, int vs float).
    """

    def _normalize_row(row: dict[str, Any]) -> tuple:
        vals = []
        for v in row.values():
            if isinstance(v, (int, float, Decimal)):
                vals.append(float(v))
            elif v is None:
                vals.append(None)
            else:
                vals.append(str(v))
        return tuple(vals)

    pred_set = set(_normalize_row(r) for r in actual_rows)
    gold_set = set(_normalize_row(r) for r in expected_rows)
    return pred_set == gold_set


def _rows_match(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    shared_cols: list[str],
    tolerance: float,
) -> bool:
    """Check if two rows match on all shared columns using fuzzy comparison."""
    return all(_values_equal(row_a.get(col), row_b.get(col), tolerance) for col in shared_cols)


def _compute_soft_f1(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    tolerance: float = 1e-2,
    ignore_order: bool = False,
) -> dict[str, float]:
    """Compute Soft-F1 over row-level overlap.

    Returns dict with keys: soft_f1, soft_f1_precision, soft_f1_recall.
    """
    if not expected and not actual:
        return {"soft_f1": 1.0, "soft_f1_precision": 1.0, "soft_f1_recall": 1.0}
    if not expected or not actual:
        return {"soft_f1": 0.0, "soft_f1_precision": 0.0, "soft_f1_recall": 0.0}

    # Use shared columns only (extra actual columns are ignored)
    expected_keys = set(expected[0].keys())
    actual_keys = set(actual[0].keys())
    shared_cols = sorted(expected_keys & actual_keys)

    if not shared_cols:
        return {"soft_f1": 0.0, "soft_f1_precision": 0.0, "soft_f1_recall": 0.0}

    if ignore_order:
        # Greedy bipartite matching
        used_actual: set[int] = set()
        tp = 0
        for exp_row in expected:
            for j, act_row in enumerate(actual):
                if j not in used_actual and _rows_match(exp_row, act_row, shared_cols, tolerance):
                    used_actual.add(j)
                    tp += 1
                    break
    else:
        # Positional matching
        tp = sum(_rows_match(exp_row, act_row, shared_cols, tolerance) for exp_row, act_row in zip(expected, actual))

    precision = tp / len(actual) if actual else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "soft_f1": round(f1, 6),
        "soft_f1_precision": round(precision, 6),
        "soft_f1_recall": round(recall, 6),
    }


def _compute_case_metrics(case: EvalCase, result: EvalResult) -> dict[str, Any]:
    """Compute per-case metrics by comparing expected vs actual."""
    metrics: dict[str, Any] = {}

    if case.expected_columns is not None:
        expected = set(case.expected_columns)
        actual = set(result.actual_columns)
        metrics["column_match"] = expected.issubset(actual)

    if case.expected_row_count is not None:
        metrics["row_count_match"] = result.actual_row_count == case.expected_row_count

    if case.expected_values is not None:
        tolerance = case.expected_value_tolerance or 1e-2
        if result.actual_rows:
            metrics["value_match"] = _compare_values(
                expected=case.expected_values,
                actual=result.actual_rows,
                tolerance=tolerance,
                ignore_order=case.ignore_order,
            )
            f1_metrics = _compute_soft_f1(
                expected=case.expected_values,
                actual=result.actual_rows,
                tolerance=tolerance,
                ignore_order=case.ignore_order,
            )
            metrics.update(f1_metrics)
            # BIRD-standard EX: set(pred_tuples) == set(gold_tuples)
            metrics["bird_ex"] = _bird_ex_match(case.expected_values, result.actual_rows)
        else:
            metrics["soft_f1"] = 0.0
            metrics["soft_f1_precision"] = 0.0
            metrics["soft_f1_recall"] = 0.0
            metrics["bird_ex"] = False

    return metrics


def run_eval(
    cases: list[EvalCase],
    dataset: str = "chinook",
    workspace: str | None = None,
    metadata_db: str | None = None,
    judge: bool = False,
) -> list[EvalResult]:
    """Run each eval case through the orchestrator and collect results."""
    results: list[EvalResult] = []

    for case in cases:
        orchestrator_result = ask(
            question=case.question,
            dataset=dataset,
            workspace=workspace,
            metadata_db=metadata_db,
        )

        eval_result = EvalResult(
            case=case,
            actual_sql=orchestrator_result.sql,
            actual_columns=[col for row in orchestrator_result.rows for col in row.keys()][:20]
            if orchestrator_result.rows
            else [],
            actual_row_count=orchestrator_result.row_count,
            actual_rows=orchestrator_result.rows,
            execution_success=orchestrator_result.error is None and orchestrator_result.row_count >= 0,
            error=orchestrator_result.error,
        )

        # Deduplicate columns
        if eval_result.actual_columns:
            seen: set[str] = set()
            unique: list[str] = []
            for c in eval_result.actual_columns:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            eval_result.actual_columns = unique

        eval_result.metrics = _compute_case_metrics(case, eval_result)

        if judge and case.expected_sql:
            from gyzasql.eval.llm_judge import judge_sql

            judge_result = judge_sql(
                question=case.question,
                gold_sql=case.expected_sql,
                predicted_sql=eval_result.actual_sql,
            )
            eval_result.metrics.update(judge_result)

        results.append(eval_result)

    return results


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [EvalCase(**item) for item in data]
