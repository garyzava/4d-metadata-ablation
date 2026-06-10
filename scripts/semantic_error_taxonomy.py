#!/usr/bin/env python3
"""Semantic error taxonomy over the per-case result JSONs.

Extracts every value-incorrect case (BEX = 0) and categorises its failure mode by
comparing gold vs predicted SQL, to test whether specific metadata dimensions fix
specific failure modes (Query Patterns -> joins, Business Context -> value-literals /
casing, Domain Knowledge -> KPI/aggregation). The heuristic `category` is a
first-pass signal meant to be hand-reviewed; rows with an unclear mode are left blank.

Usage (from this repo's root):
    uv run python scripts/semantic_error_taxonomy.py
    uv run python scripts/semantic_error_taxonomy.py --results-dir results --output data/semantic-errors.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CATEGORIES = (
    "wrong_join", "wrong_aggregation", "wrong_value_literal", "wrong_casing",
    "missing_group_by", "wrong_table", "wrong_ordering_limit", "semantically_equivalent",
)


def _tables(sql_lower: str) -> set[str]:
    """Target tables named right after FROM / JOIN (quotes stripped, lowercased)."""
    return set(re.findall(r'\b(?:from|join)\s+"?([a-z_]\w*)"?', sql_lower))


def _str_literals(sql: str) -> set[str]:
    """Single-quoted string literals, ORIGINAL case (Postgres: '...' is a value)."""
    return set(re.findall(r"'([^']*)'", sql))


def _dq_idents(sql: str) -> set[str]:
    """Double-quoted identifiers, ORIGINAL case (Postgres: \"...\" is an identifier)."""
    return set(re.findall(r'"([^"]*)"', sql))


def _heuristic_category(gold: str, pred: str) -> str:
    """Cheap first-pass signal; blank when unclear (review by hand).

    Keeps original-case SQL for literal/identifier comparison (casing is the point)
    and a lowercased copy for keyword checks. Categories applied in priority order;
    `semantically_equivalent` is intentionally left undetected (near-empty for BEX=0).
    """
    gold, pred = gold or "", pred or ""
    if not pred.strip():
        return ""
    g, p = gold.lower(), pred.lower()  # keyword checks on lowercased copies
    # --- structural ---
    if "join" in g and "join" not in p:
        return "wrong_join"
    gt, pt = _tables(g), _tables(p)
    if gt and pt and gt != pt:
        return "wrong_table"
    if "group by" in g and "group by" not in p:
        return "missing_group_by"
    gagg = set(re.findall(r"\b(count|sum|avg|min|max)\b", g))
    pagg = set(re.findall(r"\b(count|sum|avg|min|max)\b", p))
    if gagg != pagg:
        return "wrong_aggregation"
    # --- value-level (case-sensitive: compare on original-case SQL) ---
    gl, pl = _str_literals(gold), _str_literals(pred)
    if gl != pl:
        if {x.lower() for x in gl} == {x.lower() for x in pl}:
            return "wrong_casing"        # same literals, different case
        return "wrong_value_literal"     # different literal content
    gi, pi = _dq_idents(gold), _dq_idents(pred)
    if gi != pi and {x.lower() for x in gi} == {x.lower() for x in pi}:
        return "wrong_casing"            # identifier case mismatch only
    if ("order by" in g) != ("order by" in p) or ("limit" in g) != ("limit" in p):
        return "wrong_ordering_limit"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results-dir", default=str(REPO / "results"), help="Dir of per-model result subdirs")
    ap.add_argument("--output", default=str(REPO / "data" / "semantic-errors.csv"))
    args = ap.parse_args()

    rows: list[dict] = []
    for mdir in sorted(Path(args.results_dir).glob("*")):
        if not mdir.is_dir():
            continue
        for jp in sorted(mdir.glob("*.json")):
            if jp.name == "experiment_summary.json":
                continue
            try:
                rec = json.loads(jp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            model, db, level = mdir.name, rec.get("dataset", ""), rec.get("level", "")
            for i, c in enumerate(rec.get("per_case_results", [])):
                metrics = c.get("metrics", {}) or {}
                if metrics.get("bird_ex"):
                    continue  # value-correct -> not a failure
                gold = c.get("expected_sql") or c.get("gold_sql") or ""
                pred = c.get("actual_sql") or c.get("generated_sql") or ""
                rows.append({
                    "model": model, "database": db, "level": level,
                    "question_id": c.get("question_id", c.get("case_index", i)),
                    "status": c.get("status", ""),
                    "execution_success": c.get("execution_success", ""),
                    "gold_sql": gold, "predicted_sql": pred,
                    "category": _heuristic_category(gold, pred),
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "database", "level", "question_id", "status",
            "execution_success", "gold_sql", "predicted_sql", "category"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} value-incorrect (BEX=0) rows to {out}")
    print("category tally:", dict(Counter(r["category"] or "(uncoded)" for r in rows)))


if __name__ == "__main__":
    main()
