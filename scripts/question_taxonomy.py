"""Question-type and schema-complexity taxonomy for BIRD mini-dev.

Offline rules-based classifier on the gold SQL plus per-database schema metadata.
Cross-tabulates L0->L4 BEX lift by (question type, schema complexity)
to answer "where does metadata help most?".

Levels exported per lane:
  - HF lanes      (9b, 27b, 30b)            : EVIDENCE, L0, L0-PAD, L4, L4-DD, L4-QP, L4-BC, L4-DK
  - Scoped lanes  (gemma4, qwen36, opus47)  : L0, L0-PAD, L4

Every lane has native L0 and L4, so figures can compare L0 vs L4 directly for
all six models. The cross-tab summary in this script still uses its historical
baselines; the notebook figures use the native columns.

Outputs (relative to this repo):
  - data/reports/question-taxonomy.md
  - data/question-taxonomy.csv (one row per question)

Usage (from this repo's root):
  python scripts/question_taxonomy.py
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# BIRD source data lives outside the repo by default. Notebook 01 downloads BIRD mini-dev
# into <repo>/bird_mini_dev/minidev/MINIDEV/ and sets BIRD_MINIDEV_DIR to that path.
_default_bird_dir = REPO_ROOT / "bird_mini_dev" / "minidev" / "MINIDEV"
BIRD_MINIDEV_DIR = Path(os.environ.get("BIRD_MINIDEV_DIR", _default_bird_dir))
BIRD_CASES = BIRD_MINIDEV_DIR / "mini_dev_postgresql.json"
BIRD_TABLES = BIRD_MINIDEV_DIR / "dev_tables.json"
RESULTS_DIR = REPO_ROOT / "results"
REPORT_PATH = REPO_ROOT / "data" / "reports" / "question-taxonomy.md"
CSV_PATH = REPO_ROOT / "data" / "question-taxonomy.csv"

HF_MODELS = ["9b", "27b", "30b"]
SCOPED_MODELS = ["gemma4", "qwen36", "opus47"]
ALL_MODELS = HF_MODELS + SCOPED_MODELS

HF_LEVELS = ["EVIDENCE", "L0", "L0-PAD", "L4", "L4-DD", "L4-QP", "L4-BC", "L4-DK"]
SCOPED_LEVELS = ["L0", "L0-PAD", "L4"]
LEVELS_BY_MODEL: dict[str, list[str]] = {**{m: HF_LEVELS for m in HF_MODELS},
                                          **{m: SCOPED_LEVELS for m in SCOPED_MODELS}}

L4_LOO = ["L4-DD", "L4-QP", "L4-BC", "L4-DK"]


# Question-type classifier (rules on gold SQL).

RE_AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|TOTAL)\s*\(", re.IGNORECASE)
RE_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
RE_HAVING = re.compile(r"\bHAVING\b", re.IGNORECASE)
RE_JOIN = re.compile(r"\bJOIN\b", re.IGNORECASE)
RE_CTE = re.compile(r"\bWITH\s+\w+\s+AS\s*\(", re.IGNORECASE)
RE_SET_OP = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.IGNORECASE)
RE_WINDOW = re.compile(r"\b(OVER|ROW_NUMBER|RANK|DENSE_RANK|LAG|LEAD)\s*\(", re.IGNORECASE)
RE_CASE_WHEN = re.compile(r"\bCASE\s+WHEN\b", re.IGNORECASE)
RE_SUBQUERY = re.compile(r"\(\s*SELECT\b", re.IGNORECASE)


def classify_question_type(sql: str) -> dict[str, Any]:
    """Return the canonical primary type plus a multi-label set."""
    labels: list[str] = []

    if RE_SET_OP.search(sql):
        labels.append("set-op")
    if RE_CTE.search(sql):
        labels.append("cte")
    if RE_WINDOW.search(sql):
        labels.append("window")
    if RE_SUBQUERY.search(sql):
        labels.append("nested")
    join_count = len(RE_JOIN.findall(sql))
    if join_count >= 2:
        labels.append("multi-join")
    elif join_count == 1:
        labels.append("single-join")
    if RE_AGG.search(sql) or RE_GROUP_BY.search(sql) or RE_HAVING.search(sql):
        labels.append("aggregation")
    if RE_CASE_WHEN.search(sql):
        labels.append("case-when")

    priority = ("set-op", "cte", "window", "nested", "multi-join", "aggregation", "single-join")
    primary = "lookup"
    for p in priority:
        if p in labels:
            primary = p
            break

    return {"primary": primary, "labels": labels, "join_count": join_count}


def schema_complexity(schema: dict[str, Any]) -> dict[str, Any]:
    n_tables = len(schema["table_names_original"])
    n_fks = len(schema["foreign_keys"])
    n_cols = sum(1 for c in schema["column_names_original"] if c[0] != -1)

    if n_tables <= 4 and n_fks <= 3:
        bucket = "simple"
    elif n_tables >= 9 or n_fks >= 8:
        bucket = "complex"
    else:
        bucket = "moderate"

    return {"n_tables": n_tables, "n_fks": n_fks, "n_cols": n_cols, "bucket": bucket}


def load_questions() -> list[dict[str, Any]]:
    with open(BIRD_CASES) as f:
        return json.load(f)


def load_schemas() -> dict[str, dict[str, Any]]:
    with open(BIRD_TABLES) as f:
        schemas = json.load(f)
    return {s["db_id"]: s for s in schemas}


def load_per_case(model: str, level: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Load all cases for a given (model, level) keyed by (db_id, question)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for case_path in (RESULTS_DIR / f"{model}").glob(f"*_{level}_env_rep1.json"):
        db_id = case_path.name.replace(f"_{level}_env_rep1.json", "")
        with open(case_path) as f:
            r = json.load(f)
        for c in r.get("per_case_results", []):
            key = (db_id, c.get("question", ""))
            out[key] = c
    return out


def baseline_level(model: str) -> str:
    """L0 baseline used in cross-tabs: L0 for scoped lanes, L0-PAD for HF lanes."""
    return "L0" if model in SCOPED_MODELS else "L0-PAD"


def l4_effective_bex(case_map: dict[tuple[str, str], dict[str, Any]],
                     loo_maps: dict[str, dict[tuple[str, str], dict[str, Any]]],
                     model: str,
                     db_id: str,
                     question_text: str) -> list[int] | None:
    """Return a list of binary BEX values that stand in for the L4 condition.

    Scoped lanes: single value from native L4. HF lanes: up to four values from
    the L4 leave-one-out conditions (each LOO is one paired observation).
    """
    if model in SCOPED_MODELS:
        case = case_map.get((db_id, question_text))
        if case is None:
            return None
        return [int(bool(case.get("metrics", {}).get("bird_ex")))]
    # HF lane: pool the four LOO conditions
    vals: list[int] = []
    for loo_level, m in loo_maps.items():
        case = m.get((db_id, question_text))
        if case is None:
            continue
        vals.append(int(bool(case.get("metrics", {}).get("bird_ex"))))
    return vals or None


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.1f}"


def render_cross_tab(summary_dim: dict[str, dict[str, list[int]]],
                     dim_label: str,
                     order: list[str] | None = None) -> str:
    out = []
    out.append(f"| {dim_label} | n_pairs | L0 BEX% | L4 BEX% | delta (pp) |")
    out.append("|---|---:|---:|---:|---:|")
    keys = order or sorted(summary_dim.keys())
    for key in keys:
        if key not in summary_dim:
            continue
        l0 = summary_dim[key].get("L0", [])
        l4 = summary_dim[key].get("L4", [])
        if not l0 or not l4:
            continue
        l0_mean = mean(l0)
        l4_mean = mean(l4)
        delta = (l4_mean - l0_mean) * 100
        out.append(f"| {key} | {len(l4)} | {fmt_pct(l0_mean)} | {fmt_pct(l4_mean)} | {delta:+.1f} |")
    return "\n".join(out)


def main() -> None:
    if not BIRD_CASES.exists():
        raise SystemExit(
            "BIRD mini-dev source data not found at "
            f"{BIRD_MINIDEV_DIR}\n"
            "Run notebook 01 (notebooks/01-reproducibility-pipeline.ipynb) to download "
            "the BIRD distribution, or set the BIRD_MINIDEV_DIR environment variable to "
            "an existing MINIDEV/ directory."
        )
    print("Loading BIRD questions + schemas...")
    questions = load_questions()
    schemas = load_schemas()
    print(f"  questions: {len(questions)}")
    print(f"  schemas: {len(schemas)} dbs")

    print("Loading per-case JSONs (6 models x lane-specific levels)...")
    data: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = {}
    for model in ALL_MODELS:
        for level in LEVELS_BY_MODEL[model]:
            data[(model, level)] = load_per_case(model, level)
            print(f"  {model} {level}: {len(data[(model, level)])} cases")

    print("Classifying + cross-tabulating...")
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, dict[str, list[int]]]] = {
        "by_type": defaultdict(lambda: defaultdict(list)),
        "by_complexity": defaultdict(lambda: defaultdict(list)),
        "by_difficulty": defaultdict(lambda: defaultdict(list)),
    }

    for q in questions:
        db_id = q["db_id"]
        gold_sql = q["SQL"]
        question_text = q["question"]
        difficulty = q.get("difficulty", "?")
        question_id = q.get("question_id")
        qtype = classify_question_type(gold_sql)
        complexity = schema_complexity(schemas[db_id])

        per_model: dict[str, dict[str, Any]] = {}
        for model in ALL_MODELS:
            row_model: dict[str, Any] = {}
            for level in LEVELS_BY_MODEL[model]:
                case = data[(model, level)].get((db_id, question_text))
                if case is None:
                    row_model[f"{level}_bex"] = None
                    row_model[f"{level}_er"] = None
                else:
                    metrics = case.get("metrics", {})
                    row_model[f"{level}_bex"] = bool(metrics.get("bird_ex"))
                    row_model[f"{level}_er"] = bool(case.get("execution_success"))
            per_model[model] = row_model

            # Contribute to cross-tab summary.
            base_lvl = baseline_level(model)
            base_bex = per_model[model].get(f"{base_lvl}_bex")
            if base_bex is None:
                continue
            if model in SCOPED_MODELS:
                l4_vals = [per_model[model].get("L4_bex")]
                l4_vals = [int(v) for v in l4_vals if v is not None]
            else:
                l4_vals = []
                for loo in L4_LOO:
                    v = per_model[model].get(f"{loo}_bex")
                    if v is not None:
                        l4_vals.append(int(v))
            if not l4_vals:
                continue
            # Each L4 sample paired with the same baseline observation.
            for v in l4_vals:
                summary["by_type"][qtype["primary"]]["L0"].append(int(base_bex))
                summary["by_type"][qtype["primary"]]["L4"].append(v)
                summary["by_complexity"][complexity["bucket"]]["L0"].append(int(base_bex))
                summary["by_complexity"][complexity["bucket"]]["L4"].append(v)
                summary["by_difficulty"][difficulty]["L0"].append(int(base_bex))
                summary["by_difficulty"][difficulty]["L4"].append(v)

        rows.append({
            "question_id": question_id,
            "db_id": db_id,
            "difficulty": difficulty,
            "question_type_primary": qtype["primary"],
            "question_type_labels": "|".join(sorted(qtype["labels"])) or "lookup",
            "join_count": qtype["join_count"],
            "n_tables": complexity["n_tables"],
            "n_fks": complexity["n_fks"],
            "n_cols": complexity["n_cols"],
            "complexity_bucket": complexity["bucket"],
            "per_model": per_model,
        })

    # Distribution checks
    type_dist = Counter(r["question_type_primary"] for r in rows)
    complexity_dist = Counter(r["complexity_bucket"] for r in rows)
    print(f"\nQuestion type distribution: {dict(type_dist)}")
    print(f"Complexity distribution: {dict(complexity_dist)}")

    print("\nPer-DB schema complexity:")
    for db_id in sorted(schemas):
        c = schema_complexity(schemas[db_id])
        print(f"  {db_id:<28} tables={c['n_tables']:<3} fks={c['n_fks']:<3} cols={c['n_cols']:<4} bucket={c['bucket']}")

    # Write per-question CSV
    print(f"\nWriting per-question CSV to {CSV_PATH}...")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["question_id", "db_id", "difficulty", "question_type_primary",
                  "question_type_labels", "join_count", "n_tables", "n_fks", "n_cols",
                  "complexity_bucket"]
        for model in ALL_MODELS:
            for level in LEVELS_BY_MODEL[model]:
                header += [f"{model}_{level}_bex", f"{model}_{level}_er"]
        writer.writerow(header)
        for r in rows:
            row = [r["question_id"], r["db_id"], r["difficulty"], r["question_type_primary"],
                   r["question_type_labels"], r["join_count"], r["n_tables"], r["n_fks"], r["n_cols"],
                   r["complexity_bucket"]]
            for model in ALL_MODELS:
                for level in LEVELS_BY_MODEL[model]:
                    bex = r["per_model"][model].get(f"{level}_bex")
                    er = r["per_model"][model].get(f"{level}_er")
                    row += ["" if bex is None else int(bex), "" if er is None else int(er)]
            writer.writerow(row)

    type_order = ["lookup", "single-join", "multi-join", "aggregation", "nested",
                  "case-when", "cte", "set-op", "window"]
    complexity_order = ["simple", "moderate", "complex"]
    difficulty_order = ["simple", "moderate", "challenging"]

    type_table = render_cross_tab(summary["by_type"], "Question type", type_order)
    complexity_table = render_cross_tab(summary["by_complexity"], "Schema complexity", complexity_order)
    difficulty_table = render_cross_tab(summary["by_difficulty"], "BIRD difficulty", difficulty_order)

    # Write report
    md_parts = [
        "# Question-type and schema-complexity taxonomy",
        "",
        "Generated by `scripts/question_taxonomy.py`. The classifier is "
        "rules-based on gold SQL plus per-database schema metadata.",
        "",
        f"## Question-type distribution (n = {len(rows)})",
        "",
        "| Type | Count |",
        "|---|---:|",
    ]
    for k in type_order:
        md_parts.append(f"| {k} | {type_dist.get(k, 0)} |")
    md_parts += [
        "",
        f"## Schema-complexity distribution",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for k in complexity_order:
        md_parts.append(f"| {k} | {complexity_dist.get(k, 0)} |")
    md_parts += [
        "",
        "## Per-database schema complexity",
        "",
        "| Database | tables | fks | cols | bucket |",
        "|---|---:|---:|---:|---|",
    ]
    for db_id in sorted(schemas):
        c = schema_complexity(schemas[db_id])
        md_parts.append(f"| {db_id} | {c['n_tables']} | {c['n_fks']} | {c['n_cols']} | {c['bucket']} |")
    md_parts += [
        "",
        "## Cross-tab: L0 vs L4 BEX by question type",
        "",
        "Pairs each question's baseline (L0 for scoped lanes, L0-PAD for HF lanes) with "
        "its L4 outcome (native L4 for scoped, each of the four L4 leave-one-out "
        "conditions for HF) across all six models.",
        "",
        type_table,
        "",
        "## Cross-tab: L0 vs L4 BEX by schema complexity",
        "",
        complexity_table,
        "",
        "## Cross-tab: L0 vs L4 BEX by BIRD difficulty",
        "",
        difficulty_table,
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(md_parts))
    print(f"Wrote {REPORT_PATH}")

    print("\n" + "=" * 80)
    print("CROSS-TAB BY QUESTION TYPE")
    print("=" * 80)
    print(type_table)
    print("\n" + "=" * 80)
    print("CROSS-TAB BY SCHEMA COMPLEXITY")
    print("=" * 80)
    print(complexity_table)
    print("\n" + "=" * 80)
    print("CROSS-TAB BY BIRD DIFFICULTY")
    print("=" * 80)
    print(difficulty_table)


if __name__ == "__main__":
    main()
