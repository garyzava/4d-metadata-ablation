"""Aggregate per-case `status` distributions across one or more round directories.

`run_ablation.aggregate_results` (run_ablation.py:379-420) computes ER/BEX/VM/F1
but does NOT count statuses. The per-cell `status` field exists only inside
`per_case_results[*]`. This helper walks every per-case JSON in the given
round dirs and emits a markdown table of status distributions per
(model, db, level) so the operational error budget is visible without
manual JSON-walking.

Status values follow the taxonomy in `_classify_status`:
  ok / empty_response / empty_response_token_exhaustion /
  api_failure / timeout / parse_error / execution_error

Usage (from this repo's root):
    python scripts/status_budget_table.py results/9b
    python scripts/status_budget_table.py  # uses the default six * lanes

Optionally write the markdown table or a CSV to a file:
    python scripts/status_budget_table.py -o data/reports/status-budget.md
    python scripts/status_budget_table.py --csv data/status-budget.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Status values in stable display order. Anything else is bucketed as "other".
STATUS_ORDER = [
    "ok",
    "empty_response_token_exhaustion",
    "empty_response",
    "api_failure",
    "timeout",
    "parse_error",
    "execution_error",
]


def _walk_round_dir(round_dir: Path) -> list[dict[str, Any]]:
    """Yield one row per (round_dir, db, level) cell with its status counter."""
    if not round_dir.is_dir():
        print(f"WARN: {round_dir} is not a directory; skipping", file=sys.stderr)
        return []

    rows: list[dict[str, Any]] = []
    cell_files = sorted(round_dir.glob("*_env_rep1.json"))
    for f in cell_files:
        if f.name == "experiment_summary.json":
            continue
        try:
            with open(f) as fh:
                run = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: skipping {f.name}: {exc}", file=sys.stderr)
            continue

        cases = run.get("per_case_results", []) or []
        counter = Counter(c.get("status", "other") for c in cases)
        rows.append({
            "round_dir": round_dir.name,
            "dataset": run.get("dataset", "?"),
            "level": run.get("level", "?"),
            "n": len(cases),
            "counter": counter,
        })
    return rows


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    """Write the same status budget as CSV (one row per cell)."""
    if not rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("")
        return
    seen: list[str] = []
    for row in rows:
        for k in row["counter"]:
            if k not in STATUS_ORDER and k not in seen:
                seen.append(k)
    statuses = STATUS_ORDER + seen
    fieldnames = ["round", "database", "level", "n"] + statuses
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            r = {
                "round": row["round_dir"],
                "database": row["dataset"],
                "level": row["level"],
                "n": row["n"],
            }
            for s in statuses:
                r[s] = row["counter"].get(s, 0)
            w.writerow(r)


def _format_table(rows: list[dict[str, Any]]) -> str:
    """Render rows as a markdown table."""
    if not rows:
        return "_(no cells found)_\n"

    # Discover any extra status values not in STATUS_ORDER, in stable order
    seen: list[str] = []
    for row in rows:
        for k in row["counter"]:
            if k not in STATUS_ORDER and k not in seen:
                seen.append(k)
    statuses = STATUS_ORDER + seen

    header = ["round", "db", "level", "n"] + statuses
    sep = ["---"] * len(header)

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        cells: list[str] = [
            row["round_dir"],
            row["dataset"],
            row["level"],
            str(row["n"]),
        ]
        for s in statuses:
            cnt = row["counter"].get(s, 0)
            cells.append(str(cnt) if cnt else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "round_dirs", nargs="*", type=Path,
        default=[repo_root / "results" / f"{m}"
                 for m in ("9b", "27b", "30b", "gemma4", "qwen36", "opus47")],
        help="One or more round directories (default: <repo>/results/* lanes).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Optional output path; if omitted the table is printed to stdout.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional CSV output path. Independent of --output; both may be set.",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for d in args.round_dirs:
        all_rows.extend(_walk_round_dir(d))

    md = "# Status budget\n\n" + _format_table(all_rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Wrote {len(all_rows)} rows to {args.output}", file=sys.stderr)
    if args.csv is not None:
        _write_csv(all_rows, args.csv)
        print(f"Wrote {len(all_rows)} rows to {args.csv}", file=sys.stderr)
    if args.output is None and args.csv is None:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
