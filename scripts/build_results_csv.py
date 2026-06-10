"""Out-of-runner CSV exporter for all-results.csv.

The single canonical place that maps aggregate JSON fields to CSV columns.
Metrics: ER = execution rate (runs without error), BEX = BIRD strict set-equality.
Reads alias the ER aggregate keys to their legacy names so older result JSONs
still export correctly.

Critical invariant:
    CSV `gap_pp` column ← aggregate `er_bex_gap_pp` field (ER − BEX).

Usage (from this repo's root):
    python scripts/build_results_csv.py \\
        --results-dirs results/9b results/27b ... \\
        --output data/all-results.csv

The experiment runner writes one per-condition JSON per `(database x level x
model x rep)` to a results dir. This script walks them, lifts the aggregate
metrics + difficulty splits, and emits a flat 17-column CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "model", "model_full", "database", "level", "cases",
    "er_pct", "bex_pct", "vm_pct", "soft_f1",
    "gap_pp",          # ← MUST source from aggregate.er_bex_gap_pp
    "errors",
    "simple_er", "simple_bex",
    "moderate_er", "moderate_bex",
    "challenging_er", "challenging_bex",
]


# Map shortened model code -> full model name + version.
_MODEL_FULL = {
    "9B": "Qwen3.5-9B",
    "27B": "Qwen3.5-27B",
    "30B": "Qwen3-Coder-30B-A3B-Instruct",
    "QWEN36": "Qwen3.6-35B-A3B",
    "GEMMA4": "Gemma-4-26B-A4B-IT",
    "OPUS47": "Claude Opus 4.7",
}


def _model_code(record: dict, results_dir: Path) -> str:
    """Derive a short model code from the run record + dir name."""
    m = record.get("model")
    if isinstance(m, dict):
        m_id = m.get("model", "")
        if "9B" in m_id and "30B" not in m_id and "35B" not in m_id:
            return "9B"
        if "27B" in m_id:
            return "27B"
        if "Qwen3.6" in m_id or "35B" in m_id:
            return "QWEN36"
        if "30B" in m_id:
            return "30B"
        if "gemma" in m_id.lower():
            return "GEMMA4"
        if "opus" in m_id.lower() or "claude" in m_id.lower():
            return "OPUS47"
    # Fallback: derive from the parent dir name (9b -> 9B, gemma4 -> GEMMA4).
    name = results_dir.name.upper()
    return name


def _row_for(record: dict, results_dir: Path) -> dict:
    """Lift one (database × level × model × rep) JSON into a CSV row."""
    agg = record.get("aggregate_metrics", {})
    code = _model_code(record, results_dir)
    return {
        "model": code,
        "model_full": _MODEL_FULL.get(code, code),
        "database": record.get("dataset", ""),
        "level": record.get("level", ""),
        "cases": agg.get("total_cases", 0),
        # ER reads alias the new aggregate key then the legacy key (older JSONs).
        "er_pct": agg.get("execution_rate_pct", agg.get("execution_accuracy_pct", 0)),
        "bex_pct": agg.get("bird_ex_pct", 0),
        "vm_pct": agg.get("value_match_pct", 0),
        "soft_f1": agg.get("soft_f1_mean", 0),
        # CANONICAL: gap_pp = ER − BEX.
        "gap_pp": agg.get("er_bex_gap_pp", agg.get("ex_bex_gap_pp", 0)),
        "errors": agg.get("error_count", 0),
        "simple_er": agg.get("simple_er_pct", agg.get("simple_ex_pct", 0)),
        "simple_bex": agg.get("simple_bird_ex_pct", 0),
        "moderate_er": agg.get("moderate_er_pct", agg.get("moderate_ex_pct", 0)),
        "moderate_bex": agg.get("moderate_bird_ex_pct", 0),
        "challenging_er": agg.get("challenging_er_pct", agg.get("challenging_ex_pct", 0)),
        "challenging_bex": agg.get("challenging_bird_ex_pct", 0),
    }


def build(results_dirs: list[Path], output_path: Path) -> int:
    """Walk each results dir, lift aggregates, and write the flat CSV."""
    rows = []
    for d in results_dirs:
        if not d.exists():
            continue
        for jp in sorted(d.glob("*.json")):
            if jp.name == "experiment_summary.json":
                continue
            try:
                rec = json.loads(jp.read_text())
            except Exception:
                continue
            rows.append(_row_for(rec, d))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--results-dirs", nargs="+",
        default=[str(repo_root / "results" / f"{m}")
                 for m in ("9b", "27b", "30b", "gemma4", "qwen36", "opus47")],
        help="One or more results dirs to walk (default: <repo>/results/{9b,27b,30b,gemma4,qwen36,opus47}).",
    )
    p.add_argument(
        "--output",
        default=str(repo_root / "data" / "all-results.csv"),
        help="Output CSV path (default: <repo>/data/all-results.csv).",
    )
    args = p.parse_args()

    n = build([Path(d) for d in args.results_dirs], Path(args.output))
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
