#!/usr/bin/env python3
"""Revise BIRD-style documentation against an actual database schema.

Reads markdown docs from `--input-dir`, introspects a PostgreSQL database via
SQLAlchemy, and writes a revised copy of the docs to `--output-dir` with
identifier-like tokens (table names, column names) rewritten to the canonical
case stored in `information_schema`.

Safety guarantees:
  - Never writes to `--input-dir`. Enforced by resolved-path comparison.
  - Refuses to overwrite an existing `--output-dir` unless `--force` is given.
  - Only rewrites tokens that match an actual schema identifier case-insensitively
    and currently differ from the canonical case. Prose is untouched.
  - Verifies that every non-identifier byte of each revised file matches the
    corresponding byte of the input (checksum of the "uninteresting" content).
  - Emits both `_revision-report.json` (machine-readable) and
    `_revision-report.md` (human-readable) into `--output-dir`.

Usage:
    python scripts/revise_docs_against_schema.py \\
        --input-dir content-out/bird-docs \\
        --output-dir content-out/bird-docs-revised \\
        --database-url postgresql+psycopg://localhost/bird_minidev

    # Dry run (no files written):
    python scripts/revise_docs_against_schema.py \\
        --input-dir content-out/bird-docs \\
        --output-dir content-out/bird-docs-revised \\
        --database-url postgresql+psycopg://localhost/bird_minidev \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Identifier token: starts with letter or underscore, 3+ chars total, alphanumeric + underscore
# Word boundaries ensure we don't match substrings inside larger tokens
IDENT_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")

# In strict mode we only rewrite "compound identifiers" - tokens that are unambiguously code,
# not prose. A compound identifier has an underscore OR is CamelCase (>=2 uppercase letters,
# not purely uppercase). This excludes simple title-case English words like "Name", "Count",
# "Date" that happen to collide with column names.
_CAPS_RE = re.compile(r"[A-Z]")


def is_compound_identifier(token: str) -> bool:
    """True if `token` is unambiguously an identifier (not a prose word).

    Rules:
      - Contains at least one underscore in the middle (e.g. Team_Attributes, hero_id)
      - OR is CamelCase: >=2 uppercase letters AND not ALL uppercase
    """
    if "_" in token[1:-1] if len(token) > 2 else False:
        return True
    caps = _CAPS_RE.findall(token)
    if len(caps) >= 2 and token != token.upper():
        return True
    # ALL-UPPERCASE tokens like SEX, YEAR, COUNT are probably prose or SQL keywords - skip.
    return False


@dataclass
class Replacement:
    db: str
    file: str
    line: int
    col: int
    original: str
    canonical: str


@dataclass
class RevisionResult:
    replacements: list[Replacement] = field(default_factory=list)
    files_changed: set[tuple[str, str]] = field(default_factory=set)
    per_db_counts: Counter = field(default_factory=Counter)

    def add(self, r: Replacement) -> None:
        self.replacements.append(r)
        self.files_changed.add((r.db, r.file))
        self.per_db_counts[r.db] += 1


def build_schema_map(database_url: str, schema: str = "public") -> dict[str, str]:
    """Build a case-insensitive map of identifier.lower() -> canonical identifier.

    Includes both table names and column names across all tables in the given
    PostgreSQL schema. Fails loudly on case collisions (two different canonical
    names that lowercase to the same key).
    """
    from sqlalchemy import create_engine, inspect

    engine = create_engine(database_url)
    insp = inspect(engine)
    identifiers: dict[str, str] = {}
    collisions: dict[str, set[str]] = defaultdict(set)

    def register(name: str) -> None:
        key = name.lower()
        if key in identifiers and identifiers[key] != name:
            collisions[key].add(identifiers[key])
            collisions[key].add(name)
        else:
            identifiers[key] = name

    for table_name in insp.get_table_names(schema=schema):
        register(table_name)
        for col in insp.get_columns(table_name, schema=schema):
            register(col["name"])

    engine.dispose()

    if collisions:
        lines = ["Collisions detected (two canonical names lowercase to the same key):"]
        for key, names in collisions.items():
            lines.append(f"  {key!r}: {sorted(names)!r}")
        raise ValueError("\n".join(lines))

    return identifiers


def revise_text(
    text: str,
    schema_map: dict[str, str],
    mode: str = "strict",
) -> tuple[str, list[tuple[int, int, str, str]]]:
    """Revise text by replacing identifier tokens with canonical forms.

    Returns (revised_text, replacements) where each replacement is
    (line_1based, col_1based, original, canonical).

    mode="strict" (default): only rewrite compound identifiers (with underscore
        or CamelCase). Simple title-case words like "Name", "Count", "Date" are
        left alone even if they match a schema column case-insensitively - too
        risky as they often appear in prose.
    mode="permissive": rewrite any case-insensitive match.
    """
    replacements: list[tuple[int, int, str, str]] = []

    # Precompute line starts for mapping offsets to (line, col)
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def offset_to_line_col(offset: int) -> tuple[int, int]:
        # Binary search would be cleaner; linear is fine for these small files.
        line = 1
        for i in range(len(line_starts) - 1, -1, -1):
            if line_starts[i] <= offset:
                line = i + 1
                break
        col = offset - line_starts[line - 1] + 1
        return line, col

    def replace(match: re.Match) -> str:
        word = match.group()
        key = word.lower()
        canonical = schema_map.get(key)
        if canonical is None or canonical == word:
            return word
        if mode == "strict" and not is_compound_identifier(word):
            return word
        line, col = offset_to_line_col(match.start())
        replacements.append((line, col, word, canonical))
        return canonical

    revised = IDENT_TOKEN.sub(replace, text)
    return revised, replacements


def verify_non_identifier_bytes_match(original: str, revised: str) -> None:
    """Sanity check: everything EXCEPT identifier-token replacements should be byte-identical.

    Strategy: strip all identifier tokens from both strings, compare the remainder.
    """
    def strip_idents(s: str) -> str:
        return IDENT_TOKEN.sub("<IDENT>", s)

    left = strip_idents(original)
    right = strip_idents(revised)
    if left != right:
        # Find first difference
        for i, (a, b) in enumerate(zip(left, right)):
            if a != b:
                ctx_start = max(0, i - 40)
                raise AssertionError(
                    f"Non-identifier bytes differ at offset {i}:\n"
                    f"  original: {left[ctx_start:i + 40]!r}\n"
                    f"  revised:  {right[ctx_start:i + 40]!r}"
                )
        raise AssertionError(
            f"Non-identifier bytes differ in length: "
            f"original={len(left)}, revised={len(right)}"
        )


def revise_directory(
    input_dir: Path,
    output_dir: Path,
    schema_map: dict[str, str],
    dry_run: bool = False,
    mode: str = "strict",
) -> RevisionResult:
    result = RevisionResult()

    for db_dir in sorted(d for d in input_dir.iterdir() if d.is_dir()):
        db = db_dir.name
        out_db_dir = output_dir / db
        if not dry_run:
            out_db_dir.mkdir(parents=True, exist_ok=True)

        for md_path in sorted(db_dir.glob("*.md")):
            text = md_path.read_text()
            revised, reps = revise_text(text, schema_map, mode=mode)
            verify_non_identifier_bytes_match(text, revised)
            for line, col, original, canonical in reps:
                result.add(Replacement(db, md_path.name, line, col, original, canonical))
            out_path = out_db_dir / md_path.name
            if not dry_run:
                out_path.write_text(revised)

    return result


def write_reports(output_dir: Path, result: RevisionResult, schema_map_size: int, input_dir: Path) -> None:
    # JSON report
    json_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "schema_identifiers_count": schema_map_size,
        "files_changed": sorted(f"{db}/{f}" for (db, f) in result.files_changed),
        "per_db_replacement_counts": dict(result.per_db_counts),
        "total_replacements": len(result.replacements),
        "replacements": [
            {"db": r.db, "file": r.file, "line": r.line, "col": r.col,
             "original": r.original, "canonical": r.canonical}
            for r in result.replacements
        ],
    }
    (output_dir / "_revision-report.json").write_text(json.dumps(json_report, indent=2))

    # Markdown report
    md = [
        "# BIRD Docs Revision Report",
        "",
        f"- Generated: {json_report['generated_at']}",
        f"- Input: `{input_dir}`",
        f"- Output: `{output_dir}`",
        f"- Schema identifiers registered: {schema_map_size}",
        f"- Total replacements: {len(result.replacements)}",
        f"- Files changed: {len(result.files_changed)}",
        "",
        "## Replacements per database",
        "",
        "| Database | Replacements |",
        "|----------|-------------:|",
    ]
    for db, count in sorted(result.per_db_counts.items(), key=lambda x: (-x[1], x[0])):
        md.append(f"| {db} | {count} |")
    md.append("")
    md.append("## Distinct (original → canonical) pairs")
    md.append("")
    md.append("| Original | Canonical | Count |")
    md.append("|----------|-----------|------:|")
    pair_counts: Counter = Counter()
    for r in result.replacements:
        pair_counts[(r.original, r.canonical)] += 1
    for (orig, can), count in pair_counts.most_common(100):
        md.append(f"| `{orig}` | `{can}` | {count} |")
    (output_dir / "_revision-report.md").write_text("\n".join(md))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="Source docs directory (read-only)")
    p.add_argument("--output-dir", required=True, help="Destination for revised docs")
    p.add_argument("--database-url", required=True, help="PostgreSQL URL, e.g. postgresql+psycopg://localhost/bird_minidev")
    p.add_argument("--schema", default="public", help="PostgreSQL schema to introspect (default: public)")
    p.add_argument("--dry-run", action="store_true", help="Don't write any files; print what would change")
    p.add_argument("--force", action="store_true", help="Overwrite output-dir if it exists")
    p.add_argument(
        "--mode",
        choices=["strict", "permissive"],
        default="strict",
        help=(
            "strict (default): only rewrite compound identifiers (with underscore or CamelCase). "
            "permissive: rewrite any case-insensitive match, including simple title-case words. "
            "Strict avoids collateral prose rewrites like Name->name."
        ),
    )
    args = p.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Safety: distinct paths
    if input_dir == output_dir:
        print(f"ERROR: --input-dir and --output-dir resolve to the same path: {input_dir}", file=sys.stderr)
        return 2

    if not input_dir.is_dir():
        print(f"ERROR: --input-dir does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 2

    if output_dir.exists() and not args.dry_run:
        if args.force:
            # Still refuse if somehow output points inside input
            try:
                output_dir.relative_to(input_dir)
                print(f"ERROR: --output-dir is inside --input-dir; refusing even with --force", file=sys.stderr)
                return 2
            except ValueError:
                pass
            print(f"Removing existing output dir (--force): {output_dir}", file=sys.stderr)
            shutil.rmtree(output_dir)
        else:
            print(f"ERROR: --output-dir exists: {output_dir}. Use --force to overwrite.", file=sys.stderr)
            return 2

    print(f"Introspecting schema at {args.database_url}...", file=sys.stderr)
    schema_map = build_schema_map(args.database_url, schema=args.schema)
    print(f"Registered {len(schema_map)} identifiers from schema {args.schema!r}", file=sys.stderr)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    result = revise_directory(input_dir, output_dir, schema_map, dry_run=args.dry_run, mode=args.mode)

    label = f"(dry run, mode={args.mode})" if args.dry_run else f"(mode={args.mode})"
    print(f"\nResult {label}:")
    print(f"  Total replacements: {len(result.replacements)}")
    print(f"  Files changed: {len(result.files_changed)}")
    print(f"  Per-db: {dict(result.per_db_counts)}")

    if not args.dry_run:
        write_reports(output_dir, result, len(schema_map), input_dir)
        print(f"\nReports written to {output_dir}/_revision-report.{{json,md}}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
