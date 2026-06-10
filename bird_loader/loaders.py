"""Data loaders for BIRD Mini-Dev source files."""

import csv
import json
from pathlib import Path

from bird_loader.config import DEV_DATABASES_DIR, DEV_TABLES_JSON, QUESTIONS_JSON


def load_tables() -> dict:
    """Load dev_tables.json and return a dict keyed by db_id."""
    with open(DEV_TABLES_JSON) as f:
        return {db["db_id"]: db for db in json.load(f)}


def load_questions() -> dict[str, list]:
    """Load questions JSON and group by db_id."""
    with open(QUESTIONS_JSON) as f:
        data = json.load(f)
    by_db: dict[str, list] = {}
    for q in data:
        by_db.setdefault(q["db_id"], []).append(q)
    return by_db


def load_csv_descriptions(db_name: str) -> dict[str, list[dict]]:
    """Load column description CSVs for a database.

    Returns a dict keyed by table name, each value is a list of row dicts.
    """
    desc_dir = DEV_DATABASES_DIR / db_name / "database_description"
    descs: dict[str, list[dict]] = {}
    if not desc_dir.exists():
        return descs
    for csv_file in desc_dir.glob("*.csv"):
        table_name = csv_file.stem
        rows = []
        with open(csv_file, encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        descs[table_name] = rows
    return descs


def get_col_info(db: dict) -> list[dict]:
    """Build a list of column info dicts from a database schema entry."""
    cols = []
    for i, (tbl_idx, col_name) in enumerate(db["column_names_original"]):
        human = db["column_names"][i][1] if i < len(db["column_names"]) else col_name
        ctype = db["column_types"][i] if i < len(db["column_types"]) else "unknown"
        tbl_name = db["table_names_original"][tbl_idx] if tbl_idx >= 0 else None
        cols.append(
            {
                "index": i,
                "table_idx": tbl_idx,
                "table": tbl_name,
                "column": col_name,
                "human_name": human,
                "type": ctype,
            }
        )
    return cols
