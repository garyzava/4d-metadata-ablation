"""Paths and constants for BIRD doc generation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source data
MINIDEV_DIR = PROJECT_ROOT / "mini_dev" / "minidev_data" / "minidev" / "MINIDEV"
DEV_TABLES_JSON = MINIDEV_DIR / "dev_tables.json"
QUESTIONS_JSON = MINIDEV_DIR / "mini_dev_postgresql.json"
DEV_DATABASES_DIR = MINIDEV_DIR / "dev_databases"

# Output
OUTPUT_DIR = PROJECT_ROOT / "content-out" / "bird-docs"
