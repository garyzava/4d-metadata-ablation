"""Generate DataDictionary.md - entity, column, descriptions and data types."""

from bird_loader.loaders import get_col_info


def generate(db: dict, questions: list, csv_descs: dict) -> str:
    cols = get_col_info(db)
    lines = [f"# Data Dictionary: {db['db_id']}\n"]

    for tbl_idx, tbl_name in enumerate(db["table_names_original"]):
        lines.append(f"\n## {tbl_name}\n")
        lines.append("| Column | Human-Readable Name | Data Type | Description | Value Notes |")
        lines.append("|--------|-------------------|-----------|-------------|-------------|")

        csv_rows = {}
        for row in csv_descs.get(tbl_name, []):
            csv_rows[row.get("original_column_name", "")] = row

        for col in cols:
            if col["table_idx"] != tbl_idx:
                continue
            csv_row = csv_rows.get(col["column"], {})
            desc = csv_row.get("column_description", "").strip()
            val_notes = csv_row.get("value_description", "").strip().replace("\n", " ").replace("|", "\\|")
            human = col["human_name"] if col["human_name"] != col["column"] else ""
            lines.append(f"| {col['column']} | {human} | {col['type']} | {desc} | {val_notes} |")

    return "\n".join(lines) + "\n"
