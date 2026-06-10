"""Generate QueryPatterns.md - table joins (PK and FK)."""

from bird_loader.loaders import get_col_info


def generate(db: dict, questions: list, csv_descs: dict) -> str:
    cols = get_col_info(db)
    lines = [f"# Query Patterns: {db['db_id']}\n"]

    # Primary keys
    lines.append("## Primary Keys\n")
    lines.append("| Table | Primary Key Column(s) |")
    lines.append("|-------|----------------------|")
    pk_by_table: dict[str, list[str]] = {}
    for pk in db["primary_keys"]:
        if isinstance(pk, list):
            for p in pk:
                tbl = cols[p]["table"] or "?"
                pk_by_table.setdefault(tbl, []).append(cols[p]["column"])
        else:
            tbl = cols[pk]["table"] or "?"
            pk_by_table.setdefault(tbl, []).append(cols[pk]["column"])
    for tbl, pks in pk_by_table.items():
        lines.append(f"| {tbl} | {', '.join(pks)} |")

    # Foreign keys
    lines.append("\n## Foreign Key Relationships\n")
    if not db["foreign_keys"]:
        lines.append("No foreign keys defined.\n")
    else:
        for fk_from, fk_to in db["foreign_keys"]:
            from_col = cols[fk_from]
            to_col = cols[fk_to]
            lines.append(f"### {from_col['table']} → {to_col['table']}\n")
            lines.append(f"- **FK Column:** `{from_col['table']}.{from_col['column']}`")
            lines.append(f"- **References:** `{to_col['table']}.{to_col['column']}`")
            lines.append(f"\n```sql")
            lines.append(f"SELECT *")
            lines.append(f"FROM {from_col['table']}")
            lines.append(f"INNER JOIN {to_col['table']}")
            lines.append(f"  ON {from_col['table']}.{from_col['column']} = {to_col['table']}.{to_col['column']};")
            lines.append(f"```\n")

    return "\n".join(lines) + "\n"
