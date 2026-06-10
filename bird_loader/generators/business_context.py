"""Generate BusinessContext.md - terms and synonyms."""

from bird_loader.loaders import get_col_info
from bird_loader.parsers import extract_terms


def generate(db: dict, questions: list, csv_descs: dict) -> str:
    cols = get_col_info(db)
    lines = [f"# Business Context: {db['db_id']}\n"]

    # Column synonyms
    lines.append("## Column Synonyms\n")
    lines.append("Mappings between original column names and human-readable names.\n")
    lines.append("| Original Column | Table | Human-Readable Name |")
    lines.append("|----------------|-------|-------------------|")
    has_synonyms = False
    for col in cols:
        if col["table"] and col["human_name"] != col["column"] and col["human_name"] != "*":
            lines.append(f"| {col['column']} | {col['table']} | {col['human_name']} |")
            has_synonyms = True
    if not has_synonyms:
        lines.append("| _(no differences)_ | | |")

    # Domain terms from evidence
    lines.append("\n## Domain Terms & Synonyms\n")
    lines.append("Terms extracted from question evidence/hints.\n")
    lines.append("| Term | Meaning |")
    lines.append("|------|---------|")
    seen: set[tuple[str, str]] = set()
    has_terms = False
    for q in questions:
        ev = q.get("evidence", "")
        if not ev:
            continue
        for term, meaning in extract_terms(ev):
            key = (term.lower(), meaning.lower())
            if key not in seen:
                seen.add(key)
                lines.append(f"| {term} | {meaning} |")
                has_terms = True
    if not has_terms:
        lines.append("| _(none found)_ | |")

    return "\n".join(lines) + "\n"
