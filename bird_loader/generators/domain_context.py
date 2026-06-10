"""Generate DomainContext.md - KPIs and conventions."""

from bird_loader.parsers import classify_evidence


def generate(db: dict, questions: list, csv_descs: dict) -> str:
    lines = [f"# Domain Context: {db['db_id']}\n"]

    all_kpis: list[tuple[str, str]] = []
    all_dates: list[str] = []
    all_values: list[str] = []

    for q in questions:
        ev = q.get("evidence", "")
        if not ev:
            continue
        classified = classify_evidence(ev)
        for kpi in classified["kpis"]:
            all_kpis.append((kpi, q["question"]))
        all_dates.extend(classified["dates"])
        all_values.extend(classified["values"])

    # KPIs & Calculations
    lines.append("## KPIs & Calculations\n")
    if all_kpis:
        seen: set[str] = set()
        for formula, question in all_kpis:
            if formula.lower() not in seen:
                seen.add(formula.lower())
                lines.append(f"- **{formula}**")
                lines.append(f"  - _From:_ {question}\n")
    else:
        lines.append("No KPIs or calculations found in evidence.\n")

    # Date Conventions
    lines.append("\n## Date Conventions\n")
    if all_dates:
        seen_d: set[str] = set()
        for conv in all_dates:
            if conv.lower() not in seen_d:
                seen_d.add(conv.lower())
                lines.append(f"- {conv}")
    else:
        lines.append("No date conventions found in evidence.\n")

    # Value Conventions
    lines.append("\n## Value Conventions\n")
    if all_values:
        seen_v: set[str] = set()
        for conv in all_values:
            if conv.lower() not in seen_v:
                seen_v.add(conv.lower())
                lines.append(f"- {conv}")
    else:
        lines.append("No value conventions found in evidence.\n")

    return "\n".join(lines) + "\n"
