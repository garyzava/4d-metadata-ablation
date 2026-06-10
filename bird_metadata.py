"""Parse BIRD mini-dev documentation markdown into populate_* function arguments.

Each BIRD database has 4 markdown files in context/in/bird-docs/{db_name}/:
  - DataDictionary.md  → populate_data_dictionary() format
  - QueryPatterns.md   → populate_query_patterns() format
  - BusinessContext.md  → populate_business_context() format
  - DomainContext.md    → populate_domain_knowledge() format
"""

from __future__ import annotations

import re
from pathlib import Path


def _parse_md_table(lines: list[str]) -> list[dict[str, str]]:
    """Parse a markdown table into a list of dicts keyed by header names."""
    rows: list[dict[str, str]] = []
    headers: list[str] = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not headers:
            headers = cells
            continue
        # Skip separator row (|---|---|)
        if all(set(c.strip()) <= {"-", ":"} for c in cells):
            continue
        row = {h: cells[i] if i < len(cells) else "" for i, h in enumerate(headers)}
        rows.append(row)
    return rows


def parse_data_dictionary(md_path: Path) -> dict[str, str | dict[str, str]]:
    """Parse DataDictionary.md into populate_data_dictionary() format.

    Returns::

        {"table_name": {"_table": "", "col1": "description", ...}, ...}
    """
    text = md_path.read_text()
    result: dict[str, str | dict[str, str]] = {}

    # Split by ## headers (table sections)
    sections = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    # sections = ['preamble', 'table1', 'content1', 'table2', 'content2', ...]
    for i in range(1, len(sections), 2):
        table_name = sections[i].strip()
        content = sections[i + 1] if i + 1 < len(sections) else ""

        table_dict: dict[str, str] = {"_table": ""}
        rows = _parse_md_table(content.splitlines())
        for row in rows:
            col_name = row.get("Column", "").strip()
            if not col_name:
                continue
            desc = row.get("Description", "").strip()
            notes = row.get("Value Notes", "").strip()
            # Combine description and value notes
            full_desc = desc
            if notes:
                full_desc = f"{desc}; {notes}" if desc else notes
            table_dict[col_name] = full_desc

        result[table_name] = table_dict

    return result


def parse_query_patterns(md_path: Path) -> dict[str, list[str]]:
    """Parse QueryPatterns.md into populate_query_patterns() format.

    Returns::

        {"join_path": ["account.district_id → district.district_id: ...", ...]}
    """
    text = md_path.read_text()
    patterns: dict[str, list[str]] = {}

    # Extract Foreign Key Relationships section
    fk_section = ""
    if "## Foreign Key Relationships" in text:
        fk_section = text.split("## Foreign Key Relationships", 1)[1]

    # Parse each ### subsection as a join_path
    fk_parts = re.split(r"^### (.+)$", fk_section, flags=re.MULTILINE)
    for i in range(1, len(fk_parts), 2):
        header = fk_parts[i].strip()  # e.g. "account → district"
        content = fk_parts[i + 1] if i + 1 < len(fk_parts) else ""

        # Extract FK and reference info
        fk_match = re.search(r"\*\*FK Column:\*\*\s*`(.+?)`", content)
        ref_match = re.search(r"\*\*References:\*\*\s*`(.+?)`", content)

        # Extract SQL
        sql_match = re.search(r"```sql\s*\n(.+?)\n```", content, re.DOTALL)

        parts = []
        if fk_match and ref_match:
            parts.append(f"{fk_match.group(1)} → {ref_match.group(1)}")
        if sql_match:
            sql = sql_match.group(1).strip().replace("\n", " ")
            parts.append(sql)

        if parts:
            patterns.setdefault("join_path", []).append(": ".join(parts))

    return patterns


def parse_business_context(md_path: Path) -> dict[str, dict[str, str | list[str]]]:
    """Parse BusinessContext.md into populate_business_context() format.

    Returns::

        {"term": {"definition": "...", "synonyms": [...]}, ...}
    """
    text = md_path.read_text()
    terms: dict[str, dict[str, str | list[str]]] = {}

    # Parse Column Synonyms section
    if "## Column Synonyms" in text:
        syn_section = text.split("## Column Synonyms", 1)[1]
        # Cut at next ## section
        next_section = re.search(r"\n## ", syn_section)
        if next_section:
            syn_section = syn_section[: next_section.start()]

        rows = _parse_md_table(syn_section.splitlines())
        for row in rows:
            original = row.get("Original Column", "").strip()
            human = row.get("Human-Readable Name", "").strip()
            if original and human and original != human:
                # Use human-readable name as the term, original as synonym
                terms[human] = {
                    "definition": f"{original} column synonym",
                    "synonyms": [original],
                }

    # Parse Domain Terms section
    if "## Domain Terms" in text:
        dt_section = text.split("## Domain Terms", 1)[1]
        next_section = re.search(r"\n## ", dt_section)
        if next_section:
            dt_section = dt_section[: next_section.start()]

        rows = _parse_md_table(dt_section.splitlines())
        for row in rows:
            term = row.get("Term", "").strip()
            meaning = row.get("Meaning", "").strip()
            if term and meaning:
                # If term already exists from synonyms, merge
                if term in terms:
                    terms[term]["definition"] = meaning
                else:
                    terms[term] = {"definition": meaning, "synonyms": []}

    return terms


def parse_domain_knowledge(md_path: Path) -> dict[str, str]:
    """Parse DomainContext.md into populate_domain_knowledge() format.

    Returns::

        {"topic": "content", ...}
    """
    text = md_path.read_text()
    knowledge: dict[str, str] = {}

    # Parse bullet entries: - **topic**\n  - _From:_ question
    # Pattern: bold text on a line, optionally followed by a "From" line
    entries = re.findall(
        r"- \*\*(.+?)\*\*\s*\n\s*- _From:_ (.+?)(?:\n|$)",
        text,
    )
    for topic, source_question in entries:
        topic = topic.strip()
        if topic not in knowledge:  # deduplicate
            knowledge[topic] = f"From: {source_question.strip()}"

    # Also parse standalone entries without "From" lines (Date/Value conventions)
    # These appear as simple bullet points in some sections
    for section_name in ["## Date Conventions", "## Value Conventions"]:
        if section_name not in text:
            continue
        section = text.split(section_name, 1)[1]
        next_section = re.search(r"\n## ", section)
        if next_section:
            section = section[: next_section.start()]

        for line in section.splitlines():
            line = line.strip()
            if line.startswith("- ") and not line.startswith("- _From:_") and not line.startswith("- **"):
                entry = line[2:].strip()
                if entry and entry not in knowledge:
                    knowledge[entry] = section_name.replace("## ", "")

    return knowledge


def load_bird_metadata(db_name: str, bird_docs_dir: Path) -> dict:
    """Load all 4 metadata dimensions for a BIRD database.

    Returns dict with keys matching the dimension names used by LEVEL_DIMENSIONS
    in run_ablation.py.
    """
    db_dir = bird_docs_dir / db_name
    return {
        "data_dictionary": parse_data_dictionary(db_dir / "DataDictionary.md"),
        "query_patterns": parse_query_patterns(db_dir / "QueryPatterns.md"),
        "business_context": parse_business_context(db_dir / "BusinessContext.md"),
        "domain_knowledge": parse_domain_knowledge(db_dir / "DomainContext.md"),
    }
