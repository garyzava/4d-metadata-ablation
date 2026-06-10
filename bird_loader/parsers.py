"""Evidence parsing helpers for BIRD question evidence fields."""

import re


def split_evidence(evidence: str) -> list[str]:
    """Split evidence string into individual clauses by semicolons."""
    return [p.strip() for p in re.split(r";\s*", evidence) if p.strip()]


def extract_terms(evidence: str) -> list[tuple[str, str]]:
    """Extract term -> meaning mappings from evidence.

    Looks for patterns like:
      "X refers to Y"
      "X means Y"
      "X can be represented as Y"
      "X is the Y"
    """
    terms = []
    for part in split_evidence(evidence):
        m = re.match(
            r"(.+?)\s+(?:refers?\s+to|means?|can\s+be\s+(?:represented|presented)\s+as|is\s+the)\s+(.+)",
            part,
            re.IGNORECASE,
        )
        if m:
            terms.append((m.group(1).strip(), m.group(2).strip()))
    return terms


def classify_evidence(evidence: str) -> dict[str, list[str]]:
    """Classify evidence parts into KPIs, date conventions, and value conventions."""
    result = {"kpis": [], "dates": [], "values": []}

    for part in split_evidence(evidence):
        if _is_kpi(part):
            result["kpis"].append(part)
        elif _is_date_convention(part):
            result["dates"].append(part)
        elif _is_value_convention(part):
            result["values"].append(part)

    return result


def _is_kpi(text: str) -> bool:
    return bool(
        re.search(
            r"(?:ratio|percentage|average|difference|DIVIDE|SUM|COUNT|AVG|MAX|MIN|"
            r"increase|decrease|deviation|calculat)",
            text,
            re.IGNORECASE,
        )
        or re.search(r"=\s*\S+.*(?:\*|/|\+|-|count|sum|avg)", text, re.IGNORECASE)
    )


def _is_date_convention(text: str) -> bool:
    return bool(
        re.search(
            r"(?:date|year|month|period|season|201[0-9]|SUBSTR|string)",
            text,
            re.IGNORECASE,
        )
    )


def _is_value_convention(text: str) -> bool:
    return bool(re.search(r"=\s*['\"]", text))
