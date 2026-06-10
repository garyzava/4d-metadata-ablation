"""Schema filtering: select only question-relevant tables and metadata for the LLM prompt."""

from __future__ import annotations

import json
import re

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "not",
        "all",
        "each",
        "every",
        "how",
        "many",
        "much",
        "what",
        "which",
        "who",
        "where",
        "when",
        "show",
        "me",
        "get",
        "find",
        "list",
        "give",
        "tell",
        "display",
        "return",
        "from",
        "by",
        "with",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "be",
        "been",
        "can",
        "could",
        "would",
        "should",
        "will",
        "i",
        "my",
        "their",
        "our",
        "your",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "there",
        "than",
        "then",
        "so",
        "if",
        "no",
        "yes",
        "any",
        "some",
        "about",
        "into",
        "up",
        "out",
        "as",
        "also",
        "just",
        "only",
        "very",
        "more",
        "most",
        "other",
    }
)

# Scoring weights for table relevance ranking
_W_TABLE_EXACT = 3.0
_W_TABLE_FUZZY = 2.0
_W_TABLE_SUBSTR = 1.0
_W_DESC_MATCH = 1.5
_W_COL_EXACT = 1.0
_W_COL_FUZZY = 0.5
_W_COL_DESC = 0.5
_W_BUSINESS_TERM = 2.0


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-word chars, remove stop words and short tokens."""
    tokens = re.split(r"[^\w]+", text.lower())
    return {t for t in tokens if len(t) >= 2 and t not in _STOP_WORDS}


def _split_identifier(name: str) -> set[str]:
    """Split CamelCase or snake_case into lowercase component words.

    Example: "InvoiceLine" -> {"invoiceline", "invoice", "line"}
    """
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", name)
    result = {p.lower() for p in parts}
    result.update(p.lower() for p in name.split("_"))
    result.add(name.lower())
    return result


def _match_business_terms(
    tokens: set[str],
    context: dict,
    table_component_map: dict[str, set[str]],
) -> set[str]:
    """Find tables referenced in business term definitions when terms match the question."""
    matched_tables: set[str] = set()
    for bt in context.get("business_terms", []):
        term_tokens = _tokenize(bt["term"])
        raw = bt.get("synonyms_json", [])
        synonyms = json.loads(raw) if isinstance(raw, str) else raw
        for syn in synonyms:
            term_tokens |= _tokenize(syn)

        if not tokens & term_tokens:
            continue

        # Term activated — scan definition for table name references
        definition = bt.get("definition", "")
        if definition:
            def_tokens = _tokenize(definition)
            for def_tok in def_tokens:
                if def_tok in table_component_map:
                    matched_tables.update(table_component_map[def_tok])
    return matched_tables


def _expand_via_fk(matched_tables: set[str], relationships: list[dict]) -> set[str]:
    """Add tables reachable via 1-hop foreign key from matched tables (bidirectional)."""
    expanded = set(matched_tables)
    for rel in relationships:
        from_t, to_t = rel["from_table"], rel["to_table"]
        if from_t in matched_tables:
            expanded.add(to_t)
        if to_t in matched_tables:
            expanded.add(from_t)
    return expanded


def _stem(token: str) -> str:
    """Naive English stemming: strip trailing 's', 'es', 'ies'."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _fuzzy_match(token: str, target_parts: set[str]) -> bool:
    """Check if token or its stem matches any target part or its stem."""
    candidates = {token, _stem(token)}
    targets = target_parts | {_stem(p) for p in target_parts}
    return bool(candidates & targets)


def score_tables(context: dict, question: str) -> dict[str, float]:
    """Score each table by relevance to the question using keyword matching."""
    tokens = _tokenize(question)
    if not tokens:
        return {}

    tables = context.get("tables", [])

    # Build component -> set of table names map (handles shared components)
    table_component_map: dict[str, set[str]] = {}
    for t in tables:
        for component in _split_identifier(t["name"]):
            table_component_map.setdefault(component, set()).add(t["name"])

    scores: dict[str, float] = {t["name"]: 0.0 for t in tables}

    for table in tables:
        name = table["name"]
        name_parts = _split_identifier(name)

        # Table name matching (with stemming)
        for token in tokens:
            if token == name.lower():
                scores[name] += _W_TABLE_EXACT
            elif _fuzzy_match(token, name_parts):
                scores[name] += _W_TABLE_FUZZY
            elif len(token) >= 3 and token in name.lower():
                scores[name] += _W_TABLE_SUBSTR

        # Table description matching
        if table.get("description"):
            desc_tokens = _tokenize(table["description"])
            for token in tokens:
                if _fuzzy_match(token, desc_tokens):
                    scores[name] += _W_DESC_MATCH

        # Column name and description matching
        for col in table.get("columns", []):
            col_parts = _split_identifier(col["name"])
            col_desc_tokens = _tokenize(col["description"]) if col.get("description") else set()
            for token in tokens:
                if token == col["name"].lower():
                    scores[name] += _W_COL_EXACT
                elif _fuzzy_match(token, col_parts):
                    scores[name] += _W_COL_FUZZY
                if col_desc_tokens and _fuzzy_match(token, col_desc_tokens):
                    scores[name] += _W_COL_DESC

    # Business term matching
    bt_tables = _match_business_terms(tokens, context, table_component_map)
    for tname in bt_tables:
        scores[tname] += _W_BUSINESS_TERM

    return scores


def _score_metadata_entries(
    tokens: set[str],
    entries: list[dict],
    text_keys: list[str],
) -> list[tuple[dict, float]]:
    """Score metadata entries by question relevance using keyword/fuzzy matching.

    Generic scorer for query_patterns, business_terms, and domain_knowledge.
    """
    scored = []
    for entry in entries:
        parts: list[str] = [entry.get(k, "") for k in text_keys]
        # Include synonyms for business terms
        raw_syn = entry.get("synonyms_json", [])
        if raw_syn:
            syns = json.loads(raw_syn) if isinstance(raw_syn, str) else raw_syn
            parts.extend(syns)
        entry_tokens = _tokenize(" ".join(parts))
        if not entry_tokens:
            scored.append((entry, 0.0))
            continue
        score = sum(1.0 for t in tokens if t in entry_tokens or _fuzzy_match(t, entry_tokens))
        scored.append((entry, score))
    return scored


def _filter_metadata(scored: list[tuple[dict, float]], max_entries: int) -> list[dict]:
    """Keep entries with score > 0, capped at max_entries.

    Fallback: if nothing matches, return top-N by score to avoid empty context.
    """
    if not scored:
        return []
    relevant = [(e, s) for e, s in scored if s > 0]
    if not relevant:
        return [e for e, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:max_entries]]
    return [e for e, _ in sorted(relevant, key=lambda x: x[1], reverse=True)[:max_entries]]


def filter_context(
    context: dict,
    question: str,
    max_tables: int = 10,
    score_threshold: float = 0.5,
    max_patterns: int = 10,
    max_terms: int = 15,
    max_knowledge: int = 5,
) -> dict:
    """Filter context to only include question-relevant tables and metadata.

    Tables are scored by keyword matching against names, columns, and descriptions.
    Metadata dimensions (query_patterns, business_terms, domain_knowledge) are also
    filtered by question relevance to prevent context dilution for small models.

    Falls back to the full context if no tables match above the threshold.
    """
    tokens = _tokenize(question)
    scores = score_tables(context, question)

    # Fallback: no scores or nothing above threshold
    if not scores or max(scores.values()) < score_threshold:
        return context

    matched = {name for name, score in scores.items() if score >= score_threshold}

    # FK expansion (1-hop)
    relationships = context.get("relationships", [])
    expanded = _expand_via_fk(matched, relationships)

    # Cap at max_tables — prefer higher-scored direct matches, then FK neighbors
    if len(expanded) > max_tables:
        sorted_matched = sorted(matched, key=lambda t: scores.get(t, 0), reverse=True)
        fk_only = expanded - matched
        kept = set(sorted_matched[:max_tables])
        remaining = max_tables - len(kept)
        if remaining > 0:
            kept.update(list(fk_only)[:remaining])
        expanded = kept

    # Score and filter metadata dimensions by question relevance
    qp_scored = _score_metadata_entries(tokens, context.get("query_patterns", []), ["content"])
    bt_scored = _score_metadata_entries(tokens, context.get("business_terms", []), ["term", "definition"])
    dk_scored = _score_metadata_entries(tokens, context.get("domain_knowledge", []), ["topic", "content"])

    # Build filtered context (same shape as input)
    filtered = {
        "tables": [t for t in context["tables"] if t["name"] in expanded],
        "relationships": [r for r in relationships if r["from_table"] in expanded and r["to_table"] in expanded],
        "query_patterns": _filter_metadata(qp_scored, max_patterns),
        "business_terms": _filter_metadata(bt_scored, max_terms),
        "domain_knowledge": _filter_metadata(dk_scored, max_knowledge),
    }
    # Pass through related_datasets from workspace context unmodified
    if "related_datasets" in context:
        filtered["related_datasets"] = context["related_datasets"]
    return filtered
