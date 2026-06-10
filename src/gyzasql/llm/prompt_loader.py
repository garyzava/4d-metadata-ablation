"""Load and cache system prompts from markdown files in the prompts/ directory."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Project root: up from src/gyzasql/llm/ -> src/gyzasql/ -> src/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _prompts_dir() -> Path:
    """Return the prompts directory, respecting GYZASQL_PROMPTS_DIR override."""
    if env_dir := os.environ.get("GYZASQL_PROMPTS_DIR"):
        return Path(env_dir)
    return _PROJECT_ROOT / "prompts"


def _strip_frontmatter(text: str) -> str:
    """Strip YAML frontmatter (delimited by ---) from the beginning of text."""
    if not text.startswith("---"):
        return text
    # Find the closing ---
    end = text.find("---", 3)
    if end == -1:
        return text
    # Skip past the closing --- and any immediately following newline
    body = text[end + 3 :]
    if body.startswith("\n"):
        body = body[1:]
    return body


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Load a prompt from prompts/{name}.md, stripping YAML frontmatter.

    Args:
        name: Prompt file name without extension (e.g. "text2sql").

    Returns:
        The prompt body text.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    path = _prompts_dir() / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    return _strip_frontmatter(raw)


def load_template(name: str, **kwargs: str) -> str:
    """Load a prompt template and apply str.format() substitution.

    Args:
        name: Prompt file name without extension.
        **kwargs: Variables to substitute (e.g. dialect_name, quoting_rule).

    Returns:
        The formatted prompt text.
    """
    return load_prompt(name).format(**kwargs)
