#!/usr/bin/env python3
"""Ablation experiment runner for the MetaSQL-Bench study.

Supports multiple datasets: Chinook (original) and BIRD mini-dev PostgreSQL (11 databases).
Each dataset is identified by name and uses the appropriate test cases and metadata loaders.

Usage:
    # Chinook (backward compatible)
    python run_ablation.py --datasets chinook --levels L0 --reps 1

    # Single BIRD database
    python run_ablation.py --datasets financial --levels L0 L4 --reps 1

    # All 11 BIRD databases
    python run_ablation.py --datasets all-bird --levels L0 L4 --reps 1

    # Dry run (set up metadata only, skip LLM calls)
    python run_ablation.py --datasets financial --dry-run

    # Per-dimension ablation
    python run_ablation.py --datasets financial --ablation-only --reps 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Repo root: this runner lives at the public-repo root, so default to its own
# directory (overridable via GYZASQL_REPO_ROOT). The vendored engine is at src/.
PROJECT_ROOT = Path(os.environ.get("GYZASQL_REPO_ROOT") or Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gyzasql.db.connectors import create_engine_for_dataset
from gyzasql.db.introspect import SchemaInfo, introspect_schema
from gyzasql.eval.runner import EvalCase, EvalResult, load_eval_cases, run_eval
from gyzasql.semantic_layer.context import (
    populate_business_context,
    populate_data_dictionary,
    populate_domain_knowledge,
    populate_query_patterns,
    store_introspection,
)
from gyzasql.semantic_layer.store import MetadataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Note: the Opus 4.7 sampling-param fix that previously lived as a runtime
# monkey-patch in this file now lives at the source layer in the vendored
# gyzasql at src/gyzasql/llm/client.py (see src/gyzasql/_VENDORED.md). No shim
# is needed here.

# ── Paths ────────────────────────────────────────────────────────

RESULTS_DIR = Path(os.environ.get("GYZASQL_RESULTS_DIR", str(PROJECT_ROOT / "results")))
CASES_PATH = PROJECT_ROOT / "fixtures" / "chinook_advanced.json"  # legacy Chinook path (unused for BIRD)

# BIRD paths (flat public layout; all overridable by env)
# Defaults match where notebook 01 places its outputs (overridable by env).
BIRD_DATA_DIR = Path(os.environ.get("GYZASQL_BIRD_DATA_DIR", str(PROJECT_ROOT / "bird_mini_dev" / "minidev" / "MINIDEV")))
BIRD_DOCS_DIR = Path(os.environ.get("GYZASQL_BIRD_DOCS_DIR", str(PROJECT_ROOT / "bird-docs-revised")))
BIRD_CASES_PATH = BIRD_DATA_DIR / "mini_dev_postgresql.json"
BIRD_DEV_TABLES = BIRD_DATA_DIR / "dev_tables.json"
BIRD_GOLD_RESULTS_PATH = BIRD_DATA_DIR / "mini_dev_gold_results.json"
# BIRD-specific URL only — never the generic DATABASE_URL (which may point elsewhere).
BIRD_CONNECTION_URL = os.environ.get("GYZASQL_BIRD_DATABASE_URL", "postgresql+psycopg://localhost/bird_minidev")

BIRD_DB_NAMES = [
    "california_schools", "card_games", "codebase_community",
    "debit_card_specializing", "european_football_2", "financial",
    "formula_1", "student_club", "superhero", "thrombosis_prediction", "toxicology",
]

# ── Annotation data loaders ──────────────────────────────────────


def _load_annotation(filename: str) -> dict:
    with open(ANNOTATIONS_DIR / filename) as f:
        return json.load(f)


def _load_data_dictionary() -> dict[str, str | dict[str, str]]:
    """Load the L1 data dictionary in the format expected by populate_data_dictionary."""
    ann = _load_annotation("L1-data-dictionary.json")
    return ann["data_dictionary"]


def _load_query_patterns() -> dict[str, list[str]]:
    """Load L2 query patterns in the format expected by populate_query_patterns."""
    ann = _load_annotation("L2-query-patterns.json")
    patterns: dict[str, list[str]] = {}
    for p in ann["query_patterns"]:
        pt = p["pattern_type"]
        patterns.setdefault(pt, []).append(p["content"])
    return patterns


def _load_business_terms() -> dict[str, dict[str, str | list[str]]]:
    """Load L3 business terms in the format expected by populate_business_context."""
    ann = _load_annotation("L3-business-context.json")
    terms: dict[str, dict[str, str | list[str]]] = {}
    for t in ann["business_terms"]:
        terms[t["term"]] = {
            "definition": t["definition"],
            "synonyms": t.get("synonyms", []),
        }
    return terms


def _load_domain_knowledge() -> dict[str, str]:
    """Load L4 domain knowledge in the format expected by populate_domain_knowledge."""
    ann = _load_annotation("L4-full-metadata.json")
    return {dk["topic"]: dk["content"] for dk in ann["domain_knowledge"]}


# ── BIRD data loaders ────────────────────────────────────────────

# Lazy import to avoid circular issues; bird_metadata.py is in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bird_metadata import load_bird_metadata  # noqa: E402


def _load_bird_cases() -> tuple[dict[str, list[EvalCase]], dict[str, list[EvalCase]]]:
    """Load BIRD mini-dev PG test cases grouped by db_id.

    Returns (cases_by_db, evidence_cases_by_db) where evidence_cases have
    BIRD evidence hints prepended to the question for the evidence baseline.

    If pre-computed gold results exist (from precompute_gold.py), populates
    expected_values for BIRD-ER evaluation.
    """
    with open(BIRD_CASES_PATH) as f:
        raw = json.load(f)

    # Load pre-computed gold results if available
    gold_results: dict[str, dict] = {}
    if BIRD_GOLD_RESULTS_PATH.exists():
        with open(BIRD_GOLD_RESULTS_PATH) as f:
            gold_results = json.load(f)
        logger.info("Loaded %d pre-computed gold results from %s", len(gold_results), BIRD_GOLD_RESULTS_PATH)
    else:
        logger.warning("Gold results not found at %s — bird_ex metric will be unavailable", BIRD_GOLD_RESULTS_PATH)

    cases_by_db: dict[str, list[EvalCase]] = {}
    evidence_cases_by_db: dict[str, list[EvalCase]] = {}
    for item in raw:
        db_id = item["db_id"]
        qid = str(item["question_id"])

        # Populate expected_values from gold results cache
        gold = gold_results.get(qid)
        expected_values = gold["rows"] if gold and gold.get("rows") is not None else None

        case = EvalCase(
            question=item["question"],
            expected_sql=item["SQL"],
            difficulty=item.get("difficulty", "unknown"),
            expected_values=expected_values,
            ignore_order=True,  # BIRD uses set comparison
        )
        cases_by_db.setdefault(db_id, []).append(case)

        # Evidence variant: place BIRD evidence before question (official format)
        evidence = item.get("evidence", "").strip()
        if evidence:
            evidence_question = f"-- External Knowledge: {evidence}\n{item['question']}"
        else:
            evidence_question = item["question"]
        evidence_case = EvalCase(
            question=evidence_question,
            expected_sql=item["SQL"],
            difficulty=item.get("difficulty", "unknown"),
            expected_values=expected_values,
            ignore_order=True,  # BIRD uses set comparison
        )
        evidence_cases_by_db.setdefault(db_id, []).append(evidence_case)

    return cases_by_db, evidence_cases_by_db


def _get_bird_tables(db_name: str) -> set[str]:
    """Get table names for a specific BIRD database from dev_tables.json."""
    with open(BIRD_DEV_TABLES) as f:
        tables_data = json.load(f)
    for db in tables_data:
        if db["db_id"] == db_name:
            return set(db["table_names"])
    return set()


def _is_bird_dataset(dataset_name: str) -> bool:
    return dataset_name in BIRD_DB_NAMES


def _get_connection_url(dataset_name: str, fallback_url: str) -> str:
    if _is_bird_dataset(dataset_name):
        return BIRD_CONNECTION_URL
    return fallback_url


# ── Metadata level definitions ───────────────────────────────────

LEVEL_DIMENSIONS: dict[str, list[str]] = {
    "L0": [],
    "L1": ["data_dictionary"],
    "L2": ["data_dictionary", "query_patterns"],
    "L3": ["data_dictionary", "query_patterns", "business_context"],
    "L4": ["data_dictionary", "query_patterns", "business_context", "domain_knowledge"],
    # Leave-one-out: L4 minus each individual dimension. Drives the
    # marginal-contribution table (preferred over the brittle "BC dominates" framing).
    # Drive these via `--levels L4-DD L4-QP L4-BC L4-DK`,
    # NOT --ablation-only (which iterates ABLATION_DIMENSIONS only).
    "L4-DD": ["query_patterns", "business_context", "domain_knowledge"],
    "L4-QP": ["data_dictionary", "business_context", "domain_knowledge"],
    "L4-BC": ["data_dictionary", "query_patterns", "domain_knowledge"],
    "L4-DK": ["data_dictionary", "query_patterns", "business_context"],
    # Token-matched control: L0 with synthetic non-BIRD pad text
    # injected via populate_domain_knowledge so the prompt length matches L4 without
    # adding any real semantic signal. dimensions=[] here; pad injection is handled
    # post-setup in run_single_condition.
    "L0-PAD": [],
}

# Per-dimension ablation: L0 + only one dimension at a time
ABLATION_DIMENSIONS: dict[str, list[str]] = {
    "A-DD": ["data_dictionary"],
    "A-QP": ["query_patterns"],
    "A-BC": ["business_context"],
    "A-DK": ["domain_knowledge"],
}

# Model configurations
# "env" reads from existing GYZASQL_* env vars (Ollama, local models, etc.)
# Cloud models can be added when API keys are available.
# gemma4 / qwen36 / opus47 are the cross-vendor additions for the 6-model comparison.
MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "env": {
        "provider": "",  # use whatever is in env
        "model": "",     # use whatever is in env
        "env_key": "",   # no validation needed
        "description": "Use current GYZASQL_* env vars (Ollama, local, etc.)",
    },
    "openai": {
        "provider": "openai",
        "model": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
        "description": "OpenAI GPT-4o",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude Sonnet 4",
    },
    "gemma4": {
        "provider": "gemini",
        "model": "gemma-4-26b-a4b-it",
        "env_key": "GOOGLE_API_KEY",
        "description": "Google Gemma 4 26B-A4B-IT (MoE, ~4B active); gyzasql reads GOOGLE_API_KEY for the gemini provider",
    },
    "qwen36": {
        "provider": "huggingface",
        "model": "Qwen/Qwen3.6-35B-A3B",
        "env_key": "HUGGINGFACE_API_KEY",
        "description": "Qwen 3.6 35B-A3B (MoE successor to Qwen3-Coder-30B-A3B)",
    },
    "opus47": {
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "env_key": "ANTHROPIC_API_KEY",
        "description": "Anthropic Claude Opus 4.7 (frontier; scoped to L0/L4/L0-PAD)",
    },
}


# ── Metadata store setup ─────────────────────────────────────────


def _detect_db_type(connection_url: str) -> str:
    """Detect database type from connection URL."""
    if "postgresql" in connection_url or "psycopg" in connection_url:
        return "postgres"
    if "sqlite" in connection_url:
        return "sqlite"
    if "mysql" in connection_url:
        return "mysql"
    return "postgres"


def setup_metadata_store(
    db_path: Path,
    dimensions: list[str],
    connection_url: str,
    dataset_name: str = "chinook",
) -> tuple[MetadataStore, int]:
    """Create a fresh metadata store and populate it with the specified dimensions.

    For BIRD datasets, introspection is filtered to only the db's tables and
    metadata is loaded from bird-docs markdown. For Chinook, uses the existing
    JSON annotation files.

    Returns (store, dataset_id). Note: this previously also returned a
    completeness_score from gyzasql.semantic_layer.scoring, but that score is
    gyzasql-internal IP and is not a paper metric — see src/gyzasql/_VENDORED.md.
    """
    # Remove existing DB to start fresh
    if db_path.exists():
        db_path.unlink()

    db_type = _detect_db_type(connection_url)
    store = MetadataStore(db_path=db_path)
    dataset_id = store.upsert_dataset(dataset_name, connection_url, db_type=db_type)

    # Introspect the schema
    engine = create_engine_for_dataset(connection_url, db_type)
    schema_param = "public" if db_type == "postgres" else None
    full_schema = introspect_schema(engine, schema=schema_param)

    # For BIRD datasets, filter to only this database's tables
    if _is_bird_dataset(dataset_name):
        bird_tables = _get_bird_tables(dataset_name)
        schema = SchemaInfo(
            tables=[t for t in full_schema.tables if t.name in bird_tables],
            foreign_keys=[fk for fk in full_schema.foreign_keys
                          if fk.from_table in bird_tables and fk.to_table in bird_tables],
        )
    else:
        schema = full_schema

    store_introspection(store, dataset_id, schema)

    # Load metadata from appropriate source
    if _is_bird_dataset(dataset_name):
        bird_meta = load_bird_metadata(dataset_name, BIRD_DOCS_DIR)
        if "data_dictionary" in dimensions:
            populate_data_dictionary(store, dataset_id, bird_meta["data_dictionary"])
        if "query_patterns" in dimensions:
            populate_query_patterns(store, dataset_id, bird_meta["query_patterns"])
        if "business_context" in dimensions:
            populate_business_context(store, dataset_id, bird_meta["business_context"])
        if "domain_knowledge" in dimensions:
            populate_domain_knowledge(store, dataset_id, bird_meta["domain_knowledge"])
    else:
        # Chinook: use existing JSON annotation files
        if "data_dictionary" in dimensions:
            populate_data_dictionary(store, dataset_id, _load_data_dictionary())
        if "query_patterns" in dimensions:
            populate_query_patterns(store, dataset_id, _load_query_patterns())
        if "business_context" in dimensions:
            populate_business_context(store, dataset_id, _load_business_terms())
        if "domain_knowledge" in dimensions:
            populate_domain_knowledge(store, dataset_id, _load_domain_knowledge())

    return store, dataset_id


# ── Result aggregation ───────────────────────────────────────────


def aggregate_results(results: list[EvalResult]) -> dict[str, Any]:
    """Compute aggregate metrics from a list of EvalResults."""
    n = len(results)
    if n == 0:
        return {}

    er_count = sum(1 for r in results if r.execution_success)
    value_match_count = sum(1 for r in results if r.metrics.get("value_match", False))
    col_match_count = sum(1 for r in results if r.metrics.get("column_match", False))

    soft_f1_scores = [r.metrics.get("soft_f1", 0.0) for r in results]
    judge_scores = [r.metrics.get("llm_judge_score", 0.0) for r in results if "llm_judge_score" in r.metrics]

    er_pct = er_count / n * 100
    vm_pct = value_match_count / n * 100

    agg = {
        "total_cases": n,
        "execution_rate_pct": round(er_pct, 1),
        "value_match_pct": round(vm_pct, 1),
        "column_match_pct": round(col_match_count / n * 100, 1),
        "soft_f1_mean": round(sum(soft_f1_scores) / n, 4),
        # er_value_gap_pp = ER - VM (kept for backward compatibility with earlier
        # result CSVs and analysis tooling).
        "er_value_gap_pp": round(er_pct - vm_pct, 1),
        "error_count": sum(1 for r in results if r.error),
    }

    if judge_scores:
        agg["llm_judge_mean"] = round(sum(judge_scores) / len(judge_scores), 4)

    # BIRD-standard ER (only present when gold results are available)
    bird_ex_results = [r for r in results if "bird_ex" in r.metrics]
    if bird_ex_results:
        bird_ex_count = sum(1 for r in bird_ex_results if r.metrics["bird_ex"])
        bex_pct = bird_ex_count / n * 100
        agg["bird_ex_pct"] = round(bex_pct, 1)
        # er_bex_gap_pp = ER - BEX (the data dictionary's gap_pp definition).
        # CSV exporters MUST source the `gap_pp` column from this field, never
        # from er_value_gap_pp.
        agg["er_bex_gap_pp"] = round(er_pct - bex_pct, 1)

    # Per-difficulty breakdown (auto-detect labels from data)
    all_difficulties = sorted(set(r.case.difficulty for r in results if r.case.difficulty != "unknown"))
    for difficulty in all_difficulties:
        diff_results = [r for r in results if r.case.difficulty == difficulty]
        if diff_results:
            dn = len(diff_results)
            d_er = sum(1 for r in diff_results if r.execution_success) / dn * 100
            d_vm = sum(1 for r in diff_results if r.metrics.get("value_match", False)) / dn * 100
            d_f1 = sum(r.metrics.get("soft_f1", 0.0) for r in diff_results) / dn
            agg[f"{difficulty}_er_pct"] = round(d_er, 1)
            agg[f"{difficulty}_vm_pct"] = round(d_vm, 1)
            agg[f"{difficulty}_soft_f1"] = round(d_f1, 4)
            # BIRD-ER per difficulty
            d_bird_ex = [r for r in diff_results if "bird_ex" in r.metrics]
            if d_bird_ex:
                agg[f"{difficulty}_bird_ex_pct"] = round(
                    sum(1 for r in d_bird_ex if r.metrics["bird_ex"]) / dn * 100, 1
                )

    return agg


def _classify_status(r: EvalResult) -> str:
    """Coarse per-case outcome bucket for the failure analysis.

    Distinguishes the recurring failure patterns the error-extraction pass
    surfaced, so the stats script can filter on them cleanly without
    re-deriving them from raw error strings.
    """
    if r.execution_success and not r.error:
        return "ok"
    err = (r.error or "").lower()
    sql = (r.actual_sql or "").strip()
    if not sql:
        if "guardrail" in err or "token budget" in err or "empty response" in err:
            return "empty_response_token_exhaustion"
        return "empty_response"
    if "<!doctype html>" in err or "connection error" in err or "rate limit" in err:
        return "api_failure"
    if "timeout" in err or "timed out" in err:
        return "timeout"
    if "syntax error" in err or "parse error" in err:
        return "parse_error"
    return "execution_error"


def result_to_dict(r: EvalResult) -> dict[str, Any]:
    """Serialize an EvalResult to a JSON-safe dict."""
    return {
        "question": r.case.question,
        "difficulty": r.case.difficulty,
        "expected_sql": r.case.expected_sql,
        "actual_sql": r.actual_sql,
        "execution_success": r.execution_success,
        "error": r.error,
        "status": _classify_status(r),
        # prompt_token_count is None until gyzasql surfaces it on EvalResult.
        # The field is reserved so reruns produce JSONs with a consistent shape
        # and downstream tools don't need a schema migration when the field
        # eventually populates.
        "prompt_token_count": None,
        "actual_row_count": r.actual_row_count,
        "metrics": r.metrics,
    }


# ── Single run ───────────────────────────────────────────────────


def run_single_condition(
    level: str,
    dimensions: list[str],
    model_name: str,
    rep: int,
    cases: list[EvalCase],
    connection_url: str,
    dataset_name: str = "chinook",
    judge: bool = False,
    dry_run: bool = False,
    keep_thinking: bool = False,
) -> dict[str, Any]:
    """Run one condition (dataset × level × model × rep) and return results dict."""
    run_id = f"{dataset_name}_{level}_{model_name}_rep{rep}"
    logger.info("=== Starting run: %s ===", run_id)

    # Create a temp metadata DB for this run
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"ablation_{run_id}_"))
    meta_db_path = tmp_dir / "metadata.db"

    try:
        # Set up metadata at the specified level
        store, dataset_id = setup_metadata_store(
            meta_db_path, dimensions, connection_url, dataset_name=dataset_name
        )
        store.close()

        # L0-PAD post-setup hook: inject synthetic non-BIRD pad text via the
        # domain_knowledge dimension so the L0 prompt is padded to ~L4 token
        # count without adding any real semantic signal (the token-matched
        # control). pad_target_tokens=4000 is the default; it could instead be
        # computed from the real per-database L4 prompt size.
        if level == "L0-PAD":
            from padding.synthetic_pool import get_pad_domain_knowledge
            pad_dk = get_pad_domain_knowledge(dataset_name, target_tokens=4000, seed=42)
            pad_store = MetadataStore(db_path=meta_db_path)
            populate_domain_knowledge(pad_store, dataset_id, pad_dk)
            pad_store.close()
            logger.info(
                "  L0-PAD: injected %d synthetic domain_knowledge entries (~%d chars)",
                len(pad_dk), sum(len(v) for v in pad_dk.values()),
            )

        logger.info(
            "  Level=%s, Dimensions=%s",
            level, dimensions,
        )

        if dry_run:
            logger.info("  [DRY RUN] Skipping LLM calls")
            return {
                "run_id": run_id,
                "dataset": dataset_name,
                "level": level,
                "dimensions": dimensions,
                "model": model_name,
                "rep": rep,
                "dry_run": True,
            }

        # Set model environment variables
        model_cfg = MODEL_CONFIGS[model_name]
        if model_name != "env":
            # Override env vars for cloud providers
            os.environ["GYZASQL_LLM_PROVIDER"] = model_cfg["provider"]
            os.environ["GYZASQL_MODEL"] = model_cfg["model"]
            # Clear base_url for cloud providers (avoid routing to Ollama)
            os.environ.pop("GYZASQL_LLM_BASE_URL", None)
        # else: "env" model — leave all GYZASQL_* vars as-is from .env

        # Clear reasoning/thinking mode (use standard completion for reproducibility)
        if not keep_thinking:
            os.environ.pop("GYZASQL_LLM_REASONING_EFFORT", None)
            os.environ.pop("GYZASQL_LLM_THINKING", None)  # backward compat

        # Run the eval
        start_time = time.time()
        results = run_eval(
            cases=cases,
            dataset=dataset_name,
            metadata_db=str(meta_db_path),
            judge=judge,
        )
        elapsed = time.time() - start_time

        # Aggregate metrics
        agg = aggregate_results(results)

        # Record actual model used (resolve "env" to real values)
        if model_name == "env":
            actual_model_info = {
                "provider": os.environ.get("GYZASQL_LLM_PROVIDER", "openai"),
                "model": os.environ.get("GYZASQL_MODEL", "unknown"),
                "base_url": os.environ.get("GYZASQL_LLM_BASE_URL", ""),
            }
        else:
            actual_model_info = {k: v for k, v in MODEL_CONFIGS[model_name].items() if k != "env_key"}

        # Capture resolved runtime parameters for provenance
        from gyzasql.llm.client import get_config as _get_llm_config
        _resolved = _get_llm_config()

        # Resolve num_ctx: env var first, then query Ollama model config
        num_ctx_val = os.environ.get("GYZASQL_LLM_NUM_CTX")
        if not num_ctx_val:
            import subprocess
            try:
                _model_id = os.environ.get("GYZASQL_MODEL", "")
                _show = subprocess.run(
                    ["ollama", "show", _model_id, "--modelfile"],
                    capture_output=True, text=True, timeout=5,
                )
                for _line in _show.stdout.splitlines():
                    if _line.strip().startswith("PARAMETER num_ctx"):
                        num_ctx_val = _line.strip().split()[-1]
                        break
            except Exception:
                pass

        from gyzasql.llm.client import _effective_max_tokens, _reasoning_enabled
        _reasoning_on = _reasoning_enabled(_resolved)
        runtime_params = {
            "max_tokens": _resolved.max_tokens,
            "effective_max_tokens": _effective_max_tokens(_resolved),
            "temperature": _resolved.temperature,
            "seed": _resolved.seed,
            "top_p": _resolved.top_p,
            "presence_penalty": _resolved.presence_penalty,
            "top_k": _resolved.top_k,
            "reasoning_effort": _resolved.reasoning_effort,
            "num_ctx": int(num_ctx_val) if num_ctx_val else None,
            "bird_docs_dir": str(BIRD_DOCS_DIR),
        }

        # Build run record
        run_record = {
            "run_id": run_id,
            "dataset": dataset_name,
            "level": level,
            "dimensions": dimensions,
            "model": actual_model_info,
            "runtime_params": runtime_params,
            "rep": rep,
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "aggregate_metrics": agg,
            "per_case_results": [result_to_dict(r) for r in results],
        }

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result_path = RESULTS_DIR / f"{run_id}.json"
        with open(result_path, "w") as f:
            json.dump(run_record, f, indent=2, default=str)
        logger.info("  Saved results to %s", result_path)

        # Log summary: surface BOTH gaps so the ER-BEX (canonical) and
        # ER-VM (legacy) values are visible at a glance. The narrower
        # er_bex_gap_pp (canonical "Gap") only renders when bird_ex was
        # available; otherwise fall through to er_value_gap_pp.
        _gap_canonical = agg.get("er_bex_gap_pp")
        if _gap_canonical is None:
            _gap_canonical = agg.get("er_value_gap_pp", 0)
        logger.info(
            "  ER=%.1f%% BEX=%.1f%% VM=%.1f%% F1=%.4f Gap(ER-BEX)=%.1fpp Gap(ER-VM)=%.1fpp [%.1fs]",
            agg.get("execution_rate_pct", 0),
            agg.get("bird_ex_pct", 0),
            agg.get("value_match_pct", 0),
            agg.get("soft_f1_mean", 0),
            _gap_canonical,
            agg.get("er_value_gap_pp", 0),
            elapsed,
        )

        return run_record

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MetaSQL-Bench ablation experiment runner")
    parser.add_argument(
        "--datasets", nargs="+", default=["chinook"],
        help="Dataset names. Use 'chinook', BIRD db names (e.g. 'financial'), or 'all-bird' for all 11.",
    )
    parser.add_argument(
        "--levels", nargs="+", default=list(LEVEL_DIMENSIONS.keys()),
        help="Metadata levels to run (default: all L0-L4)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["env"],
        help="LLM models to test (default: 'env' = current .env config). Options: env, openai, anthropic",
    )
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per condition (default: 3)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N cases per dataset (smoke/debug); default: all")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-Judge scoring")
    parser.add_argument("--ablation-only", action="store_true", help="Run per-dimension ablation only")
    parser.add_argument("--include-ablation", action="store_true", help="Include per-dimension ablation runs")
    parser.add_argument(
        "--ablation-dims", nargs="+", default=None,
        help="Subset of ablation dimensions (e.g. A-DK). Default: all four. Only takes effect with --include-ablation or --ablation-only.",
    )
    parser.add_argument("--include-evidence", action="store_true", help="Include BIRD evidence baseline (L0 + per-question hints)")
    parser.add_argument("--evidence-only", action="store_true", help="Run ONLY the EVIDENCE baseline (no L0-L4 levels, no ablation). Implies --include-evidence.")
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip any cell whose per-case JSON already exists at the deterministic round-dir path. "
             "Skipped cells are loaded into the in-memory summary so experiment_summary.json stays complete. "
             "Makes long-running experiments idempotent: re-run after interruption to resume.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Set up metadata only, skip LLM calls")
    parser.add_argument("--keep-thinking", action="store_true",
        help="Keep reasoning mode from env (GYZASQL_LLM_REASONING_EFFORT or GYZASQL_LLM_THINKING). Default: clear for reproducibility.")
    parser.add_argument(
        "--round", type=str, default=None,
        help="Results subdirectory name (e.g. a per-model name like 9b). Default: flat results dir.",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="Database connection URL for Chinook (default: $DATABASE_URL). BIRD databases use their own URL.",
    )
    args = parser.parse_args()

    # Resolve results directory (supports --round for per-round subfolders)
    global RESULTS_DIR  # noqa: PLW0603
    if args.round:
        RESULTS_DIR = Path(os.environ.get("GYZASQL_RESULTS_DIR", str(PROJECT_ROOT / "results"))) / args.round

    # Expand dataset names
    datasets: list[str] = []
    for ds in args.datasets:
        if ds == "all-bird":
            datasets.extend(BIRD_DB_NAMES)
        else:
            datasets.append(ds)

    # Validate Chinook has a database URL
    has_chinook = "chinook" in datasets
    if has_chinook and not args.database_url:
        logger.error("DATABASE_URL not set for chinook. Provide --database-url or set the env var.")
        sys.exit(1)

    # Validate models have API keys
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            logger.error("Unknown model: %s. Available: %s", model_name, list(MODEL_CONFIGS.keys()))
            sys.exit(1)
        env_key = MODEL_CONFIGS[model_name]["env_key"]
        if env_key and not args.dry_run and not os.environ.get(env_key):
            logger.error("Missing API key: %s (required for model '%s')", env_key, model_name)
            sys.exit(1)

    # Load test cases per dataset
    bird_cases_loaded = False
    bird_cases_by_db: dict[str, list[EvalCase]] = {}
    bird_evidence_cases_by_db: dict[str, list[EvalCase]] = {}
    cases_by_dataset: dict[str, list[EvalCase]] = {}
    evidence_cases_by_dataset: dict[str, list[EvalCase]] = {}

    for ds_name in datasets:
        if ds_name == "chinook":
            cases_by_dataset[ds_name] = load_eval_cases(CASES_PATH)
        elif _is_bird_dataset(ds_name):
            if not bird_cases_loaded:
                bird_cases_by_db, bird_evidence_cases_by_db = _load_bird_cases()
                bird_cases_loaded = True
            if ds_name not in bird_cases_by_db:
                logger.error("No BIRD test cases found for database: %s", ds_name)
                sys.exit(1)
            cases_by_dataset[ds_name] = bird_cases_by_db[ds_name]
            evidence_cases_by_dataset[ds_name] = bird_evidence_cases_by_db[ds_name]
        else:
            logger.error("Unknown dataset: %s. Use 'chinook', a BIRD db name, or 'all-bird'.", ds_name)
            sys.exit(1)

    if args.limit is not None:
        for _ds in list(cases_by_dataset):
            cases_by_dataset[_ds] = cases_by_dataset[_ds][: args.limit]
        for _ds in list(evidence_cases_by_dataset):
            evidence_cases_by_dataset[_ds] = evidence_cases_by_dataset[_ds][: args.limit]

    for ds_name, cases in cases_by_dataset.items():
        logger.info("Dataset '%s': %d test cases", ds_name, len(cases))

    # --evidence-only is a convenience flag for the EVIDENCE-only rerun: no
    # regular levels, no ablation dimensions, just the EVIDENCE baseline.
    # Implies --include-evidence.
    if args.evidence_only:
        args.include_evidence = True

    # Build the run matrix
    conditions: list[tuple[str, list[str]]] = []

    if not args.ablation_only and not args.evidence_only:
        for level in args.levels:
            if level not in LEVEL_DIMENSIONS:
                logger.error("Unknown level: %s", level)
                sys.exit(1)
            conditions.append((level, LEVEL_DIMENSIONS[level]))

    if args.ablation_only or args.include_ablation:
        requested_dims = args.ablation_dims or list(ABLATION_DIMENSIONS.keys())
        for ablation_name in requested_dims:
            if ablation_name not in ABLATION_DIMENSIONS:
                logger.error("Unknown ablation dimension: %s (valid: %s)", ablation_name, sorted(ABLATION_DIMENSIONS.keys()))
                sys.exit(1)
            conditions.append((ablation_name, ABLATION_DIMENSIONS[ablation_name]))

    # Count evidence baseline runs
    evidence_datasets = [ds for ds in datasets if ds in evidence_cases_by_dataset] if args.include_evidence else []
    evidence_runs = len(evidence_datasets) * len(args.models) * args.reps

    total_runs = len(datasets) * len(conditions) * len(args.models) * args.reps + evidence_runs
    logger.info(
        "Experiment matrix: %d datasets × %d conditions × %d models × %d reps = %d runs (+%d evidence baseline)",
        len(datasets), len(conditions), len(args.models), args.reps,
        total_runs - evidence_runs, evidence_runs,
    )

    # Execute
    all_summaries: list[dict[str, Any]] = []
    run_count = 0

    def _load_existing_record(level: str, model_name: str, dataset_name: str, rep: int) -> dict[str, Any] | None:
        """If --skip-if-exists is on and the cell's JSON exists, load + return it. Else None.

        Returns the run-record dict in the same shape `run_single_condition` produces, so
        the caller's `all_summaries.append(...)` logic is identical for skipped vs computed
        cells. Logs a SKIP line on hit; logs a WARNING + returns None on parse failure
        (so the cell gets re-run rather than silently lost).
        """
        if not args.skip_if_exists:
            return None
        run_id = f"{dataset_name}_{level}_{model_name}_rep{rep}"
        result_path = RESULTS_DIR / f"{run_id}.json"
        if not result_path.exists():
            return None
        try:
            with open(result_path) as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("  --skip-if-exists: %s exists but failed to load (%s); will re-run", result_path, exc)
            return None
        logger.info("  === SKIP %s: existing JSON at %s ===", run_id, result_path)
        return record

    for dataset_name in datasets:
        cases = cases_by_dataset[dataset_name]
        connection_url = _get_connection_url(dataset_name, args.database_url or "")

        for level, dimensions in conditions:
            for model_name in args.models:
                for rep in range(1, args.reps + 1):
                    run_count += 1
                    logger.info("--- Run %d/%d [%s] ---", run_count, total_runs, dataset_name)
                    record = _load_existing_record(level, model_name, dataset_name, rep)
                    if record is None:
                        record = run_single_condition(
                            level=level,
                            dimensions=dimensions,
                            model_name=model_name,
                            rep=rep,
                            cases=cases,
                            connection_url=connection_url,
                            dataset_name=dataset_name,
                            judge=args.judge,
                            dry_run=args.dry_run,
                            keep_thinking=args.keep_thinking,
                        )
                    all_summaries.append({
                        "run_id": record["run_id"],
                        "dataset": dataset_name,
                        "level": record["level"],
                        "model": record["model"],
                        "runtime_params": record.get("runtime_params"),
                        "rep": rep,
                        **(record.get("aggregate_metrics", {})),
                    })

    # Evidence baseline: L0 schema + per-question BIRD evidence hints
    if args.include_evidence:
        for dataset_name in evidence_datasets:
            ev_cases = evidence_cases_by_dataset[dataset_name]
            connection_url = _get_connection_url(dataset_name, args.database_url or "")

            for model_name in args.models:
                for rep in range(1, args.reps + 1):
                    run_count += 1
                    logger.info("--- Run %d/%d [%s EVIDENCE] ---", run_count, total_runs, dataset_name)
                    record = _load_existing_record("EVIDENCE", model_name, dataset_name, rep)
                    if record is not None:
                        all_summaries.append({
                            "run_id": record["run_id"],
                            "dataset": dataset_name,
                            "level": "EVIDENCE",
                            "model": record["model"],
                            "runtime_params": record.get("runtime_params"),
                            "rep": rep,
                            **(record.get("aggregate_metrics", {})),
                        })
                        continue
                    record = run_single_condition(
                        level="EVIDENCE",
                        dimensions=[],  # L0 schema only — evidence is in the question
                        model_name=model_name,
                        rep=rep,
                        cases=ev_cases,
                        connection_url=connection_url,
                        dataset_name=dataset_name,
                        judge=args.judge,
                        dry_run=args.dry_run,
                        keep_thinking=args.keep_thinking,
                    )
                    all_summaries.append({
                        "run_id": record["run_id"],
                        "dataset": dataset_name,
                        "level": "EVIDENCE",
                        "model": record["model"],
                        "runtime_params": record.get("runtime_params"),
                        "rep": rep,
                        **(record.get("aggregate_metrics", {})),
                    })

    # Save summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Experiment complete. Summary saved to %s", summary_path)

    # Print summary table
    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)
    has_bird_ex = any("bird_ex_pct" in s for s in all_summaries)
    if has_bird_ex:
        print(f"{'Run ID':<40} {'ER%':>6} {'VM%':>6} {'BEX%':>6} {'F1':>7} {'Gap':>6}")
    else:
        print(f"{'Run ID':<40} {'ER%':>6} {'VM%':>6} {'F1':>7} {'Gap':>6}")
    print("-" * 92)
    for s in all_summaries:
        line = (
            f"{s['run_id']:<40} "
            f"{s.get('execution_rate_pct', 0):>6.1f} "
            f"{s.get('value_match_pct', 0):>6.1f} "
        )
        if has_bird_ex:
            line += f"{s.get('bird_ex_pct', 0):>6.1f} "
        # Display the canonical ER-BEX gap; fall back to ER-VM only
        # when bird_ex isn't available (e.g., Chinook eval with no gold).
        _gap = s.get("er_bex_gap_pp")
        if _gap is None:
            _gap = s.get("er_value_gap_pp", 0)
        line += (
            f"{s.get('soft_f1_mean', 0):>7.4f} "
            f"{_gap:>6.1f}"
        )
        print(line)
    print("=" * 98)


if __name__ == "__main__":
    main()
