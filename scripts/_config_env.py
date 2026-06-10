"""Shared config loading and YAML->env mapping for the experiment driver and the
provider smoke.

`run_ablation.py` consumes per-model settings from `GYZASQL_*` environment
variables, not from `config/experiment-parameters.yaml` directly. Both the
full-run driver (`scripts/run_experiment.py`) and the provider smoke
(`scripts/provider_matrix_smoke.py`) call `build_env` here to turn a model's
config profile into those env vars, so the two cannot drift from each other or
from the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "experiment-parameters.yaml"


def load_config(path: str | Path = CONFIG) -> dict[str, Any]:
    """Parse the experiment-parameters YAML into a dict."""
    return yaml.safe_load(Path(path).read_text())


def build_env(base: dict, model: dict, profiles: dict) -> tuple[dict, dict]:
    """Map a model's config profile onto a copy of `base` as GYZASQL_* env vars.

    Returns (env, profile). The returned env is what `run_ablation.py --models env`
    reads: provider, model id, sampling params, and (when enabled) reasoning_effort.
    """
    env = dict(base)
    prof = profiles[model["profile"]]
    env["GYZASQL_LLM_MAX_TOKENS"] = str(prof["max_tokens"])
    if "temperature" in prof:
        env["GYZASQL_LLM_TEMPERATURE"] = str(prof["temperature"])
    if "top_p" in prof:
        env["GYZASQL_LLM_TOP_P"] = str(prof["top_p"])
    for k_env, k_cfg in (("GYZASQL_LLM_TOP_K", "top_k"), ("GYZASQL_LLM_PRESENCE_PENALTY", "presence_penalty")):
        if prof.get(k_cfg) is not None:
            env[k_env] = str(prof[k_cfg])
        else:
            env.pop(k_env, None)
    if prof.get("reasoning_effort"):
        env["GYZASQL_LLM_REASONING_EFFORT"] = str(prof["reasoning_effort"])
    else:
        env.pop("GYZASQL_LLM_REASONING_EFFORT", None)
    env["GYZASQL_LLM_PROVIDER"] = model["provider"]
    env["GYZASQL_MODEL"] = model["model_id"]
    env.pop("GYZASQL_LLM_BASE_URL", None)  # HF -> router; never a stray Ollama URL
    if model["provider"] == "gemini":
        env["GOOGLE_API_KEY"] = env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY", "")
    return env, prof
