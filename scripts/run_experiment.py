#!/usr/bin/env python3
"""Config-driven full-experiment driver.

Reads config/experiment-parameters.yaml and runs the documented lane design
across every model, writing per-model results to results/<key>/. It reuses the
same YAML->env mapping as the provider smoke (scripts/_config_env.build_env), so
the real run and the smoke cannot drift from the config or from each other.

Lanes are assigned per model via the `lane:` field in the config:
  - lane: dimensional  -> leave_one_out + cumulative levels, plus the EVIDENCE baseline
  - lane: scoped       -> scoped levels only (no EVIDENCE baseline)
Conditions for each lane come from the config `conditions:` block. Cells =
model x condition x 11 BIRD databases = 429, plus a 33-cell EVIDENCE baseline.

Each model is one `run_ablation.py --models env --round <key> --skip-if-exists`
invocation, so the run is resumable: re-run after an interruption and completed
cells are skipped.

Usage (from the repo root, with the environment loaded):
    set -a; source .env; set +a
    uv run python scripts/run_experiment.py --print-plan       # show the matrix, run nothing
    uv run python scripts/run_experiment.py --dry-run          # set up metadata only, no LLM calls
    uv run python scripts/run_experiment.py                    # full run (all models)
    uv run python scripts/run_experiment.py --models 9B 27B    # subset of models
    uv run python scripts/run_experiment.py --models GEMMA4 --limit 3   # quick check
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config_env import CONFIG, build_env, load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "run_ablation.py"
N_DBS = 11  # BIRD mini-dev databases (--datasets all-bird); run_ablation is authoritative.

# Provider -> the env var that must hold its API key (for a friendly pre-flight check).
PROVIDER_KEY = {
    "huggingface": "HUGGINGFACE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",  # build_env also accepts GEMINI_API_KEY
}


def lane_levels(lane: str, conditions: dict) -> tuple[list[str], bool]:
    """Return (levels, include_evidence) for a model's lane.

    Levels are deduped while preserving order. include_evidence is True only for
    the dimensional lane (the 33-cell EVIDENCE baseline = 3 Qwen x 11 dbs).
    """
    if lane == "dimensional":
        ordered = list(conditions["cumulative"]) + list(conditions["leave_one_out"])
        levels = list(dict.fromkeys(ordered))  # dedupe, keep order
        return levels, True
    if lane == "scoped":
        return list(conditions["scoped"]), False
    raise SystemExit(f"Unknown lane '{lane}'. Expected 'dimensional' or 'scoped'.")


def build_cmd(model: dict, levels: list[str], include_evidence: bool, args) -> list[str]:
    cmd = [
        sys.executable, str(RUNNER),
        "--datasets", "all-bird",
        "--levels", *levels,
        "--reps", "1",
        "--models", "env",
        "--round", model["key"].lower(),
        "--skip-if-exists",
    ]
    if model.get("keep_thinking"):
        cmd.append("--keep-thinking")
    if include_evidence:
        cmd.append("--include-evidence")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of model keys to run (e.g. 9B 27B). Default: all in the config.")
    ap.add_argument("--print-plan", action="store_true",
                    help="Print the per-model lane/condition plan and exit without running.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Pass --dry-run to run_ablation (set up metadata, no LLM calls).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Pass --limit N to run_ablation (first N cases per db; smoke/debug only). "
                         "Use only against an empty results/<key>/: --skip-if-exists loads any "
                         "existing full-size cell, so mixing --limit with prior results is unsafe.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    profiles, conditions, all_models = cfg["profiles"], cfg["conditions"], cfg["models"]

    selected = all_models
    if args.models:
        want = {m.lower() for m in args.models}
        selected = [m for m in all_models if m["key"].lower() in want]
        missing = want - {m["key"].lower() for m in selected}
        if missing:
            raise SystemExit(f"Unknown model key(s): {sorted(missing)}. "
                             f"Available: {[m['key'] for m in all_models]}")

    # Build the plan first so --print-plan and the real run share one source of truth.
    plan = []
    total_cells = 0
    for m in selected:
        levels, include_evidence = lane_levels(m["lane"], conditions)
        cells = len(levels) * N_DBS + (N_DBS if include_evidence else 0)
        total_cells += cells
        plan.append((m, levels, include_evidence, cells))

    print("=" * 78)
    print("FULL-EXPERIMENT PLAN (config-driven)")
    print("=" * 78)
    for m, levels, include_evidence, cells in plan:
        ev = " + EVIDENCE" if include_evidence else ""
        print(f"  {m['key']:<8} lane={m['lane']:<11} {len(levels)} levels x {N_DBS} dbs{ev} = {cells} cells")
        print(f"           levels: {' '.join(levels)}")
        print(f"           {m['provider']} / {m['model_id']} / profile={m['profile']} keep_thinking={bool(m.get('keep_thinking'))}")
    print("-" * 78)
    print(f"  TOTAL: {total_cells} cells across {len(plan)} model(s) "
          f"(reps=1, all {N_DBS} BIRD dbs). run_ablation reports the authoritative count per model.")
    print("=" * 78)

    if args.print_plan:
        return

    # Friendly pre-flight: warn (don't fail) on a missing provider key unless dry-run.
    if not args.dry_run:
        for m, *_ in plan:
            key_env = PROVIDER_KEY.get(m["provider"])
            if key_env and not (os.environ.get(key_env) or (m["provider"] == "gemini" and os.environ.get("GEMINI_API_KEY"))):
                print(f"  WARNING: {key_env} not set in the environment (needed for {m['key']} / {m['provider']}). "
                      f"Did you `set -a; source .env; set +a`?")

    base = dict(os.environ)
    base["GYZASQL_MIN_COMPLETENESS"] = "0"  # match the smoke; the ablation is not completeness-gated

    failures = []
    for m, levels, include_evidence, cells in plan:
        env, _prof = build_env(base, m, profiles)
        cmd = build_cmd(m, levels, include_evidence, args)
        print(f"\n=== {m['key']} :: {m['model_id']} ({m['provider']}) -> results/{m['key'].lower()}/ ===", flush=True)
        print("    " + " ".join(cmd[1:]), flush=True)
        result = subprocess.run(cmd, env=env, cwd=str(REPO))
        if result.returncode != 0:
            failures.append((m["key"], result.returncode))
            print(f"  WARNING: {m['key']} exited with code {result.returncode} (continuing).", flush=True)

    if failures:
        print(f"\nDone with {len(failures)} model(s) reporting a non-zero exit: {failures}")
        sys.exit(1)
    print("\nAll selected models completed.")


if __name__ == "__main__":
    main()
