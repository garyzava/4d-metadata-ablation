#!/usr/bin/env python3
"""Cross-provider verification smoke (config-driven).

Runs the first N questions of one database at L4 across every model in
config/experiment-parameters.yaml, loading each model's provider + params from
the config, then asserts the run matched the config and emitted ER-keyed results.

Prereqs: BIRD inputs present (run notebook 01) + API keys in the environment
(e.g. `set -a; source .env; set +a`).

Usage (from this repo's root):
    uv run python scripts/provider_matrix_smoke.py
    uv run python scripts/provider_matrix_smoke.py --db financial --limit 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config_env import build_env, load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "run_ablation.py"
CONFIG = REPO / "config" / "experiment-parameters.yaml"


def assert_model(model: dict, prof: dict, result_path: Path, limit: int) -> list[tuple[str, bool, str]]:
    if not result_path.exists():
        return [("result_exists", False, f"missing {result_path}")]
    d = json.loads(result_path.read_text())
    agg, rp, cases = d.get("aggregate_metrics", {}), d.get("runtime_params", {}), d.get("per_case_results", [])
    checks = []
    checks.append(("er_keys_not_ex", "execution_rate_pct" in agg and "execution_accuracy_pct" not in agg, str([k for k in agg if k.startswith("er") or "ex" in k.lower()])))
    blob = json.dumps(d.get("model", "")) + json.dumps(rp)
    checks.append(("model_matches_config", model["model_id"] in blob, f"configured={model['model_id']}"))
    re_cfg = prof.get("reasoning_effort")
    pm = str(rp.get("max_tokens")) == str(prof["max_tokens"])
    pr = (str(rp.get("reasoning_effort")) == str(re_cfg)) if re_cfg else (rp.get("reasoning_effort") in (None, "None"))
    checks.append(("params_match_config", pm and pr, f"max_tokens={rp.get('max_tokens')}/{prof['max_tokens']} reasoning={rp.get('reasoning_effort')}/{re_cfg}"))
    sql_n = sum(1 for c in cases if (c.get("actual_sql") or c.get("generated_sql")))
    need = max(1, math.ceil(limit / 3))
    checks.append(("emits_sql_>=1/3", sql_n >= need, f"{sql_n}/{len(cases)} emitted SQL (need >={need})"))
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--db", default="financial")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--round-dir", default="provider_matrix_smoke")
    args = ap.parse_args()

    cfg = load_config(args.config)
    profiles, models = cfg["profiles"], cfg["models"]
    base = dict(os.environ)
    base["GYZASQL_MIN_COMPLETENESS"] = "0"

    runs = []
    for m in models:
        env, prof = build_env(base, m, profiles)
        rd = f"{args.round_dir}/{m['key'].lower()}"
        cmd = [".venv/bin/python", str(RUNNER), "--datasets", args.db, "--levels", "L4",
               "--reps", "1", "--limit", str(args.limit), "--models", "env", "--round", rd, "--skip-if-exists"]
        if m.get("keep_thinking"):
            cmd.append("--keep-thinking")
        print(f"\n=== {m['key']} :: {m['model_id']} ({m['provider']}, {m['profile']}) ===", flush=True)
        subprocess.run(cmd, env=env, cwd=str(REPO))
        runs.append((m, prof, REPO / "results" / rd / f"{args.db}_L4_env_rep1.json"))

    print("\n" + "=" * 74 + "\nPROVIDER MATRIX SMOKE — ASSERTIONS\n" + "=" * 74)
    all_ok = True
    for m, prof, path in runs:
        for name, ok, detail in assert_model(m, prof, path, args.limit):
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {m['key']:<8} {name:<22} {detail}")
    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
