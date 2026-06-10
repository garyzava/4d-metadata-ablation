"""Per-question paired statistical analysis on data.

Addresses the review point "aggregate database-level percentages are not enough"
by pairing the same question under a baseline vs an L4-effective condition.

Lane structure (verified):
  - Dimensional lanes  (9b, 27b, 30b)           : EVIDENCE, L0-PAD, L4-DD, L4-QP, L4-BC, L4-DK
  - Cross-vendor lanes (gemma4, qwen36, opus47) : L0, L0-PAD, L4

Baseline used per lane (the "L0" reference):
  - Cross-vendor : L0
  - Dimensional  : L0-PAD (the token-matched control; the dimensional lanes have no native L0)

L4-effective per lane (the "L4" treatment):
  - Cross-vendor : L4
  - Dimensional  : each of the four leave-one-out conditions (L4-DD/L4-QP/L4-BC/L4-DK),
             pooled across LOO into a single L4-family contrast for the headline,
             AND broken out separately for the LOO ablation table.

Outputs (relative to this repo):
  - data/reports/stats-tables.md
  - data/per-question-stats.csv  (when --csv is passed)

Usage (from this repo's root):
    python scripts/per_question_stats.py
    python scripts/per_question_stats.py --csv data/per-question-stats.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
REPO = HERE.parent

HF_MODELS = ["9B", "27B", "30B"]
SCOPED_MODELS = ["GEMMA4", "QWEN36", "OPUS47"]
ALL_MODELS = HF_MODELS + SCOPED_MODELS

RESULTS_DIRS = {
    "9B":     REPO / "results" / "9b",
    "27B":    REPO / "results" / "27b",
    "30B":    REPO / "results" / "30b",
    "GEMMA4": REPO / "results" / "gemma4",
    "QWEN36": REPO / "results" / "qwen36",
    "OPUS47": REPO / "results" / "opus47",
}
OUT_PATH = REPO / "data" / "reports" / "stats-tables.md"

L4_LOO = ["L4-DD", "L4-QP", "L4-BC", "L4-DK"]


def baseline_level(model: str) -> str:
    return "L0" if model in SCOPED_MODELS else "L0-PAD"


def load_long_dataframe() -> pd.DataFrame:
    """Walk all per-cell result JSONs into one long-format DataFrame.

    One row per (model, database, level, case_index). bird_ex is 0/1.
    case_index is stable across levels for the same database (the runner
    iterates cases in load order), so it serves as a question_id within
    a (model, database) pair.
    """
    rows: list[dict] = []
    for model_code, dir_path in RESULTS_DIRS.items():
        if not dir_path.exists():
            continue
        for jp in sorted(dir_path.glob("*.json")):
            if jp.name == "experiment_summary.json":
                continue
            try:
                rec = json.loads(jp.read_text())
            except Exception:
                continue
            db = rec.get("dataset", "")
            level = rec.get("level", "")
            for i, c in enumerate(rec.get("per_case_results", [])):
                m = c.get("metrics", {}) or {}
                if "bird_ex" not in m:
                    continue
                rows.append({
                    "model": model_code,
                    "lane": "HF" if model_code in HF_MODELS else "SCOPED",
                    "database": db,
                    "level": level,
                    "case_index": i,
                    "difficulty": c.get("difficulty", "unknown"),
                    "bird_ex": int(bool(m.get("bird_ex", False))),
                    "value_match": int(bool(m.get("value_match", False))),
                    "soft_f1": float(m.get("soft_f1", 0.0)),
                    "execution_success": int(bool(c.get("execution_success", False))),
                })
    return pd.DataFrame(rows)


def audit_missing_bird_ex() -> pd.DataFrame:
    """List every per-case row WITHOUT a bird_ex metric (BIRD gold missing)."""
    rows: list[dict] = []
    for model_code, dir_path in RESULTS_DIRS.items():
        if not dir_path.exists():
            continue
        for jp in sorted(dir_path.glob("*.json")):
            if jp.name == "experiment_summary.json":
                continue
            try:
                rec = json.loads(jp.read_text())
            except Exception:
                continue
            db = rec.get("dataset", "")
            level = rec.get("level", "")
            for i, c in enumerate(rec.get("per_case_results", [])):
                m = c.get("metrics", {}) or {}
                if "bird_ex" in m:
                    continue
                if not m:
                    reason = "no metrics dict"
                elif "value_match" in m and "soft_f1" in m:
                    reason = "gold result missing from mini_dev_gold_results.json"
                else:
                    reason = "metrics dict present but bird_ex absent"
                rows.append({
                    "model": model_code,
                    "database": db,
                    "level": level,
                    "case_index": i,
                    "difficulty": c.get("difficulty", "unknown"),
                    "question": (c.get("question") or "").replace("\n", " ")[:90],
                    "reason": reason,
                })
    return pd.DataFrame(rows)


def _paired_long(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Build (baseline_bex, l4_bex) paired rows for a single model.

    Scoped lanes contribute one paired row per question (L0 vs L4).
    HF lanes contribute up to four paired rows per question (L0-PAD vs each LOO).
    """
    sub = df[df["model"] == model]
    base = sub[sub["level"] == baseline_level(model)][
        ["database", "case_index", "bird_ex"]
    ].rename(columns={"bird_ex": "bex_base"})
    if model in SCOPED_MODELS:
        treat_levels = ["L4"]
    else:
        treat_levels = L4_LOO
    pieces: list[pd.DataFrame] = []
    for lvl in treat_levels:
        t = sub[sub["level"] == lvl][
            ["database", "case_index", "bird_ex"]
        ].rename(columns={"bird_ex": "bex_l4"})
        if t.empty:
            continue
        m = base.merge(t, on=["database", "case_index"], how="inner")
        m["l4_level"] = lvl
        pieces.append(m)
    if not pieces:
        return pd.DataFrame(columns=["database", "case_index", "bex_base", "bex_l4", "l4_level"])
    return pd.concat(pieces, ignore_index=True)


def mcnemar_baseline_vs_l4(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model paired McNemar on bird_ex, baseline vs L4-effective.

    For HF lanes the unit is (question, L4-LOO condition); for scoped lanes it
    is just question. b = baseline wrong but L4 right; c = baseline right but
    L4 wrong. Two-sided exact-binomial p.
    """
    out: list[dict] = []
    for model in [m for m in ALL_MODELS if m in df["model"].unique()]:
        pair = _paired_long(df, model)
        n = len(pair)
        if n == 0:
            continue
        b = int(((pair["bex_base"] == 0) & (pair["bex_l4"] == 1)).sum())
        c = int(((pair["bex_base"] == 1) & (pair["bex_l4"] == 0)).sum())
        agree = n - b - c
        if b + c > 0:
            p = stats.binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue
        else:
            p = 1.0
        out.append({
            "model": model,
            "baseline": baseline_level(model),
            "n_paired_obs": n,
            "agree": agree,
            "baseline_wins_c": c,
            "L4_wins_b": b,
            "delta_b_minus_c": b - c,
            "mcnemar_p": p,
        })
    return pd.DataFrame(out)


def bootstrap_question_delta(df: pd.DataFrame, n_resamples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap 95% CI on baseline -> L4 BEX delta per model.

    For HF lanes the bootstrap samples paired (question, LOO) observations
    (so a question can contribute up to 4 rows); for scoped lanes each
    question contributes one row.
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for model in [m for m in ALL_MODELS if m in df["model"].unique()]:
        pair = _paired_long(df, model)
        if pair.empty:
            continue
        deltas = (pair["bex_l4"] - pair["bex_base"]).to_numpy()
        n = len(deltas)
        point = float(np.mean(deltas)) * 100
        idx = rng.integers(0, n, size=(n_resamples, n))
        boot_means = deltas[idx].mean(axis=1) * 100
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        out.append({
            "model": model,
            "baseline": baseline_level(model),
            "n_obs": n,
            "point_estimate_pp": round(point, 2),
            "ci_lo_pp": round(float(lo), 2),
            "ci_hi_pp": round(float(hi), 2),
        })
    return pd.DataFrame(out)


def bootstrap_database_delta(df: pd.DataFrame, n_resamples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap 95% CI on baseline -> L4 BEX delta, resampling DATABASES."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for model in [m for m in ALL_MODELS if m in df["model"].unique()]:
        pair = _paired_long(df, model)
        if pair.empty:
            continue
        per_db = pair.assign(delta=pair["bex_l4"] - pair["bex_base"]) \
            .groupby("database")["delta"].mean()
        deltas_by_db = per_db.to_numpy() * 100
        n_dbs = len(deltas_by_db)
        if n_dbs == 0:
            continue
        point = float(np.mean(deltas_by_db))
        idx = rng.integers(0, n_dbs, size=(n_resamples, n_dbs))
        boot_means = deltas_by_db[idx].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        out.append({
            "model": model,
            "baseline": baseline_level(model),
            "n_databases": n_dbs,
            "point_estimate_pp": round(point, 2),
            "ci_lo_pp": round(float(lo), 2),
            "ci_hi_pp": round(float(hi), 2),
        })
    return pd.DataFrame(out)


def loo_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dimension LOO ablation on the HF lanes, paired against L0-PAD.

    For each (HF model, L4-LOO condition) pair, compute paired McNemar of
    L0-PAD vs the LOO. Holm-Bonferroni adjusts across the 12 comparisons
    (4 dimensions x 3 HF models). p_holm_lt_05 indicates significance after
    correction.
    """
    raw: list[dict] = []
    for model in HF_MODELS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        base = sub[sub["level"] == "L0-PAD"][
            ["database", "case_index", "bird_ex"]
        ].rename(columns={"bird_ex": "bex_base"})
        for dim in L4_LOO:
            ax = sub[sub["level"] == dim][
                ["database", "case_index", "bird_ex"]
            ].rename(columns={"bird_ex": "bex_dim"})
            pair = base.merge(ax, on=["database", "case_index"], how="inner")
            n = len(pair)
            if n == 0:
                continue
            b = int(((pair["bex_base"] == 0) & (pair["bex_dim"] == 1)).sum())
            c = int(((pair["bex_base"] == 1) & (pair["bex_dim"] == 0)).sum())
            delta = float((pair["bex_dim"] - pair["bex_base"]).mean()) * 100
            if b + c > 0:
                p_raw = stats.binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue
            else:
                p_raw = 1.0
            raw.append({
                "model": model,
                "dimension_dropped": dim,
                "n": n,
                "delta_pp": round(delta, 2),
                "L0PAD_wins_c": c,
                "dim_wins_b": b,
                "p_raw": p_raw,
            })
    out = pd.DataFrame(raw)
    if out.empty:
        return out
    out_sorted = out.sort_values("p_raw").reset_index(drop=True)
    m = len(out_sorted)
    out_sorted["p_holm"] = [min(1.0, p * (m - i)) for i, p in enumerate(out_sorted["p_raw"])]
    holm = out_sorted["p_holm"].to_numpy().copy()
    for i in range(1, len(holm)):
        if holm[i] < holm[i - 1]:
            holm[i] = holm[i - 1]
    out_sorted["p_holm"] = holm
    out_sorted["p_holm_lt_05"] = out_sorted["p_holm"] < 0.05
    return out_sorted.sort_values(["model", "dimension_dropped"]).reset_index(drop=True)


def scoped_logit(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Logit on scoped lanes: bird_ex ~ C(level) + C(model) + C(difficulty)
    with database-cluster-robust SE. Reference categories: L0, GEMMA4, the
    alphabetically-first difficulty class.
    """
    sub = df[(df["lane"] == "SCOPED") & (df["level"].isin(["L0", "L4"]))].copy()
    if sub.empty:
        return None, "no scoped L0/L4 rows"
    sub["level"] = pd.Categorical(sub["level"], categories=["L0", "L4"], ordered=True)
    sub["model"] = pd.Categorical(sub["model"], categories=SCOPED_MODELS)
    sub["difficulty"] = pd.Categorical(sub["difficulty"], categories=sorted(sub["difficulty"].unique()))
    try:
        m = smf.logit("bird_ex ~ C(level) + C(model) + C(difficulty)", data=sub).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": sub["database"]},
        )
    except Exception as e:
        return None, f"scoped logit fit failed: {e}"
    rows: list[dict] = []
    for term, coef, se, p in zip(m.params.index, m.params.values, m.bse.values, m.pvalues.values):
        rows.append({
            "term": term,
            "coef": round(float(coef), 4),
            "stderr": round(float(se), 4),
            "z": round(float(coef / se) if se else 0.0, 3),
            "p": round(float(p), 6),
            "odds_ratio": round(float(np.exp(coef)), 3),
        })
    return pd.DataFrame(rows), f"Scoped lanes only; logit on L0 vs L4 with cluster-robust SE on database (n={len(sub)})"


def hf_logit(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Logit on HF lanes: bird_ex ~ C(level) + C(model) + C(difficulty)
    with database-cluster-robust SE. Levels included: L0-PAD plus the four
    L4-LOO conditions. Reference category: L0-PAD.
    """
    sub = df[(df["lane"] == "HF") & (df["level"].isin(["L0-PAD", *L4_LOO]))].copy()
    if sub.empty:
        return None, "no HF rows for the requested levels"
    sub["level"] = pd.Categorical(sub["level"], categories=["L0-PAD", *L4_LOO], ordered=True)
    sub["model"] = pd.Categorical(sub["model"], categories=HF_MODELS)
    sub["difficulty"] = pd.Categorical(sub["difficulty"], categories=sorted(sub["difficulty"].unique()))
    try:
        m = smf.logit("bird_ex ~ C(level) + C(model) + C(difficulty)", data=sub).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": sub["database"]},
        )
    except Exception as e:
        return None, f"HF logit fit failed: {e}"
    rows: list[dict] = []
    for term, coef, se, p in zip(m.params.index, m.params.values, m.bse.values, m.pvalues.values):
        rows.append({
            "term": term,
            "coef": round(float(coef), 4),
            "stderr": round(float(se), 4),
            "z": round(float(coef / se) if se else 0.0, 3),
            "p": round(float(p), 6),
            "odds_ratio": round(float(np.exp(coef)), 3),
        })
    return pd.DataFrame(rows), f"HF lanes only; logit on L0-PAD + L4-LOO with cluster-robust SE on database (n={len(sub)})"


def df_to_md(df: pd.DataFrame, *, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_(no rows)_\n"
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells: list[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_report(df: pd.DataFrame, mcn: pd.DataFrame, boot_q: pd.DataFrame,
                  boot_db: pd.DataFrame, loo: pd.DataFrame,
                  scoped_coef: pd.DataFrame | None, scoped_note: str,
                  hf_coef: pd.DataFrame | None, hf_note: str,
                  excluded: pd.DataFrame) -> str:
    n_rows = len(df)
    n_models = df["model"].nunique()
    n_dbs = df["database"].nunique()
    n_levels = df["level"].nunique()
    n_excluded = len(excluded)

    parts = [
        "# Per-question statistical analysis",
        "",
        "Generated by `scripts/per_question_stats.py`. Walks the existing "
        "* per-case JSONs (no new LLM calls). Pairs the same question under a baseline "
        "(L0 for scoped lanes, L0-PAD for HF lanes) with an L4-effective condition "
        "(native L4 for scoped, each of the four L4 leave-one-out conditions for HF).",
        "",
        "## 1. Data shape",
        "",
        f"- Long-format rows: **{n_rows:,}** (across {n_models} models, {n_dbs} databases, {n_levels} levels).",
        f"- Levels covered: {sorted(df['level'].unique())}.",
        f"- Cases excluded for missing `bird_ex`: **{n_excluded}** (audit at the bottom).",
        "",
        "## 2. Paired McNemar - baseline vs L4-effective",
        "",
        "For each model, b = baseline wrong but L4 right; c = baseline right but L4 wrong. "
        "Two-sided exact-binomial p-value. HF rows pool the four L4-LOO conditions, so "
        "n_paired_obs is approximately 4x the question count.",
        "",
        df_to_md(mcn, floatfmt=".6f"),
        "",
        "## 3. Bootstrap 95% CI on the BEX delta (percentage points)",
        "",
        "**3a. Observation-level (10,000 resamples per model)** - robustness to which "
        "(question, LOO) observations were sampled.",
        "",
        df_to_md(boot_q, floatfmt=".2f"),
        "",
        "**3b. Database-level (10,000 resamples of the 11 databases per model)** - "
        "robustness to which databases are in the study. The wider CI is the more "
        "honest cross-database generalization claim.",
        "",
        df_to_md(boot_db, floatfmt=".2f"),
        "",
        "## 4. Leave-one-out ablation (HF lanes)",
        "",
        "Each row pairs L0-PAD against an L4-LOO condition for an HF model. delta_pp > 0 "
        "means dropping that dimension still beats L0-PAD; delta_pp closer to zero or negative "
        "means the dropped dimension is load-bearing. Holm-Bonferroni adjusts across the 12 "
        "comparisons (4 dimensions x 3 HF models).",
        "",
        df_to_md(loo, floatfmt=".4f"),
        "",
        "## 5. Logistic regression with cluster-robust SE on database",
        "",
        "Two fits, one per lane type, because the level set is lane-specific.",
        "",
        f"**5a. Scoped lanes.** _{scoped_note}_",
        "",
        df_to_md(scoped_coef if scoped_coef is not None else pd.DataFrame(), floatfmt=".4f"),
        "",
        f"**5b. HF lanes.** _{hf_note}_",
        "",
        df_to_md(hf_coef if hf_coef is not None else pd.DataFrame(), floatfmt=".4f"),
        "",
        "## 6. Excluded cases - missing `bird_ex` audit",
        "",
        f"The loader skips per-case rows where the `metrics` dict has no `bird_ex` key. "
        f"**{n_excluded}** rows total are excluded; the table below lists every one.",
        "",
        df_to_md(excluded, floatfmt=".2f"),
        "",
        "## 7. Headline takeaways",
        "",
        "- Paired McNemar p-values together with question-level AND database-level bootstrap "
        "CIs give the inferential backbone for the L4 effect. They replace aggregate "
        "percentage-point claims with significance tests robust to both question-sampling "
        "and database-sampling assumptions.",
        "- The HF-lane logit `C(level)` coefficients are log-odds contrasts of each "
        "L4-LOO against the L0-PAD reference. The LOO table in section 4 is the "
        "primary reference for the \"which dimension matters most\" question.",
        "- The database-level bootstrap CI is the cross-database generalization number to "
        "quote in the paper; the question-level CI is the within-database significance.",
        "",
    ]
    return "\n".join(parts)


def export_csv(mcn: pd.DataFrame, boot_q: pd.DataFrame, boot_db: pd.DataFrame,
               loo: pd.DataFrame, scoped_coef: pd.DataFrame | None,
               hf_coef: pd.DataFrame | None, csv_path: Path) -> None:
    """Export all stats tables to a single long CSV.

    Each row: table, model (or term), plus the table-specific fields. Useful for
    notebook 02 to render figures without re-running the analysis.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["table", "model", "term", "baseline", "n", "metric", "value"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _, r in mcn.iterrows():
            for metric in ("n_paired_obs", "agree", "baseline_wins_c", "L4_wins_b",
                           "delta_b_minus_c", "mcnemar_p"):
                w.writerow({
                    "table": "mcnemar", "model": r["model"], "term": "",
                    "baseline": r["baseline"], "n": r["n_paired_obs"],
                    "metric": metric, "value": r[metric],
                })
        for _, r in boot_q.iterrows():
            for metric in ("point_estimate_pp", "ci_lo_pp", "ci_hi_pp"):
                w.writerow({
                    "table": "bootstrap_question", "model": r["model"], "term": "",
                    "baseline": r["baseline"], "n": r["n_obs"],
                    "metric": metric, "value": r[metric],
                })
        for _, r in boot_db.iterrows():
            for metric in ("point_estimate_pp", "ci_lo_pp", "ci_hi_pp"):
                w.writerow({
                    "table": "bootstrap_database", "model": r["model"], "term": "",
                    "baseline": r["baseline"], "n": r["n_databases"],
                    "metric": metric, "value": r[metric],
                })
        for _, r in loo.iterrows():
            for metric in ("delta_pp", "L0PAD_wins_c", "dim_wins_b", "p_raw", "p_holm",
                           "p_holm_lt_05"):
                w.writerow({
                    "table": "loo_ablation", "model": r["model"],
                    "term": r["dimension_dropped"], "baseline": "L0-PAD",
                    "n": r["n"], "metric": metric, "value": r[metric],
                })
        for label, coef_df in (("logit_scoped", scoped_coef), ("logit_hf", hf_coef)):
            if coef_df is None or coef_df.empty:
                continue
            for _, r in coef_df.iterrows():
                for metric in ("coef", "stderr", "z", "p", "odds_ratio"):
                    w.writerow({
                        "table": label, "model": "", "term": r["term"],
                        "baseline": "", "n": "", "metric": metric, "value": r[metric],
                    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", type=str, default=None,
                    help="Optional path to also write a long-form CSV export of all stats tables.")
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_long_dataframe()
    print(f"Loaded {len(df):,} per-question rows across "
          f"{df['model'].nunique()} models, {df['database'].nunique()} dbs, "
          f"{df['level'].nunique()} levels.")
    if df.empty:
        print("No data - verify * result dirs exist.")
        return

    mcn = mcnemar_baseline_vs_l4(df)
    boot_q = bootstrap_question_delta(df)
    boot_db = bootstrap_database_delta(df)
    loo = loo_ablation_table(df)
    scoped_coef, scoped_note = scoped_logit(df)
    hf_coef, hf_note = hf_logit(df)
    excluded = audit_missing_bird_ex()

    md = render_report(df, mcn, boot_q, boot_db, loo, scoped_coef, scoped_note,
                       hf_coef, hf_note, excluded)
    OUT_PATH.write_text(md)
    print(f"Wrote {OUT_PATH}")

    if args.csv:
        export_csv(mcn, boot_q, boot_db, loo, scoped_coef, hf_coef, Path(args.csv))
        print(f"Wrote {args.csv}")

    print()
    print("McNemar (baseline vs L4-effective):")
    print(mcn.to_string(index=False))
    print()
    print("Bootstrap (observation-level):")
    print(boot_q.to_string(index=False))
    print()
    print("Bootstrap (database-level):")
    print(boot_db.to_string(index=False))
    print()
    print(f"Excluded cases: {len(excluded)}")


if __name__ == "__main__":
    main()
