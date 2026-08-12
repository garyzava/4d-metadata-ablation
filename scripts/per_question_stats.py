"""Per-question paired statistical analysis on data.

Addresses the review point "aggregate database-level percentages are not enough"
by pairing the same question under two conditions and testing the paired
difference. All pairing is by (database, question text), never by row position.

Lane structure (verified against results/):
  - Dimensional lanes  (9b, 27b, 30b): full ladder L0, L0-PAD, L1..L4,
    four leave-one-out conditions (L4-DD/QP/BC/DK), and EVIDENCE.
  - Cross-vendor lanes (gemma4, qwen36, opus47): L0, L0-PAD, L4.

Primary contrasts (camera-ready design):
  - H1: native L4 vs L0-PAD, per model, all six models. Two-sided exact
    McNemar, Holm-corrected across the six tests. No pooling: every model
    has a native L4 condition, so the earlier pooled treatment (L0-PAD
    against the four leave-one-out conditions) is retired from inference.
  - Padding effect: L0 vs L0-PAD, per model, all six models. Two-sided
    exact McNemar, separate Holm correction across the six tests.
  - H3 drop cost: native L4 vs each L4-minus-dimension condition, the three
    dimensional Qwen lanes only. Two metric families (BEX and ER), each a
    12-test Holm family (3 models x 4 dimensions).
  - The defined-condition LOO table (L0-PAD vs each leave-one-out condition)
    is retained as a descriptive companion, not as the H1 evidence.

EVIDENCE rows are never paired: that condition stores the question with the
BIRD hint prepended, so its question text differs by design.

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

EXPECTED_CELLS = 462
EXPECTED_RECORDS = 21_000
EXPECTED_QUESTIONS = 500


def load_long_dataframe() -> pd.DataFrame:
    """Walk all per-cell result JSONs into one long-format DataFrame.

    One row per (model, database, level, question). The question text is the
    pairing key within a database; it is unique per database and identical
    across all non-EVIDENCE levels (checked by acceptance_checks).
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
                    "question": c.get("question", ""),
                    "difficulty": c.get("difficulty", "unknown"),
                    "bird_ex": int(bool(m.get("bird_ex", False))),
                    "value_match": int(bool(m.get("value_match", False))),
                    "soft_f1": float(m.get("soft_f1", 0.0)),
                    "execution_success": int(bool(c.get("execution_success", False))),
                })
    return pd.DataFrame(rows)


def count_result_cells() -> int:
    """Count per-cell result JSONs (one file per model x condition x database)."""
    n = 0
    for dir_path in RESULTS_DIRS.values():
        if not dir_path.exists():
            continue
        n += sum(1 for jp in dir_path.glob("*.json")
                 if jp.name != "experiment_summary.json")
    return n


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


def _pair(df: pd.DataFrame, model: str, base_level: str, treat_level: str,
          col: str = "bird_ex") -> pd.DataFrame:
    """Pair one model's questions under base_level vs treat_level on `col`.

    Merge key is (database, question). Returns columns: database, question,
    y_base, y_treat.
    """
    sub = df[df["model"] == model]
    base = sub[sub["level"] == base_level][
        ["database", "question", col]
    ].rename(columns={col: "y_base"})
    treat = sub[sub["level"] == treat_level][
        ["database", "question", col]
    ].rename(columns={col: "y_treat"})
    return base.merge(treat, on=["database", "question"], how="inner")


def _holm(p_raw: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjusted p-values, order-preserving.

    Returns (p_holm, significant_at_05) aligned to the input order.
    """
    p = p_raw.to_numpy(dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.minimum(1.0, p[order] * (m - np.arange(m)))
    adj_sorted = np.maximum.accumulate(adj_sorted)
    p_holm = np.empty(m)
    p_holm[order] = adj_sorted
    return p_holm, p_holm < 0.05


def _mcnemar_rows(df: pd.DataFrame, models: list[str], base_level: str,
                  treat_level: str, col: str = "bird_ex") -> pd.DataFrame:
    """Two-sided exact McNemar per model for one (base, treat) contrast.

    b = base wrong, treat right; c = base right, treat wrong. The effect
    size is the discordant odds ratio b/c and the paired delta in
    percentage points.
    """
    out: list[dict] = []
    for model in [m for m in models if m in df["model"].unique()]:
        pair = _pair(df, model, base_level, treat_level, col)
        n = len(pair)
        if n == 0:
            continue
        b = int(((pair["y_base"] == 0) & (pair["y_treat"] == 1)).sum())
        c = int(((pair["y_base"] == 1) & (pair["y_treat"] == 0)).sum())
        delta_pp = float((pair["y_treat"] - pair["y_base"]).mean()) * 100
        if b + c > 0:
            p = stats.binomtest(min(b, c), n=b + c, p=0.5,
                                alternative="two-sided").pvalue
        else:
            p = 1.0
        odds = round(b / c, 2) if c > 0 else float("inf")
        out.append({
            "model": model,
            "baseline": base_level,
            "treatment": treat_level,
            "n_questions": n,
            "agree": n - b - c,
            "b_treat_wins": b,
            "c_base_wins": c,
            "delta_pp": round(delta_pp, 2),
            "odds_ratio": odds,
            "p_exact": p,
        })
    res = pd.DataFrame(out)
    if not res.empty:
        p_holm, sig = _holm(res["p_exact"])
        res["p_holm"] = p_holm
        res["holm_sig_05"] = sig
    return res


def mcnemar_l4_vs_l0pad(df: pd.DataFrame) -> pd.DataFrame:
    """H1 primary test: native L4 vs L0-PAD, all six models, Holm across 6."""
    return _mcnemar_rows(df, ALL_MODELS, "L0-PAD", "L4", "bird_ex")


def mcnemar_padding(df: pd.DataFrame) -> pd.DataFrame:
    """Padding effect: L0 vs L0-PAD, all six models, separate Holm across 6."""
    return _mcnemar_rows(df, ALL_MODELS, "L0", "L0-PAD", "bird_ex")


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    """Database-macro BEX (%) at L0, L0-PAD, and native L4, with both deltas.

    Database-macro: mean over the 11 per-database means, equal weight per
    database. This matches the paper's reporting unit.
    """
    out: list[dict] = []
    for model in [m for m in ALL_MODELS if m in df["model"].unique()]:
        sub = df[df["model"] == model]
        vals: dict[str, float] = {}
        for lvl in ("L0", "L0-PAD", "L4"):
            g = sub[sub["level"] == lvl]
            if g.empty:
                vals[lvl] = float("nan")
                continue
            vals[lvl] = float(g.groupby("database")["bird_ex"].mean().mean()) * 100
        out.append({
            "model": model,
            "L0": round(vals["L0"], 1),
            "L0-PAD": round(vals["L0-PAD"], 1),
            "L4": round(vals["L4"], 1),
            "delta_L4_minus_L0": round(vals["L4"] - vals["L0"], 1),
            "delta_L4_minus_L0PAD": round(vals["L4"] - vals["L0-PAD"], 1),
        })
    return pd.DataFrame(out)


def bootstrap_question_delta(df: pd.DataFrame, n_resamples: int = 10000,
                             seed: int = 42) -> pd.DataFrame:
    """Bootstrap 95% CI on the L0-PAD -> native L4 BEX delta per model.

    Resamples the paired questions (one row per question, all six models).
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for model in [m for m in ALL_MODELS if m in df["model"].unique()]:
        pair = _pair(df, model, "L0-PAD", "L4", "bird_ex")
        if pair.empty:
            continue
        deltas = (pair["y_treat"] - pair["y_base"]).to_numpy()
        n = len(deltas)
        point = float(np.mean(deltas)) * 100
        idx = rng.integers(0, n, size=(n_resamples, n))
        boot_means = deltas[idx].mean(axis=1) * 100
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        out.append({
            "model": model,
            "baseline": "L0-PAD",
            "n_obs": n,
            "point_estimate_pp": round(point, 2),
            "ci_lo_pp": round(float(lo), 2),
            "ci_hi_pp": round(float(hi), 2),
        })
    return pd.DataFrame(out)


def bootstrap_database_delta(df: pd.DataFrame, n_resamples: int = 10000,
                             seed: int = 42) -> pd.DataFrame:
    """Bootstrap 95% CI on the L0-PAD -> native L4 BEX delta, resampling DATABASES."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for model in [m for m in ALL_MODELS if m in df["model"].unique()]:
        pair = _pair(df, model, "L0-PAD", "L4", "bird_ex")
        if pair.empty:
            continue
        per_db = pair.assign(delta=pair["y_treat"] - pair["y_base"]) \
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
            "baseline": "L0-PAD",
            "n_databases": n_dbs,
            "point_estimate_pp": round(point, 2),
            "ci_lo_pp": round(float(lo), 2),
            "ci_hi_pp": round(float(hi), 2),
        })
    return pd.DataFrame(out)


def loo_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive LOO companion: L0-PAD vs each leave-one-out condition.

    Answers "does a three-dimension configuration still beat the distractor
    control", per (dimensional model, dropped dimension). Holm across the 12
    comparisons. This is not the H1 evidence and not the drop-cost table.
    """
    raw: list[dict] = []
    for model in HF_MODELS:
        for dim in L4_LOO:
            pair = _pair(df, model, "L0-PAD", dim, "bird_ex")
            n = len(pair)
            if n == 0:
                continue
            b = int(((pair["y_base"] == 0) & (pair["y_treat"] == 1)).sum())
            c = int(((pair["y_base"] == 1) & (pair["y_treat"] == 0)).sum())
            delta = float((pair["y_treat"] - pair["y_base"]).mean()) * 100
            if b + c > 0:
                p_raw = stats.binomtest(min(b, c), n=b + c, p=0.5,
                                        alternative="two-sided").pvalue
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
    p_holm, sig = _holm(out["p_raw"])
    out["p_holm"] = p_holm
    out["p_holm_lt_05"] = sig
    return out.sort_values(["model", "dimension_dropped"]).reset_index(drop=True)


def loo_drop_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """H3 drop cost: native L4 vs each L4-minus-dimension, dimensional lanes only.

    One 12-test Holm family per metric (col = bird_ex for BEX, or
    execution_success for ER). delta_pp = mean(LOO) - mean(L4): a negative
    value means removing the dimension costs accuracy. b = L4 wrong but LOO
    right; c = L4 right but LOO wrong.
    """
    raw: list[dict] = []
    for model in HF_MODELS:
        for dim in L4_LOO:
            pair = _pair(df, model, "L4", dim, col)
            n = len(pair)
            if n == 0:
                continue
            b = int(((pair["y_base"] == 0) & (pair["y_treat"] == 1)).sum())
            c = int(((pair["y_base"] == 1) & (pair["y_treat"] == 0)).sum())
            delta = float((pair["y_treat"] - pair["y_base"]).mean()) * 100
            if b + c > 0:
                p_raw = stats.binomtest(min(b, c), n=b + c, p=0.5,
                                        alternative="two-sided").pvalue
            else:
                p_raw = 1.0
            raw.append({
                "model": model,
                "dimension_dropped": dim,
                "metric": "BEX" if col == "bird_ex" else "ER",
                "n": n,
                "delta_pp": round(delta, 2),
                "L4_wins_c": c,
                "loo_wins_b": b,
                "p_raw": p_raw,
            })
    out = pd.DataFrame(raw)
    if out.empty:
        return out
    p_holm, sig = _holm(out["p_raw"])
    out["p_holm"] = p_holm
    out["p_holm_lt_05"] = sig
    return out.sort_values(["model", "dimension_dropped"]).reset_index(drop=True)


def scoped_logit(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Logit on the Cross-vendor lanes: bird_ex ~ C(level) + C(model) + C(difficulty)
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
    """Logit on dimensional lanes: bird_ex ~ C(level) + C(model) + C(difficulty)
    with database-cluster-robust SE. Levels included: L0-PAD plus the four
    leave-one-out conditions, under their own definitions. Reference: L0-PAD.
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
    return pd.DataFrame(rows), f"Dimensional lanes; logit on L0-PAD + leave-one-out conditions with cluster-robust SE on database (n={len(sub)})"


def cumulative_logit(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Logit on the dimensional lanes over the cumulative ladder L0..L4.

    Ported from the original round analysis (hf_monolithic_logit). Fits
    bird_ex ~ C(level) + C(model) + C(difficulty) with cluster-robust SE
    on database, restricted to levels {L0, L1, L2, L3, L4}. Reference: L0.
    """
    levels = ["L0", "L1", "L2", "L3", "L4"]
    sub = df[(df["lane"] == "HF") & (df["level"].isin(levels))].copy()
    if sub.empty:
        return None, "no HF rows for the cumulative L0..L4 levels"
    sub["level"] = pd.Categorical(sub["level"], categories=levels, ordered=True)
    sub["model"] = pd.Categorical(sub["model"], categories=HF_MODELS)
    sub["difficulty"] = pd.Categorical(sub["difficulty"], categories=sorted(sub["difficulty"].unique()))
    try:
        m = smf.logit("bird_ex ~ C(level) + C(model) + C(difficulty)", data=sub).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": sub["database"]},
        )
    except Exception as e:
        return None, f"cumulative logit fit failed: {e}"
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
    return pd.DataFrame(rows), f"Dimensional lanes; logit on the cumulative ladder L0..L4 with cluster-robust SE on database (n={len(sub)})"


def acceptance_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Camera-ready acceptance checks. Every row must PASS."""
    rows: list[dict] = []

    def add(name: str, expected, actual) -> None:
        rows.append({"check": name, "expected": str(expected),
                     "actual": str(actual),
                     "status": "PASS" if str(expected) == str(actual) else "FAIL"})

    add("result cells (JSON files)", EXPECTED_CELLS, count_result_cells())
    add("question-level records", EXPECTED_RECORDS, len(df))

    # Unique questions per direct comparison must be exactly 500.
    comp_ns: set[int] = set()
    for model in [m for m in ALL_MODELS if m in df["model"].unique()]:
        comp_ns.add(len(_pair(df, model, "L0-PAD", "L4")))
        comp_ns.add(len(_pair(df, model, "L0", "L0-PAD")))
    for model in HF_MODELS:
        for dim in L4_LOO:
            comp_ns.add(len(_pair(df, model, "L4", dim)))
            comp_ns.add(len(_pair(df, model, "L0-PAD", dim)))
    add("questions per direct comparison", f"{{{EXPECTED_QUESTIONS}}}", str(set(sorted(comp_ns))))

    # Pairing keys must be unique: no duplicate (model, level, database, question).
    dup = int(df.duplicated(subset=["model", "level", "database", "question"]).sum())
    add("duplicate pairing keys", 0, dup)

    # Non-EVIDENCE levels must share one question set per (model, database).
    mismatched = 0
    non_ev = df[df["level"] != "EVIDENCE"]
    for (model, db), g in non_ev.groupby(["model", "database"]):
        sets = g.groupby("level")["question"].apply(frozenset).unique()
        if len(sets) > 1:
            mismatched += 1
    add("(model, database) groups with level question-set mismatch", 0, mismatched)

    return pd.DataFrame(rows)


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


def render_report(df: pd.DataFrame, headline: pd.DataFrame, mcn: pd.DataFrame,
                  pad: pd.DataFrame, boot_q: pd.DataFrame, boot_db: pd.DataFrame,
                  loo: pd.DataFrame, drop_bex: pd.DataFrame, drop_er: pd.DataFrame,
                  scoped_coef: pd.DataFrame | None, scoped_note: str,
                  hf_coef: pd.DataFrame | None, hf_note: str,
                  cum_coef: pd.DataFrame | None, cum_note: str,
                  checks: pd.DataFrame, excluded: pd.DataFrame) -> str:
    n_rows = len(df)
    n_models = df["model"].nunique()
    n_dbs = df["database"].nunique()
    n_levels = df["level"].nunique()
    n_excluded = len(excluded)

    parts = [
        "# Per-question statistical analysis",
        "",
        "Generated by `scripts/per_question_stats.py`. Walks the existing "
        "per-case JSONs (no new LLM calls). All paired tests match the same "
        "question by (database, question text) under two conditions. The "
        "primary H1 contrast is native L4 vs the L0-PAD distractor control; "
        "no pooled treatment is used anywhere in this report.",
        "",
        "## 1. Data shape",
        "",
        f"- Long-format rows: **{n_rows:,}** (across {n_models} models, {n_dbs} databases, {n_levels} levels).",
        f"- Levels covered: {sorted(df['level'].unique())}.",
        f"- Cases excluded for missing `bird_ex`: **{n_excluded}** (audit at the bottom).",
        "",
        "## 2. Headline table - database-macro BEX (%) at L0, L0-PAD, and native L4",
        "",
        "Equal weight per database (mean of the 11 per-database means).",
        "",
        df_to_md(headline, floatfmt=".1f"),
        "",
        "## 3. H1 primary test - paired McNemar, native L4 vs L0-PAD",
        "",
        "b = L0-PAD wrong but L4 right; c = L0-PAD right but L4 wrong. "
        "Two-sided exact binomial p, Holm-corrected across the six models. "
        "odds_ratio is the discordant ratio b/c. delta_pp is the paired "
        "question-pooled BEX delta.",
        "",
        df_to_md(mcn, floatfmt=".6f"),
        "",
        "## 4. Padding effect - paired McNemar, L0 vs L0-PAD",
        "",
        "b = L0 wrong but L0-PAD right; c = L0 right but L0-PAD wrong. "
        "A negative delta_pp means padding hurt BEX. Separate Holm "
        "correction across these six tests.",
        "",
        df_to_md(pad, floatfmt=".6f"),
        "",
        "## 5. Bootstrap 95% CI on the L0-PAD to native-L4 BEX delta (pp)",
        "",
        "**5a. Question-level (10,000 resamples per model)** - robustness to "
        "which questions were sampled.",
        "",
        df_to_md(boot_q, floatfmt=".2f"),
        "",
        "**5b. Database-level (10,000 resamples of the 11 databases per model)** - "
        "robustness to which databases are in the study. The wider CI is the more "
        "honest cross-database generalization claim.",
        "",
        df_to_md(boot_db, floatfmt=".2f"),
        "",
        "## 6. Leave-one-out vs the distractor control (descriptive companion)",
        "",
        "Each row pairs L0-PAD against a leave-one-out condition for a "
        "dimensional model. delta_pp > 0 means the three-dimension "
        "configuration still beats L0-PAD. Holm across the 12 comparisons. "
        "This table is descriptive; the H3 drop-cost evidence is section 7.",
        "",
        df_to_md(loo, floatfmt=".4f"),
        "",
        "## 7. H3 drop cost - native L4 vs each L4-minus-dimension",
        "",
        "Dimensional Qwen lanes only. delta_pp = LOO minus L4: negative "
        "means removing the dimension costs accuracy. BEX and ER are "
        "separate 12-test Holm families.",
        "",
        "**7a. BEX family.**",
        "",
        df_to_md(drop_bex, floatfmt=".4f"),
        "",
        "**7b. ER family.**",
        "",
        df_to_md(drop_er, floatfmt=".4f"),
        "",
        "## 8. Logistic regression with cluster-robust SE on database",
        "",
        "Three fits: one on the scoped lanes, two on the dimensional lanes "
        "(leave-one-out set and cumulative ladder), because the level set is "
        "lane-specific.",
        "",
        f"**8a. Scoped lanes.** _{scoped_note}_",
        "",
        df_to_md(scoped_coef if scoped_coef is not None else pd.DataFrame(), floatfmt=".4f"),
        "",
        f"**8b. Dimensional lanes, leave-one-out set.** _{hf_note}_",
        "",
        df_to_md(hf_coef if hf_coef is not None else pd.DataFrame(), floatfmt=".4f"),
        "",
        f"**8c. Dimensional lanes, cumulative ladder.** _{cum_note}_",
        "",
        df_to_md(cum_coef if cum_coef is not None else pd.DataFrame(), floatfmt=".4f"),
        "",
        "## 9. Acceptance checks",
        "",
        df_to_md(checks),
        "",
        "## 10. Excluded cases - missing `bird_ex` audit",
        "",
        f"The loader skips per-case rows where the `metrics` dict has no `bird_ex` key. "
        f"**{n_excluded}** rows total are excluded; the table below lists every one.",
        "",
        df_to_md(excluded, floatfmt=".2f"),
        "",
    ]
    return "\n".join(parts)


def export_csv(headline: pd.DataFrame, mcn: pd.DataFrame, pad: pd.DataFrame,
               boot_q: pd.DataFrame, boot_db: pd.DataFrame, loo: pd.DataFrame,
               drop_bex: pd.DataFrame, drop_er: pd.DataFrame,
               scoped_coef: pd.DataFrame | None,
               hf_coef: pd.DataFrame | None,
               cum_coef: pd.DataFrame | None, csv_path: Path) -> None:
    """Export all stats tables to a single long CSV.

    Each row: table, model (or term), plus the table-specific fields. Useful for
    notebook 02 to render figures without re-running the analysis.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["table", "model", "term", "baseline", "n", "metric", "value"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _, r in headline.iterrows():
            for metric in ("L0", "L0-PAD", "L4", "delta_L4_minus_L0",
                           "delta_L4_minus_L0PAD"):
                w.writerow({
                    "table": "headline", "model": r["model"], "term": "",
                    "baseline": "", "n": "", "metric": metric, "value": r[metric],
                })
        for label, tbl in (("mcnemar", mcn), ("mcnemar_padding", pad)):
            for _, r in tbl.iterrows():
                for metric in ("n_questions", "agree", "b_treat_wins", "c_base_wins",
                               "delta_pp", "odds_ratio", "p_exact", "p_holm",
                               "holm_sig_05"):
                    w.writerow({
                        "table": label, "model": r["model"], "term": "",
                        "baseline": r["baseline"], "n": r["n_questions"],
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
        for label, tbl in (("loo_drop_bex", drop_bex), ("loo_drop_er", drop_er)):
            for _, r in tbl.iterrows():
                for metric in ("delta_pp", "L4_wins_c", "loo_wins_b", "p_raw",
                               "p_holm", "p_holm_lt_05"):
                    w.writerow({
                        "table": label, "model": r["model"],
                        "term": r["dimension_dropped"], "baseline": "L4",
                        "n": r["n"], "metric": metric, "value": r[metric],
                    })
        for label, coef_df in (("logit_scoped", scoped_coef), ("logit_hf", hf_coef),
                               ("logit_cumulative", cum_coef)):
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

    headline = headline_table(df)
    mcn = mcnemar_l4_vs_l0pad(df)
    pad = mcnemar_padding(df)
    boot_q = bootstrap_question_delta(df)
    boot_db = bootstrap_database_delta(df)
    loo = loo_ablation_table(df)
    drop_bex = loo_drop_table(df, "bird_ex")
    drop_er = loo_drop_table(df, "execution_success")
    scoped_coef, scoped_note = scoped_logit(df)
    hf_coef, hf_note = hf_logit(df)
    cum_coef, cum_note = cumulative_logit(df)
    checks = acceptance_checks(df)
    excluded = audit_missing_bird_ex()

    md = render_report(df, headline, mcn, pad, boot_q, boot_db, loo,
                       drop_bex, drop_er, scoped_coef, scoped_note,
                       hf_coef, hf_note, cum_coef, cum_note, checks, excluded)
    OUT_PATH.write_text(md)
    print(f"Wrote {OUT_PATH}")

    if args.csv:
        export_csv(headline, mcn, pad, boot_q, boot_db, loo, drop_bex, drop_er,
                   scoped_coef, hf_coef, cum_coef, Path(args.csv))
        print(f"Wrote {args.csv}")

    print()
    print("Headline (database-macro BEX %):")
    print(headline.to_string(index=False))
    print()
    print("H1 McNemar (native L4 vs L0-PAD):")
    print(mcn.to_string(index=False))
    print()
    print("Padding McNemar (L0 vs L0-PAD):")
    print(pad.to_string(index=False))
    print()
    print("H3 drop cost (BEX):")
    print(drop_bex.to_string(index=False))
    print()
    print("H3 drop cost (ER):")
    print(drop_er.to_string(index=False))
    print()
    print("Acceptance checks:")
    print(checks.to_string(index=False))
    print()
    print(f"Excluded cases: {len(excluded)}")


if __name__ == "__main__":
    main()
