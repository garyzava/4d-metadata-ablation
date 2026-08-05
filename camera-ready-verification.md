# Camera-ready verification report

Date: 2026-08-05. This report verifies the statistics used in the SYNASC camera-ready revision. Every number comes from the committed per-cell JSONs in `results/`. No new model calls were made.

To regenerate: run `notebooks/02-analysis-and-charts.ipynb`, or directly `uv run python scripts/per_question_stats.py --csv data/per-question-stats.csv`. The full statistical report is written to `data/reports/stats-tables.md` (regenerated, not committed).

All paired tests match the same question by (database, question text). The primary treatment is native L4 for every model. The earlier pooled treatment (L0-PAD paired against the four leave-one-out conditions) is retired from all inference.

## 1. Provenance audit (metadata construction inputs)

Counts from `bird-docs-revised/`, the metadata actually used in the experiment. DK = DomainContext, BC = BusinessContext. "Retained questions" counts benchmark questions kept verbatim as `_From:_` provenance lines in DK entries.

| Database | DK question lines | DK distinct questions | BC synonym rows (schema) | BC term rows (evidence) |
| :--- | ---: | ---: | ---: | ---: |
| california_schools | 17 | 16 | 37 | 24 |
| card_games | 17 | 16 | 55 | 100 |
| codebase_community | 38 | 33 | 45 | 64 |
| debit_card_specializing | 18 | 12 | 11 | 16 |
| european_football_2 | 41 | 31 | 60 | 88 |
| financial | 19 | 18 | 35 | 24 |
| formula_1 | 43 | 38 | 44 | 97 |
| student_club | 23 | 23 | 24 | 61 |
| superhero | 24 | 23 | 18 | 87 |
| thrombosis_prediction | 38 | 28 | 47 | 98 |
| toxicology | 31 | 23 | 6 | 53 |
| **Total** | **309** | **261 (52.2% of 500)** | **382** | **712** |

Schema reconciliation (from `bird-docs-revised/_revision-report.json`): 619 automatic replacements, 183 distinct original-to-canonical mappings, 25 files, 7 databases.

Gold SQL fields, executed gold results, predicted SQL, and model outcomes were not accessed during metadata construction. They were used later for evaluation and error analysis.

## 2. Headline: database-macro BEX (%) at L0, L0-PAD, and native L4

| Model | L0 | L0-PAD | L4 | Δ L4−L0 | Δ L4−L0-PAD |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 10.3 | 8.5 | 28.2 | +18.0 | +19.7 |
| Qwen3.5-27B | 14.7 | 14.6 | 35.6 | +20.9 | +21.0 |
| Qwen3-Coder-30B | 16.4 | 16.2 | 33.4 | +17.0 | +17.2 |
| Qwen3.6-35B-A3B | 17.4 | 19.1 | 37.3 | +19.8 | +18.1 |
| Gemma-4-26B | 15.1 | 12.2 | 29.0 | +13.9 | +16.9 |
| Claude Opus 4.7 | 32.2 | 27.8 | 44.4 | +12.3 | +16.7 |

ER-BEX gap at native L4 (pp): 9B 35.0, 27B 36.8, 30B 35.5, Qwen3.6 37.5, Gemma-4 30.0, Opus 34.6.

## 3. H1 primary test: paired McNemar, native L4 vs L0-PAD (n = 500 each)

Two-sided exact binomial p, Holm-corrected across the six models. b = L0-PAD wrong, L4 right; c = the reverse.

| Model | b | c | Δ pp | odds ratio | p (exact) | p (Holm) |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| Qwen3.5-9B | 103 | 6 | +19.4 | 17.2 | 6.6e-24 | 4.0e-23 |
| Qwen3.5-27B | 114 | 11 | +20.6 | 10.4 | 9.6e-23 | 4.8e-22 |
| Qwen3-Coder-30B | 97 | 12 | +17.0 | 8.1 | 1.1e-17 | 3.3e-17 |
| Gemma-4-26B | 94 | 11 | +16.6 | 8.5 | 1.4e-17 | 3.3e-17 |
| Qwen3.6-35B-A3B | 104 | 16 | +17.6 | 6.5 | 5.5e-17 | 5.5e-17 |
| Claude Opus 4.7 | 93 | 9 | +16.8 | 10.3 | 1.0e-18 | 4.0e-18 |

All six survive Holm correction.

## 4. Padding effect: paired McNemar, L0 vs L0-PAD (n = 500 each)

Separate Holm family. A negative Δ means padding hurt BEX.

| Model | b | c | Δ pp | p (exact) | p (Holm) | Significant |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| Qwen3.5-9B | 12 | 21 | −1.8 | 0.163 | 0.488 | no |
| Qwen3.5-27B | 13 | 12 | +0.2 | 1.000 | 1.000 | no |
| Qwen3-Coder-30B | 12 | 14 | −0.4 | 0.845 | 1.000 | no |
| Gemma-4-26B | 8 | 24 | −3.2 | 0.007 | 0.035 | **yes** |
| Qwen3.6-35B-A3B | 23 | 12 | +2.2 | 0.090 | 0.358 | no |
| Claude Opus 4.7 | 7 | 30 | −4.6 | 0.0002 | 0.001 | **yes** |

Padding significantly reduced BEX for Gemma-4-26B and Claude Opus 4.7.

## 5. Database-level bootstrap 95% CI on Δ BEX (L0-PAD to native L4)

10,000 resamples of the 11 databases. Every interval excludes zero.

| Model | Δ pp | 95% CI |
| :--- | ---: | :--- |
| Qwen3.5-9B | +19.7 | [12.4, 27.1] |
| Qwen3.5-27B | +21.0 | [11.9, 30.9] |
| Qwen3-Coder-30B | +17.2 | [9.1, 26.4] |
| Qwen3.6-35B-A3B | +18.1 | [9.3, 28.5] |
| Gemma-4-26B | +16.9 | [10.3, 24.2] |
| Claude Opus 4.7 | +16.7 | [9.1, 27.9] |

## 6. H3 drop cost: native L4 vs each L4-minus-dimension (dimensional Qwen lanes)

Two separate 12-test Holm families (BEX and ER). Δ pp = LOO minus L4; negative means removing the dimension costs accuracy.

BEX family, Holm-significant rows only:

| Model | Dropped | Δ pp | p (Holm) |
| :--- | :--- | ---: | :--- |
| Qwen3.5-9B | QP | −7.4 | 0.0003 |
| Qwen3.5-9B | BC | −5.2 | 0.044 |
| Qwen3.5-27B | BC | −6.2 | 0.002 |
| Qwen3.5-27B | QP | −4.4 | 0.021 |

No other BEX row survives correction (all 30B rows, all DD rows, all DK rows do not).

ER family, Holm-significant rows only:

| Model | Dropped | Δ pp | p (Holm) |
| :--- | :--- | ---: | :--- |
| Qwen3.5-9B | QP | −11.4 | 0.00001 |

No other ER row survives correction.

**H3 verdict: partly supported.** QP and BC have significant BEX effects for the 9B and 27B models. No BEX effect survives correction for the 30B model, DD, or DK. For ER, only the 9B QP effect survives correction.

## 7. Acceptance checks

| Check | Expected | Actual | Status |
| :--- | :--- | :--- | :--- |
| Result cells (JSON files) | 462 | 462 | PASS |
| Question-level records | 21,000 | 21,000 | PASS |
| Questions per direct comparison | 500 | 500 | PASS |
| Duplicate pairing keys | 0 | 0 | PASS |
| Level question-set mismatches (non-EVIDENCE) | 0 | 0 | PASS |
| Rows missing bird_ex (gold coverage) | 0 | 0 | PASS |

## 8. Changed artifacts

- `scripts/per_question_stats.py`: pooled treatment removed; native L4 everywhere; padding test, drop-cost tables, Holm families, and acceptance checks added.
- `scripts/question_taxonomy.py`: native L0 and L4 columns added for the dimensional lanes.
- `notebooks/02-analysis-and-charts.ipynb` (and executed copy): all figures use native L4; fig06 now compares L4 vs L4-QP; fig08 uses native L0 vs L4 for all six models.
- Figures changed: fig01, fig03, fig05, fig06, fig07, fig08, fig10, fig11, fig12. Unchanged: fig00, fig02, fig04, fig09.
