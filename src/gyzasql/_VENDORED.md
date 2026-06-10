# Vendored engine: provenance and patches

`src/gyzasql/` is a self-contained, trimmed copy of the text-to-SQL agent used as
the system-under-test for this experiment. It is vendored (committed directly into
this repo) so the experiment runs from `uv sync` alone, with no external package
dependency and no network fetch of the engine. License: MIT.

The vendor contains only the modules the experiment actually exercises: the
Postgres connector, the evaluation runner, the semantic-layer store, the
orchestrator, and the LLM client. Components that are never invoked by the ablation
(the HTTP API server, the CLI, the MCP server, alternative LLM backends, the
Qdrant/RAG indexing path, the LLM-as-judge scorer, and the synthetic test-case
generators) are not included, which also keeps the dependency set small.

## Patches that affect measurement

Two behavioral edits are load-bearing for the results and are documented here so
they are reproducible:

1. **SQL extraction** (`tools/text2sql.py`). `_extract_sql()` pulls the final SQL
   statement out of the model response. It (1) takes the last fenced code block if
   one is present, (2) otherwise takes the last column-0 `SELECT`/`WITH` through the
   end of the response, (3) otherwise falls back to a plain fence-strip. This is
   needed for models that emit a long block of inline reasoning prose before the
   SQL; a naive fence-strip yields 0% accuracy for those outputs. Verified to be a
   no-op on models that already return clean SQL.

2. **Opus-4.7 request shaping** (`llm/client.py`). For `claude-opus-4-7+` the client
   skips the deprecated `temperature` / `top_p` / `top_k` sampling fields (the API
   rejects them) and skips the legacy extended-thinking parameter (the newer API
   uses a different shape). The model answers SQL well without extended thinking,
   and skipping it also avoids thinking-token billing. Both behaviors are guarded by
   the same model-prefix check.

## Dormant lazy import

`eval/runner.py` lazy-imports an LLM-as-judge scorer only when the `--judge` flag is
passed to `run_ablation.py` (off by default). That scorer module is not part of this
vendor; the experiment never uses it, so the import stays dormant. Do not enable
`--judge` against this vendored copy.
