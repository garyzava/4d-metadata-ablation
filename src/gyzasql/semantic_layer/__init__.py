"""Semantic metadata layer: store, context, schema filtering.

Trimmed for the metadata-ablation experiment. Upstream gyzasql also exports
scoring (CompletenessResult, compute_completeness, compute_and_save) and
classify; those modules were dropped from this vendor - compute_completeness
is gyzasql-internal IP that is not a paper metric, and the classifier and
vector_sync are not in the experiment's import closure (see _VENDORED.md).
"""

from gyzasql.semantic_layer.context import get_context, get_workspace_context
from gyzasql.semantic_layer.store import MetadataStore

__all__ = [
    "MetadataStore",
    "get_context",
    "get_workspace_context",
]
