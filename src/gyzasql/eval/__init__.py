"""Eval package - runner only in this vendored copy.

Upstream gyzasql also exports synthetic-suite generators; those modules
(generate.py, llm_judge.py, synthetic.py, metrics.py) were dropped from
this vendor (see src/gyzasql/_VENDORED.md). The metadata-ablation experiment
uses pre-generated BIRD-mini-dev cases via runner.load_eval_cases.
"""
