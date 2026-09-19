"""Evaluation — the only place a number in the report is allowed to come from.

VC-12: ``reports/results.json`` and everything in ``reports/tables/`` are
written **only** by this package and by the scripts that drive it, and every
row carries the config hash and the seed that produced it.
"""
