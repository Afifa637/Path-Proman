"""Verification: the veto layer, five signals, fusion, calibration, conformal.

PLAN.md §10 — the headline contribution, and entirely Tier A.  The import order
matters: :mod:`constraints` depends on nothing but preprocessing, which is what
lets the veto layer be tested (and trusted) in isolation from every model.
"""

from .constraints import Veto, VetoResult, check  # noqa: F401
from .signals import SignalContext, compute_all  # noqa: F401
from .threshold import Decision, decide  # noqa: F401
