"""Conformal selective-risk control (PLAN.md §10.6, RQ8) — the guarantee.

Most QA systems pick an abstention threshold to make a curve look nice.  This
one **controls the risk with a distribution-free guarantee**:

1. nonconformity ``s(x) = 1 - calibrated_confidence(x)`` on a **calibration
   split carved from val**, never from test;
2. for each candidate threshold τ, the **empirical selective risk** — the error
   rate among *answered* questions — and a one-sided **Clopper-Pearson upper
   confidence bound** at level δ;
3. choose the τ that **maximises coverage** subject to ``UCB(risk) <= α``.

With α = 0.10 and δ = 0.10 the claim is: *with probability at least 90% over
the calibration draw, the error rate among answered questions is at most 10%.*

Then it is verified on test **once**, reporting the achieved selective risk and
coverage and whether the bound held.  "We guarantee ≤10% error on answered
questions with 90% confidence; on test we achieved 8.3% at 71% coverage" is
worth more than any single F1 number in this project — and unlike an F1 it is a
*contract*, which is the right engineering object for a student-facing tool.

Why Clopper-Pearson rather than a normal approximation: the selective risk is a
binomial proportion measured on a few hundred answered items and often close to
0, exactly where the normal interval is anti-conservative.  An exact bound
cannot silently under-cover the thing we are promising.

*(Geifman & El-Yaniv's selective-risk framing; Angelopoulos & Bates for the
conformal machinery.)*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import CFG, FIGURES


# --------------------------------------------------------------------------- #
# The exact upper bound                                                        #
# --------------------------------------------------------------------------- #


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial proportion."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - delta, errors + 1, n - errors))


# --------------------------------------------------------------------------- #
# Threshold selection                                                          #
# --------------------------------------------------------------------------- #


@dataclass
class ConformalThreshold:
    tau: float
    alpha: float
    delta: float
    calibration_n: int
    answered: int
    coverage: float
    empirical_risk: float
    risk_upper_bound: float
    feasible: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def decide(self, confidence: float) -> bool:
        """``True`` = answer, ``False`` = abstain."""
        return confidence >= self.tau


def select_threshold(confidence: Sequence[float], correct: Sequence[bool], *,
                     alpha: float | None = None, delta: float | None = None,
                     grid: int = 201) -> tuple[ConformalThreshold, list[dict]]:
    """Maximise coverage subject to ``UCB(selective risk) <= alpha``."""
    alpha = CFG.conformal_alpha if alpha is None else alpha
    delta = CFG.conformal_delta if delta is None else delta

    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    n = len(conf)

    curve: list[dict] = []
    best: ConformalThreshold | None = None
    for tau in np.linspace(0.0, 1.0, grid):
        answered = conf >= tau
        n_ans = int(answered.sum())
        if n_ans == 0:
            continue
        errors = int((~ok[answered]).sum())
        risk = errors / n_ans
        ucb = clopper_pearson_upper(errors, n_ans, delta)
        coverage = n_ans / n
        curve.append({"tau": round(float(tau), 4), "coverage": round(coverage, 4),
                      "selective_risk": round(risk, 4), "risk_ucb": round(ucb, 4),
                      "answered": n_ans, "errors": errors,
                      "feasible": bool(ucb <= alpha)})
        if ucb <= alpha and (best is None or coverage > best.coverage):
            best = ConformalThreshold(
                tau=float(tau), alpha=alpha, delta=delta, calibration_n=n,
                answered=n_ans, coverage=coverage, empirical_risk=risk,
                risk_upper_bound=ucb, feasible=True)

    if best is None:
        # No threshold satisfies the bound.  This is reported, never papered
        # over: it means the system cannot honour a 10% promise on this data,
        # and the honest response is to say so and show the tightest τ.
        tau = 1.0
        best = ConformalThreshold(tau=tau, alpha=alpha, delta=delta, calibration_n=n,
                                  answered=0, coverage=0.0, empirical_risk=0.0,
                                  risk_upper_bound=1.0, feasible=False)
    return best, curve


# --------------------------------------------------------------------------- #
# Verification on test — run once                                              #
# --------------------------------------------------------------------------- #


def verify(threshold: ConformalThreshold, confidence: Sequence[float],
           correct: Sequence[bool]) -> dict:
    """Achieved selective risk and coverage at the chosen τ."""
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    answered = conf >= threshold.tau
    n_ans = int(answered.sum())
    errors = int((~ok[answered]).sum()) if n_ans else 0
    risk = errors / n_ans if n_ans else 0.0

    # **A bound satisfied at zero coverage is not satisfied.**  When no
    # threshold was feasible, tau is 1.0, nothing is answered, and the
    # selective risk is trivially 0 — reporting "bound HELD" there would be
    # the single most misleading number this project could print, because the
    # guarantee is about the questions we *do* answer.  Say so instead.
    vacuous = n_ans == 0
    if vacuous or not threshold.feasible:
        claim = (f"NOT IN FORCE — no threshold met the "
                 f"≤{100 * threshold.alpha:.0f}% risk bound on the calibration "
                 f"split, so the system answers nothing under this policy")
    else:
        claim = (f"with {100 * (1 - threshold.delta):.0f}% confidence, at most "
                 f"{100 * threshold.alpha:.0f}% of answered questions are wrong")

    return {
        "tau": threshold.tau,
        "alpha": threshold.alpha,
        "delta": threshold.delta,
        "n_test": len(conf),
        "answered": n_ans,
        "coverage": round(n_ans / max(len(conf), 1), 4),
        "achieved_selective_risk": round(risk, 4),
        "errors_among_answered": errors,
        "guarantee_in_force": bool(threshold.feasible and not vacuous),
        "bound_held": (None if vacuous else bool(risk <= threshold.alpha)),
        "vacuous_zero_coverage": bool(vacuous),
        "claim": claim,
    }


# --------------------------------------------------------------------------- #
# The abstention-policy comparison (RQ8b)                                      #
# --------------------------------------------------------------------------- #


def risk_coverage_curve(confidence: Sequence[float], correct: Sequence[bool],
                        *, points: int = 101) -> list[dict]:
    """Selective risk as a function of coverage, for one policy."""
    conf = np.asarray(confidence, dtype=float)
    ok = np.asarray(correct, dtype=bool)
    order = np.argsort(-conf)
    ok_sorted = ok[order]
    rows: list[dict] = []
    n = len(conf)
    for frac in np.linspace(1.0 / max(n, 1), 1.0, points):
        k = max(int(round(frac * n)), 1)
        risk = 1.0 - ok_sorted[:k].mean()
        rows.append({"coverage": round(k / n, 4), "selective_risk": round(float(risk), 4)})
    return rows


def curve_auc(curve: Sequence[dict]) -> float:
    """Area under the risk-coverage curve.  **Lower is better.**"""
    if len(curve) < 2:
        return float("nan")
    x = np.asarray([r["coverage"] for r in curve])
    y = np.asarray([r["selective_risk"] for r in curve])
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def compare_policies(policies: dict[str, tuple[Sequence[float], Sequence[bool]]]
                     ) -> tuple[list[dict], dict[str, list[dict]]]:
    """RQ8b's four arms: confidence only · fused · fused+veto · +conformal."""
    rows: list[dict] = []
    curves: dict[str, list[dict]] = {}
    for name, (conf, ok) in policies.items():
        curve = risk_coverage_curve(conf, ok)
        curves[name] = curve
        rows.append({
            "policy": name,
            "rc_auc": round(curve_auc(curve), 4),
            "risk_at_full_coverage": curve[-1]["selective_risk"],
            "risk_at_50pct_coverage": next(
                (r["selective_risk"] for r in curve if r["coverage"] >= 0.5), None),
            "risk_at_80pct_coverage": next(
                (r["selective_risk"] for r in curve if r["coverage"] >= 0.8), None),
        })
    rows.sort(key=lambda r: r["rc_auc"])
    return rows, curves


def risk_coverage_figure(curves: dict[str, list[dict]], *, tau_point: dict | None = None,
                         filename: str = "risk_coverage.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for name, curve in curves.items():
        ax.plot([r["coverage"] for r in curve], [r["selective_risk"] for r in curve],
                lw=1.8, label=name)
    ax.axhline(CFG.conformal_alpha, color="#c53030", ls="--", lw=1,
               label=f"α = {CFG.conformal_alpha}")
    if tau_point:
        ax.plot([tau_point["coverage"]], [tau_point["achieved_selective_risk"]],
                "k*", ms=14, label="conformal τ (test)")
    ax.set_xlabel("coverage — fraction of questions answered")
    ax.set_ylabel("selective risk — error rate among answered")
    ax.set_title("Risk–coverage (RQ8b)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
