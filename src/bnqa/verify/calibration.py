"""Calibration: uncalibrated vs temperature vs Platt vs isotonic (§10.5).

A confidence bar that says 0.9 must mean "right about nine times in ten", or
every downstream promise — the abstention threshold, the conformal guarantee,
the number the student reads — is decoration.  So four calibrators are compared
on **ECE, ACE and a reliability diagram**, fitted on val and never on test.

ECE and ACE are both reported because they disagree in a way that matters.
ECE uses equal-**width** bins, so a model whose confidences cluster in one
region gets most of its mass in one bin and looks well calibrated; ACE uses
equal-**mass** bins and does not.  Quoting only ECE is a known way to make a
miscalibrated model look fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import CFG, FIGURES

METHODS = ("uncalibrated", "temperature", "platt", "isotonic")


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ECE — equal-**width** bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def adaptive_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ACE — equal-**mass** bins, which equal-width binning can hide."""
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    total = 0.0
    for chunk in chunks:
        if not len(chunk):
            continue
        total += (len(chunk) / len(p)) * abs(y[chunk].mean() - p[chunk].mean())
    return float(total)


def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        rows.append({
            "bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
            "n": int(mask.sum()),
            "mean_confidence": round(float(p[mask].mean()), 4) if mask.any() else None,
            "empirical_accuracy": round(float(y[mask].mean()), 4) if mask.any() else None,
        })
    return rows


# --------------------------------------------------------------------------- #
# Calibrators                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class Calibrator:
    method: str
    _fn: object = None

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self.method == "uncalibrated" or self._fn is None:
            return np.clip(p, 1e-6, 1 - 1e-6)
        return np.clip(self._fn(p), 1e-6, 1 - 1e-6)  # type: ignore[operator]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray) -> Calibrator:
    if method == "uncalibrated":
        return Calibrator(method)

    if method == "temperature":
        # One parameter, fitted by a coarse-to-fine sweep on NLL.  A scalar
        # temperature cannot change the ranking, only the spread — which is
        # why it is the honest first thing to try.
        z = _logit(p)
        best_t, best_nll = 1.0, float("inf")
        for t in np.concatenate([np.linspace(0.05, 5.0, 100), np.linspace(0.5, 2.0, 150)]):
            q = _sigmoid(z / t)
            nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
            if nll < best_nll:
                best_t, best_nll = float(t), float(nll)
        return Calibrator(method, lambda x, t=best_t: _sigmoid(_logit(x) / t))

    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000, random_state=CFG.seed)
        lr.fit(_logit(p).reshape(-1, 1), y)
        return Calibrator(method,
                          lambda x, m=lr: m.predict_proba(_logit(x).reshape(-1, 1))[:, 1])

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        return Calibrator(method, lambda x, m=iso: m.predict(x))

    raise ValueError(f"unknown calibration method {method!r}")


def compare(p_fit: np.ndarray, y_fit: np.ndarray, p_eval: np.ndarray,
            y_eval: np.ndarray) -> tuple[list[dict], dict[str, Calibrator]]:
    """The calibration study table, plus the fitted calibrators."""
    from sklearn.metrics import brier_score_loss

    rows: list[dict] = []
    fitted: dict[str, Calibrator] = {}
    for method in METHODS:
        cal = fit_calibrator(method, p_fit, y_fit)
        q = cal(p_eval)
        rows.append({
            "method": method,
            "ece": round(expected_calibration_error(q, y_eval), 4),
            "ace": round(adaptive_calibration_error(q, y_eval), 4),
            "brier": round(float(brier_score_loss(y_eval, q)), 4),
            "mean_confidence": round(float(q.mean()), 4),
            "empirical_accuracy": round(float(y_eval.mean()), 4),
        })
        fitted[method] = cal
    rows.sort(key=lambda r: r["ece"])
    return rows, fitted


# --------------------------------------------------------------------------- #
# The reliability diagram                                                      #
# --------------------------------------------------------------------------- #


def reliability_figure(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                       *, filename: str = "reliability.png", bins: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#718096", lw=1, label="perfect calibration")
    for name, (p, y) in curves.items():
        rows = [r for r in reliability_bins(p, y, bins) if r["n"]]
        ax.plot([r["mean_confidence"] for r in rows],
                [r["empirical_accuracy"] for r in rows], "o-", label=name, lw=1.6)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram (val)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
