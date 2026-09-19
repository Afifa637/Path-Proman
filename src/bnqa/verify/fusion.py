"""Signal fusion — Naive Bayes vs Logistic Regression vs GBDT (§10.5, RQ5).

Three fusion models on **identical features**, fitted on **val only**.  That
comparison *is* RQ5, and it answers Sir's Q5 ("generative or discriminative?")
with a number instead of a paragraph:

* **Naive Bayes** is the generative arm.  Its independence assumption is
  visibly false here — S1 and S5 are both functions of the same reader — so
  the interesting result is *how much* that costs.
* **Logistic Regression** is the discriminative linear arm, and the one whose
  coefficients can be read out in the UI.
* **GBDT** is the discriminative non-linear arm, which can represent "high
  similarity **but** a veto fired" as an interaction rather than as a sum.

The veto layer is **not** one of the three's inputs to outrank: it is applied
before fusion as a cap (``VetoResult.cap``), because a contradiction must bound
support *regardless of similarity*, and a model that can trade it off against
a strong S2 will learn to do exactly that.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import CFG
from .signals import FUSION_FEATURE_NAMES, to_matrix

MODELS = ("naive_bayes", "logreg", "gbdt")


class FusionModel:
    """P(correct | signals).  Fitted on val, never on test."""

    def __init__(self, kind: str = "logreg") -> None:
        if kind not in MODELS:
            raise ValueError(f"kind must be one of {MODELS}")
        self.kind = kind
        self.name = f"fusion-{kind}"
        self.clf = None
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FusionModel":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import StandardScaler

        if self.kind == "naive_bayes":
            self.scaler = StandardScaler().fit(X)
            self.clf = GaussianNB().fit(self.scaler.transform(X), y)
        elif self.kind == "logreg":
            self.scaler = StandardScaler().fit(X)
            self.clf = LogisticRegression(
                max_iter=2000, C=1.0, random_state=CFG.seed, class_weight="balanced",
            ).fit(self.scaler.transform(X), y)
        else:
            self.clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.08, max_depth=5, random_state=CFG.seed,
                early_stopping=True, validation_fraction=0.15,
            ).fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("fit() first")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.clf.predict_proba(X)[:, 1]

    def score_rows(self, rows: Sequence[dict]) -> np.ndarray:
        return self.predict_proba(to_matrix(rows))

    def coefficients(self) -> list[dict]:
        """Readable weights — only the linear arm has them."""
        if self.kind != "logreg" or self.clf is None:
            return []
        out = [{"feature": n, "weight": float(w)}
               for n, w in zip(FUSION_FEATURE_NAMES, self.clf.coef_.ravel())]
        out.sort(key=lambda r: -abs(r["weight"]))
        for i, r in enumerate(out, start=1):
            r["rank"] = i
        return out

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"kind": self.kind, "clf": self.clf, "scaler": self.scaler,
                         "features": FUSION_FEATURE_NAMES}, fh)
        return path

    @classmethod
    def load(cls, path: Path) -> "FusionModel":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if tuple(blob["features"]) != FUSION_FEATURE_NAMES:
            raise RuntimeError("pickled fusion model has a different feature set; retrain")
        out = cls(blob["kind"])
        out.clf, out.scaler = blob["clf"], blob["scaler"]
        return out


def compare(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray,
            y_eval: np.ndarray) -> tuple[list[dict], dict[str, FusionModel]]:
    """Fit all three arms and score them — the RQ5 table."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    rows: list[dict] = []
    fitted: dict[str, FusionModel] = {}
    for kind in MODELS:
        model = FusionModel(kind).fit(X_fit, y_fit)
        p = model.predict_proba(X_eval)
        rows.append({
            "fusion": kind,
            "auroc": round(float(roc_auc_score(y_eval, p)), 4),
            "auprc": round(float(average_precision_score(y_eval, p)), 4),
            "brier": round(float(brier_score_loss(y_eval, p)), 4),
            "accuracy@0.5": round(float(np.mean((p >= 0.5) == y_eval.astype(bool))), 4),
            "mean_confidence": round(float(p.mean()), 4),
        })
        fitted[kind] = model
    rows.sort(key=lambda r: -r["auroc"])
    return rows, fitted
