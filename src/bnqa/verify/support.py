"""S3, the support classifier (PLAN.md §10.4, task T11b).

Tier A's S3 is a ``HistGradientBoostingClassifier`` over lexical + veto +
overlap features on BanglaVerify.  It trains in seconds and every feature is
nameable, which is the Tier-A bargain: an examiner can audit it, and Tier B's
cross-encoder (X10) has a printed baseline to beat rather than an assumed one.

*(File note: PLAN.md §13 lists the verify package without a separate
``support.py`` — S3's classifier is split out from ``signals.py`` because
``signals.py`` must stay import-light for the UI, which draws the five bars
without ever training anything.)*

Negatives come from **four** sources, and the fourth is the one that matters:

1. unanswerable items,
2. wrong-passage retrievals,
3. BM25 hard distractors (BanglaVerify's NEUTRAL),
4. **minimally-contrasting swaps** (BanglaVerify's CONTRADICTED).

Without (4) the model learns topicality and nothing else.  The three-way head
is kept rather than collapsed to binary because CONTRADICTED and NEUTRAL are
different failures — one means "the book says otherwise", the other means "the
book does not say" — and the UI has to tell a student which it is.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import CFG
from ..preprocess.stopwords import content_tokens
from ..preprocess.tokenize import tokenize_lower
from ..reader.candidates import trigram_cosine
from .banglaverify import CONTRADICTED, LABELS, NEUTRAL, SUPPORTED
from .constraints import best_evidence_sentence, veto_features

FEATURE_NAMES: tuple[str, ...] = (
    "veto_numeral", "veto_date", "veto_unit", "veto_entity", "veto_polarity",
    "veto_relation_order", "veto_any", "veto_count", "polarity_match",
    "year_in_evidence", "answer_has_year", "nums_in_evidence", "answer_has_number",
    "token_overlap", "token_jaccard", "trigram_cos", "len_ratio",
    "statement_tokens", "evidence_tokens", "content_overlap", "novel_token_frac",
)


def features(statement: str, evidence: str, *, gazetteer=None) -> dict[str, float]:
    # The veto features are sentence-level; handing them a whole passage makes
    # the polarity rule misfire on almost every claim (see
    # ``constraints.best_evidence_sentence``).  The lexical-overlap features
    # below still see the full passage, because "is this claim anywhere in
    # this passage" is exactly what they are for.
    veto_evidence = best_evidence_sentence(statement, evidence)
    s_toks = tokenize_lower(statement)
    e_toks = tokenize_lower(evidence)
    s_set, e_set = set(s_toks), set(e_toks)
    inter = s_set & e_set
    union = s_set | e_set
    s_content = set(content_tokens(s_toks))
    e_content = set(content_tokens(e_toks))

    feats = veto_features(statement, veto_evidence, gazetteer=gazetteer)
    feats.update({
        "token_overlap": len(inter) / max(len(s_set), 1),
        "token_jaccard": len(inter) / max(len(union), 1),
        "trigram_cos": trigram_cosine(statement, evidence),
        "len_ratio": len(s_toks) / max(len(e_toks), 1),
        "statement_tokens": float(len(s_toks)),
        "evidence_tokens": float(len(e_toks)),
        "content_overlap": len(s_content & e_content) / max(len(s_content), 1),
        # The minimal-pair signal: in a one-token swap exactly one content token
        # is new, so this is small and non-zero — which is what separates a
        # contradiction from a neutral pair, where it is large.
        "novel_token_frac": len(s_content - e_content) / max(len(s_content), 1),
    })
    return feats


def to_matrix(rows: Sequence[dict[str, float]]) -> np.ndarray:
    return np.asarray([[r.get(n, 0.0) for n in FEATURE_NAMES] for r in rows],
                      dtype=np.float32)


class SupportClassifier:
    """Three-way SUPPORTED / CONTRADICTED / NEUTRAL over BanglaVerify."""

    def __init__(self, name: str = "s3-gbdt") -> None:
        self.name = name
        self.clf = None
        self.classes_: list[str] = []

    def fit(self, items: Sequence[dict]) -> "SupportClassifier":
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = to_matrix([features(i["statement"], i["evidence"]) for i in items])
        y = np.asarray([i["label"] for i in items])
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=6, random_state=CFG.seed,
            early_stopping=True, validation_fraction=0.1,
        ).fit(X, y)
        self.classes_ = list(self.clf.classes_)
        return self

    # -- prediction ---------------------------------------------------------

    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("fit() or load() first")
        X = to_matrix([features(s, e) for s, e in pairs])
        return self.clf.predict_proba(X)

    def predict_support(self, statement: str, evidence: str, *, question: str = "") -> float:
        """P(SUPPORTED) — the scalar S3 contributes to fusion."""
        if self.clf is None:
            return 0.5
        proba = self.predict_proba([(statement, evidence)])[0]
        idx = self.classes_.index(SUPPORTED) if SUPPORTED in self.classes_ else 0
        return float(proba[idx])

    def predict_label(self, statement: str, evidence: str) -> str:
        if self.clf is None:
            return NEUTRAL
        proba = self.predict_proba([(statement, evidence)])[0]
        return self.classes_[int(np.argmax(proba))]

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, items: Sequence[dict]) -> dict:
        """Accuracy overall, per label and **per generator**.

        The per-generator breakdown is the one that matters on the contrast
        set: an average hides a generator the model has not learned at all.
        """
        if not items:
            return {}
        proba = self.predict_proba([(i["statement"], i["evidence"]) for i in items])
        pred = [self.classes_[int(k)] for k in np.argmax(proba, axis=1)]
        gold = [i["label"] for i in items]

        out: dict = {"n": len(items),
                     "accuracy": float(np.mean([p == g for p, g in zip(pred, gold)]))}
        for label in LABELS:
            idx = [i for i, g in enumerate(gold) if g == label]
            if idx:
                out[f"acc_{label.lower()}"] = float(
                    np.mean([pred[i] == gold[i] for i in idx]))
                out[f"n_{label.lower()}"] = len(idx)

        by_gen: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            by_gen.setdefault(item.get("generator", "?"), []).append(i)
        out["by_generator"] = {
            g: {"n": len(idx),
                "accuracy": round(float(np.mean([pred[i] == gold[i] for i in idx])), 4)}
            for g, idx in sorted(by_gen.items())}
        # Chance for a 3-way task with these priors — the number §10.7 says the
        # contrast-set result must be compared against, not just reported.
        counts = {label: gold.count(label) for label in set(gold)}
        out["majority_baseline"] = round(max(counts.values()) / len(gold), 4)
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"clf": self.clf, "classes": self.classes_,
                         "features": FEATURE_NAMES, "name": self.name}, fh)
        return path

    @classmethod
    def load(cls, path: Path) -> "SupportClassifier":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if tuple(blob["features"]) != FEATURE_NAMES:
            raise RuntimeError("pickled S3 has a different feature set; retrain it")
        out = cls(blob["name"])
        out.clf, out.classes_ = blob["clf"], blob["classes"]
        return out


__all__ = ["SupportClassifier", "features", "FEATURE_NAMES",
           "SUPPORTED", "CONTRADICTED", "NEUTRAL"]
