"""The feature-based span reader (PLAN.md §7.6, task T10).

`LogisticRegression` and `HistGradientBoostingClassifier` over the features in
:mod:`bnqa.reader.features`, with a **per-question softmax** over candidates:
the model scores each span independently, the scores are normalised within the
question, and argmax wins.  Training takes under two minutes on CPU.

Why per-question softmax rather than a plain binary classifier
--------------------------------------------------------------
Reading is a *ranking* problem wearing a classification costume.  The absolute
probability that a given span is the answer is almost meaningless — it depends
on how many candidates the passage happened to yield — while the comparison
*between* spans of the same question is exactly what we need.  Normalising
within the question also makes the two quantities the verifier wants
(``margin`` = best minus second, and the absolute best score) commensurable
across questions, which is what lets S1 be a feature rather than a threshold.

Three decoding routes (§7.5)
----------------------------
``polarity`` questions never extract a span: the verifier's polarity check *is*
the yes/no decision (§10.7, "one mechanism, two jobs").  ``list`` questions take
the top-n non-overlapping spans and are scored as a set.  Everything else takes
the single best span.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import CFG
from ..preprocess.normalize import normalize
from .base import Answer, Candidate
from .candidates import IdfTable, generate
from .features import FEATURE_NAMES, extract, to_matrix
from .qtype import QuestionAnalysis, analyze


def _softmax(x: np.ndarray) -> np.ndarray:
    if not len(x):
        return x
    z = x - x.max()
    e = np.exp(z)
    return e / max(e.sum(), 1e-12)


class SpanRanker:
    """Score candidate spans; the best one is the answer."""

    def __init__(self, model: str = "logreg", *, name: str | None = None) -> None:
        if model not in ("logreg", "gbdt"):
            raise ValueError("model must be 'logreg' or 'gbdt'")
        self.model_kind = model
        self.name = name or f"reader-{model}"
        self.clf = None
        self.scaler = None
        self.n_train_questions = 0

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Sequence[int] | None = None
            ) -> "SpanRanker":
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if self.model_kind == "logreg":
            self.scaler = StandardScaler().fit(X)
            self.clf = LogisticRegression(
                max_iter=2000, C=1.0, random_state=CFG.seed, class_weight="balanced",
            ).fit(self.scaler.transform(X), y)
        else:
            self.clf = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.1, max_depth=6,
                random_state=CFG.seed, early_stopping=True, validation_fraction=0.1,
            ).fit(X, y)
        self.n_train_questions = len(set(groups)) if groups is not None else 0
        return self

    # ------------------------------------------------------------------ #
    # Scoring                                                            #
    # ------------------------------------------------------------------ #

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("fit() or load() first")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.clf.predict_proba(X)[:, 1]

    def rank(self, cands: Sequence[Candidate], X: np.ndarray) -> np.ndarray:
        """Per-question softmax over candidate scores."""
        if not len(cands):
            return np.zeros(0, dtype=np.float32)
        return _softmax(np.log(np.clip(self.raw_scores(X), 1e-9, 1.0)))

    # ------------------------------------------------------------------ #
    # Explainability                                                     #
    # ------------------------------------------------------------------ #

    def feature_importance(self) -> list[dict]:
        """The table PLAN.md §7.6 calls a deliverable."""
        if self.clf is None:
            return []
        if self.model_kind == "logreg":
            weights = self.clf.coef_.ravel()
            rows = [{"feature": n, "weight": float(w), "abs_weight": float(abs(w))}
                    for n, w in zip(FEATURE_NAMES, weights)]
        else:
            # HistGradientBoosting has no native importances; permutation
            # importance is computed by the evaluation script, which has the
            # held-out data this object does not.
            rows = [{"feature": n, "weight": float("nan"), "abs_weight": float("nan")}
                    for n in FEATURE_NAMES]
        rows.sort(key=lambda r: -(r["abs_weight"] if r["abs_weight"] == r["abs_weight"] else 0))
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
            r["model"] = self.name
        return rows

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"kind": self.model_kind, "name": self.name, "clf": self.clf,
                         "scaler": self.scaler, "features": FEATURE_NAMES,
                         "n_train_questions": self.n_train_questions}, fh)
        return path

    @classmethod
    def load(cls, path: Path) -> "SpanRanker":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if tuple(blob["features"]) != FEATURE_NAMES:
            raise RuntimeError(
                "the pickled reader was trained on a different feature set; retrain it")
        out = cls(blob["kind"], name=blob["name"])
        out.clf, out.scaler = blob["clf"], blob["scaler"]
        out.n_train_questions = blob.get("n_train_questions", 0)
        return out


# --------------------------------------------------------------------------- #
# The Reader facade                                                            #
# --------------------------------------------------------------------------- #


class FeatureReader:
    """``read(question, passages) -> Answer`` — implements ``reader.base.Reader``.

    ``ensemble`` holds S5's bagged rankers.  They are scored over the *same*
    candidate matrix rather than by re-reading, because candidate generation
    dominates the cost and the bags differ only in their weights — re-running
    it five times would multiply the verifier's wall-clock for identical
    candidates.
    """

    def __init__(self, ranker: SpanRanker, *, qtype_clf=None, top_n_list: int = 5,
                 ensemble: Sequence[SpanRanker] = ()) -> None:
        self.ranker = ranker
        self.qtype_clf = qtype_clf
        self.top_n_list = top_n_list
        self.ensemble = list(ensemble)
        self.name = ranker.name

    def analyse(self, question: str, question_type: str | None = None) -> QuestionAnalysis:
        if question_type is None and self.qtype_clf is not None:
            question_type = self.qtype_clf.predict([question])[0]
        return analyze(question, question_type)

    def read(self, question: str, passages: Sequence[dict], *,
             ranks: Sequence[tuple[int, float]] | None = None,
             question_type: str | None = None, top_k_report: int = 5) -> Answer:
        return self.read_with_ensemble(question, passages, ranks=ranks,
                                       question_type=question_type,
                                       top_k_report=top_k_report)[0]

    def read_with_ensemble(self, question: str, passages: Sequence[dict], *,
                           ranks: Sequence[tuple[int, float]] | None = None,
                           question_type: str | None = None,
                           top_k_report: int = 5) -> tuple[Answer, list[Answer]]:
        """``(answer, ensemble_answers)`` — S5 for the price of one read."""
        qa = self.analyse(question, question_type)
        cands = generate(question, passages, expected_class=qa.expected_class, ranks=ranks)
        if not cands:
            # Fall back to the unfiltered class so a strict type filter can never
            # produce an empty candidate set — abstention is the verifier's
            # decision to make, not a side effect of span generation.
            cands = generate(question, passages, expected_class="UNKNOWN", ranks=ranks)
        if not cands:
            return Answer(text="", pid=passages[0]["pid"] if passages else "",
                          char_start=0, char_end=0, sentence="", sentence_start=0,
                          sentence_end=0, route=qa.route,
                          expected_class=qa.expected_class,
                          question_type=qa.question_type), []

        idf = IdfTable([p["text"] for p in passages])
        feats = [extract(question, c, idf=idf, expected_class=qa.expected_class,
                         question_type=qa.question_type) for c in cands]
        X = to_matrix(feats)
        scores = self.ranker.rank(cands, X)
        for c, f, s in zip(cands, feats, scores):
            c.features, c.score = f, float(s)

        order = np.argsort(-scores)
        best = cands[int(order[0])]
        second = float(scores[order[1]]) if len(order) > 1 else 0.0

        answer = Answer(
            text=best.text, pid=best.pid, char_start=best.char_start, char_end=best.char_end,
            sentence=best.sentence, sentence_start=best.sentence_start,
            sentence_end=best.sentence_end, score=float(scores[order[0]]),
            margin=float(scores[order[0]]) - second,
            null_margin=float(scores[order[0]]) - float(np.median(scores)),
            route=qa.route, expected_class=qa.expected_class,
            question_type=qa.question_type, features=best.features,
            passage_rank=best.passage_rank, passage_score=best.passage_score,
            top_candidates=[{"text": cands[int(i)].text, "pid": cands[int(i)].pid,
                             "score": float(scores[int(i)]),
                             "start": cands[int(i)].char_start,
                             "end": cands[int(i)].char_end}
                            for i in order[:top_k_report]],
        )

        if qa.route == "list":
            answer.spans = _non_overlapping(cands, order, self.top_n_list)

        # S5: each bagged ranker picks from the same candidates.  Only the
        # chosen span matters to the signal, so nothing else is rebuilt.
        bag_answers: list[Answer] = []
        for bag in self.ensemble:
            bag_scores = bag.rank(cands, X)
            if not len(bag_scores):
                continue
            pick = cands[int(np.argmax(bag_scores))]
            bag_answers.append(Answer(
                text=pick.text, pid=pick.pid, char_start=pick.char_start,
                char_end=pick.char_end, sentence=pick.sentence,
                sentence_start=pick.sentence_start, sentence_end=pick.sentence_end,
                score=float(np.max(bag_scores)), route=qa.route,
                expected_class=qa.expected_class, question_type=qa.question_type))
        return answer, bag_answers


def _non_overlapping(cands: Sequence[Candidate], order: np.ndarray,
                     n: int) -> list[tuple[str, int, int]]:
    """Top-n spans that do not overlap — the set a LIST answer is scored as."""
    chosen: list[tuple[str, int, int]] = []
    taken: list[tuple[str, int, int]] = []
    for i in order:
        c = cands[int(i)]
        if any(c.pid == pid and not (c.char_end <= s or c.char_start >= e)
               for pid, s, e in taken):
            continue
        chosen.append((c.text, c.char_start, c.char_end))
        taken.append((c.pid, c.char_start, c.char_end))
        if len(chosen) >= n:
            break
    return chosen


def normalise_passages(passages: Sequence[dict]) -> list[dict]:
    """Passage texts exactly as the index stores them — offsets depend on it."""
    return [{**p, "text": normalize(p["text"])} for p in passages]
