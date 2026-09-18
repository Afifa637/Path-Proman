"""TF-IDF retrieval, word and character n-gram (PLAN.md §7.3, task T5).

Two arms, and **reporting both is itself an RQ1 result**:

* **word 1–2 grams** over the stemmed, stopword-filtered analyser;
* **char 3–5 grams**, which absorb Bangla inflection without a morphological
  analyser — ``উদ্ভিদ`` and ``উদ্ভিদের`` share every n-gram of the stem and differ
  only at the tail, so the char arm matches them even where the stemmer fails.

Scoring is cosine similarity.  ``TfidfVectorizer`` L2-normalises rows, so the
cosine is a plain sparse dot product against the query vector.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import CFG
from .base import BaseRetriever, Hit, analyze_char, analyze_word


class TfidfRetriever(BaseRetriever):
    def __init__(self, kind: str = "word", *, min_df: int | None = None,
                 max_features: int | None = None, sublinear_tf: bool = True) -> None:
        super().__init__()
        if kind not in ("word", "char"):
            raise ValueError("kind must be 'word' or 'char'")
        self.kind = kind
        self.name = f"tfidf-{kind}"
        self.analyzer = analyze_word if kind == "word" else analyze_char
        self.min_df = (CFG.tfidf_min_df if kind == "word" else 5) if min_df is None else min_df
        self.max_features = max_features if max_features is not None else (
            CFG.tfidf_max_features_word if kind == "word" else CFG.tfidf_max_features_char)
        self.sublinear_tf = sublinear_tf
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def build(self, passages) -> "TfidfRetriever":
        t0 = self._start_build(passages)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32,  # halves the index; cosine does not need float64
        )
        self.matrix = self.vectorizer.fit_transform(p["text"] for p in passages)
        self.matrix = self.matrix.tocsr()
        self.index_bytes = int(self.matrix.data.nbytes + self.matrix.indices.nbytes
                               + self.matrix.indptr.nbytes)
        self._end_build(t0)
        return self

    def score(self, query: str) -> np.ndarray:
        assert self.vectorizer is not None and self.matrix is not None, "call build() first"
        q = self.vectorizer.transform([query])
        if q.nnz == 0:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray((self.matrix @ q.T).todense()).ravel()

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self.score(query)
        if not scores.any():
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.pids[i], float(scores[i])) for i in top if scores[i] > 0]

    def search_batch(self, queries, k: int = 10) -> list[list[Hit]]:
        """Vectorised: one sparse product for the whole batch."""
        assert self.vectorizer is not None and self.matrix is not None
        queries = list(queries)
        Q = self.vectorizer.transform(queries)
        S = (self.matrix @ Q.T).tocsc()
        out: list[list[Hit]] = []
        for j in range(len(queries)):
            start, end = S.indptr[j], S.indptr[j + 1]
            rows, vals = S.indices[start:end], S.data[start:end]
            if len(rows) == 0:
                out.append([])
                continue
            kk = min(k, len(rows))
            sel = np.argpartition(-vals, kk - 1)[:kk]
            sel = sel[np.argsort(-vals[sel])]
            out.append([(self.pids[rows[i]], float(vals[i])) for i in sel if vals[i] > 0])
        return out
