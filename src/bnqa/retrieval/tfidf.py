"""TF-IDF retrieval, word 1-2 grams and char 3-5 grams (task T5).

Two vectorisers, and **reporting both is itself an RQ1 result** (PLAN.md §7.3):

* the **word** arm sees stemmed, stopword-free tokens and is the conventional
  lexical baseline;
* the **char** arm sees character n-grams and therefore absorbs Bangla
  inflection *without* a morphological analyser — ``উদ্ভিদ`` and ``উদ্ভিদের``
  share every n-gram of the stem and differ only at the tail.

The interesting comparison is not "which wins" but **how much of the word arm's
gap the char arm closes for free**, because that is the budget Tier B's learned
subword tokenizer (X2) has to beat.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from ..config import CFG
from ..preprocess.tokenize import char_ngrams
from .base import BaseRetriever, Hit, sparse_analyzer
from .bm25 import _topk


def _char_analyzer(text: str) -> list[str]:
    lo, hi = CFG.tfidf_char_ngrams
    return char_ngrams(text, lo, hi)


class TfidfRetriever(BaseRetriever):
    """Cosine similarity over an L2-normalised TF-IDF matrix."""

    def __init__(self, kind: str = "word", *, name: str | None = None) -> None:
        super().__init__()
        if kind not in ("word", "char"):
            raise ValueError(f"kind must be 'word' or 'char', got {kind!r}")
        self.kind = kind
        self.name = name or f"tfidf-{kind}"
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    def build(self, passages: Sequence[dict]) -> "TfidfRetriever":
        texts = self._ingest(passages)
        # A callable analyser bypasses scikit-learn's own n-gram machinery, so
        # both ``ngram_range`` settings are applied inside the callables below.
        analyzer = _word_ngram_analyzer if self.kind == "word" else _char_analyzer
        max_features = (CFG.tfidf_max_features_word if self.kind == "word"
                        else CFG.tfidf_max_features_char)
        vec = TfidfVectorizer(
            analyzer=analyzer,
            min_df=CFG.tfidf_min_df,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.matrix = l2_normalize(vec.fit_transform(texts)).tocsr()
        self.vectorizer = vec
        return self

    def score(self, query: str) -> np.ndarray:
        if self.vectorizer is None or self.matrix is None:  # pragma: no cover
            raise RuntimeError("build() first")
        q = l2_normalize(self.vectorizer.transform([query]))
        return np.asarray((self.matrix @ q.T).todense()).ravel()

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return _topk(self.score(query), self.pids, k)

    def _state_for_size(self) -> object:
        return {"vectorizer": self.vectorizer, "matrix": self.matrix, "pids": self.pids}


def _word_ngram_analyzer(text: str) -> list[str]:
    """Stemmed unigrams plus bigrams, per ``CFG.tfidf_word_ngrams``."""
    toks = sparse_analyzer(text)
    lo, hi = CFG.tfidf_word_ngrams
    out: list[str] = []
    for n in range(lo, hi + 1):
        if n == 1:
            out.extend(toks)
        else:
            out.extend(" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return out
