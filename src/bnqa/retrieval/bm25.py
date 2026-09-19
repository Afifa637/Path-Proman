"""Okapi BM25, written from scratch over a scipy sparse matrix (task T5).

PLAN.md §0.2 promises BM25 is **ours**, not a library call, and PLAN.md §7.3
requires it to be validated against a hand-worked example in ``pytest``
(``tests/test_bm25.py``).  The scoring function is the standard one:

.. math::

    \\mathrm{score}(q, d) = \\sum_{t \\in q} \\mathrm{idf}(t)
        \\frac{f_{t,d}\\,(k_1 + 1)}{f_{t,d} + k_1\\bigl(1 - b + b\\,|d| / \\mathrm{avgdl}\\bigr)}

with the standard Robertson/Sparck-Jones probabilistic idf

.. math::  \\mathrm{idf}(t) = \\ln\\Bigl(1 + \\frac{N - n_t + 0.5}{n_t + 0.5}\\Bigr).

**Why the weights are precomputed.**  ``k1`` and ``b`` change only how a term
frequency is normalised, never which documents contain which terms.  So the
count matrix is built once and re-weighted per grid point, which is what makes
the 16-point ``k1 x b`` tuning sweep in ``scripts/run_retrieval.py`` cheap
enough to run at both index sizes.  ``retain_counts=True`` keeps the raw counts
around for exactly that purpose.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..config import CFG
from .base import BaseRetriever, Hit, sparse_terms


class BM25Retriever(BaseRetriever):
    """Our BM25.  ``k1``/``b`` are tuned on **val** and never on test."""

    def __init__(self, k1: float | None = None, b: float | None = None, *,
                 retain_counts: bool = False, name: str = "bm25") -> None:
        super().__init__()
        self.name = name
        self.k1 = CFG.bm25_k1 if k1 is None else float(k1)
        self.b = CFG.bm25_b if b is None else float(b)
        self.retain_counts = retain_counts

        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)
        self.doc_len: np.ndarray = np.zeros(0)
        self.avgdl: float = 0.0
        self._counts: sp.csc_matrix | None = None   # docs x terms, raw tf
        self._weights: sp.csc_matrix | None = None  # docs x terms, BM25-weighted

    # ------------------------------------------------------------------ #
    # Build                                                              #
    # ------------------------------------------------------------------ #

    def build(self, passages: Sequence[dict]) -> "BM25Retriever":
        texts = self._ingest(passages)

        indptr = [0]
        indices: list[int] = []
        data: list[int] = []
        doc_len: list[int] = []
        vocab = self.vocab

        for text in texts:
            terms = sparse_terms(text)
            doc_len.append(len(terms))
            for term, count in Counter(terms).items():
                idx = vocab.get(term)
                if idx is None:
                    idx = len(vocab)
                    vocab[term] = idx
                indices.append(idx)
                data.append(count)
            indptr.append(len(indices))

        n_docs, n_terms = len(texts), len(vocab)
        counts = sp.csr_matrix(
            (np.asarray(data, dtype=np.float32), np.asarray(indices), np.asarray(indptr)),
            shape=(n_docs, n_terms),
        )

        self.doc_len = np.asarray(doc_len, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if n_docs else 0.0

        # document frequency -> probabilistic idf, floored at 0 so a term in
        # every document contributes nothing rather than a negative score.
        df = np.diff(counts.tocsc().indptr).astype(np.float32)
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        self._counts = counts.tocsc() if self.retain_counts else counts.tocsc()
        self._reweight()
        if not self.retain_counts:
            self._counts = None
        return self

    # ------------------------------------------------------------------ #
    # Weighting                                                          #
    # ------------------------------------------------------------------ #

    def _reweight(self) -> None:
        counts = self._counts
        if counts is None:  # pragma: no cover - guarded by set_params
            raise RuntimeError("raw counts were dropped; build with retain_counts=True")
        coo = counts.tocoo()
        dl = self.doc_len[coo.row]
        norm = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
        tf = coo.data
        weighted = self.idf[coo.col] * (tf * (self.k1 + 1.0)) / (tf + norm)
        self._weights = sp.csc_matrix(
            (weighted.astype(np.float32), (coo.row, coo.col)), shape=counts.shape)

    def set_params(self, k1: float, b: float) -> "BM25Retriever":
        """Re-weight in place for a tuning grid point.  Needs ``retain_counts``."""
        if self._counts is None:
            raise RuntimeError("set_params needs BM25Retriever(retain_counts=True)")
        self.k1, self.b = float(k1), float(b)
        self._reweight()
        return self

    # ------------------------------------------------------------------ #
    # Query                                                              #
    # ------------------------------------------------------------------ #

    def score(self, query: str) -> np.ndarray:
        """Dense score vector over all documents."""
        weights = self._weights
        if weights is None:  # pragma: no cover
            raise RuntimeError("build() first")
        scores = np.zeros(weights.shape[0], dtype=np.float32)
        for term, count in Counter(sparse_terms(query)).items():
            col = self.vocab.get(term)
            if col is None:
                continue
            start, end = weights.indptr[col], weights.indptr[col + 1]
            # A repeated query term counts once per occurrence, which is the
            # textbook BM25 sum over the multiset of query terms.
            scores[weights.indices[start:end]] += count * weights.data[start:end]
        return scores

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self.score(query)
        return _topk(scores, self.pids, k)

    def _state_for_size(self) -> object:
        return {"vocab": self.vocab, "idf": self.idf, "doc_len": self.doc_len,
                "weights": self._weights, "pids": self.pids}


def _topk(scores: np.ndarray, pids: list[str], k: int) -> list[Hit]:
    """Top-``k`` by score, ties broken by index order so runs are deterministic."""
    k = min(k, len(pids))
    if k <= 0:
        return []
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.lexsort((part, -scores[part]))]
    return [(pids[i], float(scores[i])) for i in order if scores[i] > 0]
