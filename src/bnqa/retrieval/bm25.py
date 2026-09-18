"""Okapi BM25, written by us (PLAN.md §7.3, task T5).

Deliberately **not** ``rank_bm25``: implementing it ourselves satisfies the
from-scratch requirement and removes a dependency.  Validated in
``tests/test_bm25.py`` against a hand-worked three-document example, because a
subtly wrong BM25 still returns plausible-looking rankings and would quietly
become the baseline every other arm is compared against.

Implementation note — the whole score matrix is precomputed.  BM25's document
term weight

    w(d, t) = idf(t) · tf(d,t)·(k1 + 1) / (tf(d,t) + k1·(1 − b + b·|d|/avgdl))

depends only on the document, so it can be folded into the sparse count matrix
once at build time.  Scoring a query is then a **sum of the columns** for its
terms, which is a single sparse matrix-vector product rather than a Python loop
over postings.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..config import CFG
from .base import BaseRetriever, Hit, analyze_word


class BM25Retriever(BaseRetriever):
    """Okapi BM25 with the standard Robertson/Sparck-Jones idf."""

    def __init__(self, k1: float | None = None, b: float | None = None,
                 analyzer=analyze_word, name: str = "bm25",
                 retain_counts: bool = False) -> None:
        super().__init__()
        self.k1 = CFG.bm25_k1 if k1 is None else k1
        self.b = CFG.bm25_b if b is None else b
        self.analyzer = analyzer
        self.name = name
        #: keep the raw term counts so ``set_params`` can re-weight without
        #: re-tokenising the corpus — a k1/b grid search is otherwise 16 full
        #: rebuilds of a matrix that does not depend on k1 or b at all.
        self.retain_counts = retain_counts
        self.vocab: dict[str, int] = {}
        self.weights: sparse.csc_matrix | None = None
        self.idf: np.ndarray | None = None
        self.doc_len: np.ndarray | None = None
        self._counts: sparse.coo_matrix | None = None

    # ------------------------------------------------------------------ #
    def build(self, passages) -> "BM25Retriever":
        t0 = self._start_build(passages)

        indptr = [0]
        indices: list[int] = []
        data: list[int] = []
        doc_len = np.zeros(len(passages), dtype=np.float64)

        for i, p in enumerate(passages):
            counts: dict[int, int] = {}
            toks = self.analyzer(p["text"])
            doc_len[i] = len(toks)
            for tok in toks:
                idx = self.vocab.get(tok)
                if idx is None:
                    idx = len(self.vocab)
                    self.vocab[tok] = idx
                counts[idx] = counts.get(idx, 0) + 1
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr.append(len(indices))

        n_docs, n_terms = len(passages), len(self.vocab)
        tf = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float64), np.asarray(indices), np.asarray(indptr)),
            shape=(n_docs, n_terms))

        # document frequency -> idf
        df = np.diff(tf.tocsc().indptr).astype(np.float64)
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        self.doc_len = doc_len
        self._counts = tf.tocoo()
        self._reweight()
        if not self.retain_counts:
            self._counts = None

        self.index_bytes = int(self.weights.data.nbytes + self.weights.indices.nbytes
                               + self.weights.indptr.nbytes)
        self._end_build(t0)
        return self

    # ------------------------------------------------------------------ #
    def _reweight(self) -> None:
        """Fold the current k1/b length normalisation into the count matrix."""
        assert self._counts is not None and self.doc_len is not None and self.idf is not None
        tf = self._counts
        n_docs, n_terms = tf.shape
        avgdl = float(self.doc_len.mean()) if n_docs else 0.0
        norm = self.k1 * (1.0 - self.b + self.b * (self.doc_len / (avgdl or 1.0)))
        numer = tf.data * (self.k1 + 1.0)
        denom = tf.data + norm[tf.row]
        weighted = (numer / denom) * self.idf[tf.col]
        self.weights = sparse.csc_matrix(
            (weighted, (tf.row, tf.col)), shape=(n_docs, n_terms))

    def set_params(self, k1: float, b: float) -> "BM25Retriever":
        """Re-weight in place for a new (k1, b).  Requires ``retain_counts=True``."""
        if self._counts is None:
            raise RuntimeError("build with retain_counts=True to retune without rebuilding")
        self.k1, self.b = k1, b
        self._reweight()
        return self

    # ------------------------------------------------------------------ #
    def score(self, query: str) -> np.ndarray:
        assert self.weights is not None, "call build() first"
        cols = [self.vocab[t] for t in self.analyzer(query) if t in self.vocab]
        scores = np.zeros(self.weights.shape[0], dtype=np.float64)
        if not cols:
            return scores
        # summing the query's columns == a sparse mat-vec with a 0/1 query vector,
        # but done directly so repeated query terms count once per occurrence
        for c in cols:
            start, end = self.weights.indptr[c], self.weights.indptr[c + 1]
            np.add.at(scores, self.weights.indices[start:end], self.weights.data[start:end])
        return scores

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self.score(query)
        if not scores.any():
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.pids[i], float(scores[i])) for i in top if scores[i] > 0]
