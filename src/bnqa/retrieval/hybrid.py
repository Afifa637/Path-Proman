"""Reciprocal Rank Fusion and query expansion (PLAN.md §7.3, task T7).

**RRF, not weighted score interpolation** (v1's trim, preserved).  BM25 scores
are unbounded sums of idf terms while TF-IDF cosines live in [0, 1]; combining
them by value needs a normalisation that is itself a tuned hyperparameter and a
place for the comparison to go wrong.  RRF only reads **ranks**, so it is
invariant to every monotone rescaling of either arm:

    score(d) = Σ_r  1 / (k + rank_r(d)),     k = 60

**Query expansion** adds the T6b synonym class and surface variants of each
content word to the query.  This is the cheap, direct probe of RQ1's real
question — how much of a lexical retriever's failure is vocabulary mismatch
rather than a missing notion of semantics — and it reuses resources we already
built instead of adding machinery.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..config import CFG
from ..preprocess.stopwords import content_tokens
from ..preprocess.tokenize import tokenize_lower
from ..resources.loader import expand_term
from .base import BaseRetriever, Hit


class RRFHybrid(BaseRetriever):
    """Fuse any number of retrievers by reciprocal rank."""

    def __init__(self, retrievers: Sequence[BaseRetriever], *, k: int | None = None,
                 depth: int = 100, name: str | None = None) -> None:
        super().__init__()
        self.retrievers = list(retrievers)
        self.k = CFG.rrf_k if k is None else k
        self.depth = depth
        self.name = name or "rrf(" + "+".join(r.name for r in self.retrievers) + ")"

    def build(self, passages) -> "RRFHybrid":
        t0 = self._start_build(passages)
        for r in self.retrievers:
            if not r.pids:
                r.build(passages)
        self.index_bytes = sum(getattr(r, "index_bytes", 0) for r in self.retrievers)
        self._end_build(t0)
        return self

    def _fuse(self, rankings: Sequence[Sequence[Hit]], k: int) -> list[Hit]:
        scores: defaultdict[str, float] = defaultdict(float)
        for hits in rankings:
            for rank, (pid, _) in enumerate(hits, start=1):
                scores[pid] += 1.0 / (self.k + rank)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self._fuse([r.search(query, self.depth) for r in self.retrievers], k)

    def search_batch(self, queries, k: int = 10) -> list[list[Hit]]:
        queries = list(queries)
        per_arm = [r.search_batch(queries, self.depth) for r in self.retrievers]
        return [self._fuse([arm[i] for arm in per_arm], k) for i in range(len(queries))]


class ExpandedQuery(BaseRetriever):
    """Wrap a retriever so every query is lexically expanded before search."""

    def __init__(self, retriever: BaseRetriever, *, per_word: int | None = None,
                 include_unreviewed: bool = False) -> None:
        super().__init__()
        self.inner = retriever
        self.per_word = CFG.query_expansion_n if per_word is None else per_word
        self.include_unreviewed = include_unreviewed
        self.name = f"{retriever.name}+qe"
        self.expanded_queries = 0
        self.added_terms = 0

    def build(self, passages) -> "ExpandedQuery":
        t0 = self._start_build(passages)
        if not self.inner.pids:
            self.inner.build(passages)
        self.index_bytes = getattr(self.inner, "index_bytes", 0)
        self._end_build(t0)
        return self

    def expand(self, query: str) -> str:
        extra: list[str] = []
        for word in content_tokens(tokenize_lower(query)):
            alts = [t for t in expand_term(word, include_unreviewed=self.include_unreviewed)
                    if t != word]
            extra.extend(alts[: self.per_word])
        if extra:
            self.expanded_queries += 1
            self.added_terms += len(extra)
        return query + " " + " ".join(extra) if extra else query

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return self.inner.search(self.expand(query), k)

    def search_batch(self, queries, k: int = 10) -> list[list[Hit]]:
        return self.inner.search_batch([self.expand(q) for q in queries], k)
