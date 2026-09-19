"""Reciprocal Rank Fusion and the query-expansion arm (task T7).

**RRF** (Cormack et al., 2009) fuses rankings, not scores:

.. math::  \\mathrm{RRF}(d) = \\sum_{r \\in R} \\frac{1}{k + \\mathrm{rank}_r(d)}

with ``k = 60``.  Fusing *ranks* is the right choice here because the arms it
combines are not commensurable — a BM25 score and a cosine similarity live on
different scales, and normalising them would introduce a free parameter that
would quietly become a tuned hyperparameter on top of the two we already tune.

**Query expansion** (PLAN.md §7.3) expands each content word of the question
with up to ``CFG.query_expansion_n`` neighbours from the T6b resources.  It
keeps count of how often it actually fires, because the honest result here is
that a hand-authored curriculum lexicon barely touches Wikipedia-domain
BanglaRQA questions — and that measured coverage is the baseline Tier B's
induced synonyms (X3) have to beat.  A number beats a paragraph.
"""

from __future__ import annotations

from typing import Sequence

from ..config import CFG
from ..preprocess.stopwords import content_tokens
from ..preprocess.tokenize import tokenize_lower
from ..resources.loader import expand_term
from .base import BaseRetriever, Hit


class RRFHybrid(BaseRetriever):
    """Rank fusion over any set of already-built retrievers."""

    def __init__(self, retrievers: Sequence[BaseRetriever], *, k: int | None = None,
                 name: str | None = None, depth: int = 100) -> None:
        super().__init__()
        self.retrievers = list(retrievers)
        self.k = CFG.rrf_k if k is None else int(k)
        self.depth = depth
        self.name = name or "rrf(" + "+".join(r.name.replace("tfidf-", "") for r in self.retrievers) + ")"

    def build(self, passages: Sequence[dict]) -> "RRFHybrid":
        # The arms are built by the caller and shared, so fusing costs nothing
        # beyond the rank arithmetic.  Only the pid list is needed here.
        self.pids = [p["pid"] for p in passages]
        return self

    def search(self, query: str, k: int = 10) -> list[Hit]:
        fused: dict[str, float] = {}
        for r in self.retrievers:
            for rank, (pid, _score) in enumerate(r.search(query, self.depth), start=1):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.k + rank)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(pid, float(score)) for pid, score in ranked[:k]]

    def _state_for_size(self) -> object:
        # An RRF index is the union of its arms; reporting their sum is the
        # honest figure for what a deployment would have to ship.
        return {"arms": [r.size_mb() for r in self.retrievers]}

    def size_mb(self) -> float:
        return float(sum(r.size_mb() for r in self.retrievers))


class ExpandedQuery(BaseRetriever):
    """Wraps a retriever and expands the query with T6b lexical resources."""

    def __init__(self, inner: BaseRetriever, *, n: int | None = None,
                 name: str | None = None) -> None:
        super().__init__()
        self.inner = inner
        self.n = CFG.query_expansion_n if n is None else int(n)
        self.name = name or f"qe({inner.name})"
        self.expanded_queries = 0
        self.total_queries = 0
        self.added_terms = 0

    def build(self, passages: Sequence[dict]) -> "ExpandedQuery":
        self.pids = [p["pid"] for p in passages]
        return self

    def expand(self, query: str) -> tuple[str, int]:
        """Return ``(expanded_query, n_added)``."""
        added: list[str] = []
        seen = set(tokenize_lower(query))
        for tok in content_tokens(tokenize_lower(query)):
            extras = sorted(expand_term(tok) - {tok})
            for extra in extras[: self.n]:
                key = extra.lower()
                if key not in seen:
                    seen.add(key)
                    added.append(extra)
        return (query + " " + " ".join(added)) if added else query, len(added)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        expanded, n_added = self.expand(query)
        self.total_queries += 1
        if n_added:
            self.expanded_queries += 1
            self.added_terms += n_added
        return self.inner.search(expanded, k)

    def reset_counters(self) -> None:
        self.expanded_queries = self.total_queries = self.added_terms = 0

    def coverage(self) -> dict:
        """The number that explains this arm's row in ``retrieval.csv``."""
        return {
            "queries": self.total_queries,
            "expanded_queries": self.expanded_queries,
            "expanded_pct": round(100 * self.expanded_queries / max(self.total_queries, 1), 2),
            "added_terms_per_expanded_query": round(
                self.added_terms / max(self.expanded_queries, 1), 2),
        }

    def size_mb(self) -> float:
        return self.inner.size_mb()
