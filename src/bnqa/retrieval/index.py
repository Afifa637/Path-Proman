"""Index loading, the stemmer's corpus vocabulary, and caching (task T5).

``prepare(size)`` is the single entry point every retrieval script and the
pipeline use.  It does two things, and the second one matters more than it
looks:

1. loads the passages of one nested index (``index_{10,50}k.manifest.json``);
2. **installs the corpus vocabulary into the stemmer.**  PLAN.md §7.2's
   stemmer accepts the longest suffix strip whose result is an *attested* token
   of our own corpus, which is what stops ``উদ্ভিদের`` becoming ``উদ্ভি``.  That
   evidence is the index itself, so the vocabulary has to be installed before
   any retriever tokenises anything — otherwise the fallback tier order is used
   and the arms are silently built on a different analyser than the queries.

Latency and index size are measured here rather than in the metrics module,
because they are properties of the index and not of a query set
(PLAN.md §7.3).
"""

from __future__ import annotations

import time
from typing import Sequence

from ..config import CFG
from ..data.index_build import load_index, load_manifest
from ..preprocess.stem import set_vocabulary, vocabulary_size
from ..preprocess.tokenize import tokenize_lower
from .base import BaseRetriever

_cache: dict[int, list[dict]] = {}
_vocab_for: int | None = None


# --------------------------------------------------------------------------- #
# Corpus vocabulary                                                            #
# --------------------------------------------------------------------------- #


def corpus_vocabulary(passages: Sequence[dict], *, min_count: int = 2) -> set[str]:
    """Attested standalone tokens, used to validate candidate stems.

    ``min_count`` keeps a typo from licensing a bad split: a form seen once may
    itself be a broken token, and accepting it as evidence would reintroduce the
    exact over-stripping the corpus validation exists to prevent.
    """
    counts: dict[str, int] = {}
    for p in passages:
        for tok in tokenize_lower(p["text"]):
            counts[tok] = counts.get(tok, 0) + 1
    return {tok for tok, n in counts.items() if n >= min_count}


def install_vocabulary(passages: Sequence[dict]) -> int:
    set_vocabulary(corpus_vocabulary(passages))
    return vocabulary_size()


# --------------------------------------------------------------------------- #
# Index loading                                                                #
# --------------------------------------------------------------------------- #


def prepare(size: int | None = None, *, verbose: bool = True) -> list[dict]:
    """Passages of one index, with the stemmer vocabulary installed."""
    global _vocab_for
    size = CFG.headline_index if size is None else int(size)

    passages = _cache.get(size)
    if passages is None:
        t0 = time.perf_counter()
        passages = load_index(size)
        _cache[size] = passages
        if verbose:
            print(f"  loaded {len(passages):,d} passages "
                  f"({time.perf_counter() - t0:.1f}s)")

    if _vocab_for != size:
        n = install_vocabulary(passages)
        _vocab_for = size
        if verbose:
            print(f"  stemmer vocabulary: {n:,d} attested tokens from this index")
    return passages


def manifest(size: int) -> dict:
    return load_manifest(size)


def gold_pids(size: int) -> set[str]:
    """The BanglaRQA gold contexts present in an index — all of them, at every size."""
    return {p["pid"] for p in prepare(size, verbose=False) if p.get("source") == "gold"}


def clear_cache() -> None:
    global _vocab_for
    _cache.clear()
    _vocab_for = None
    set_vocabulary(None)


# --------------------------------------------------------------------------- #
# Latency                                                                      #
# --------------------------------------------------------------------------- #


def measure_latency(retriever: BaseRetriever, queries: Sequence[str], *,
                    k: int | None = None, warmup: int = 3) -> float:
    """Milliseconds per query, after a warm-up that pays the lazy-init costs."""
    k = CFG.top_k if k is None else k
    for q in queries[:warmup]:
        retriever.search(q, k)
    if not queries:
        return float("nan")
    t0 = time.perf_counter()
    for q in queries:
        retriever.search(q, k)
    return 1000.0 * (time.perf_counter() - t0) / len(queries)
