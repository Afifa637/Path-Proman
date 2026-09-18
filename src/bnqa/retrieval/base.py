"""The one interface that makes the whole comparison valid (PLAN.md §6, §13).

Every retriever — Tier A's TF-IDF and BM25, Tier B's bi-encoder, late-interaction
and cross-encoder arms — implements this single protocol, and the CLI, the app
and every evaluation notebook consume **only** this protocol.  Two consequences:

* no arm can get an accidental advantage from better plumbing, so RQ1 measures
  retrievers rather than implementations;
* a Tier-B retriever is a **swap**, not a rewrite, which is what makes Tier A
  Tier B's ablation baseline (PLAN.md §0.1).

Analysers live here too, as module-level functions rather than lambdas, so a
fitted vectoriser can be pickled.
"""

from __future__ import annotations

import time
from typing import Iterable, Protocol, Sequence, runtime_checkable

from ..preprocess.stem import stem_tokens
from ..preprocess.stopwords import remove_stopwords
from ..preprocess.tokenize import char_ngrams, tokenize_lower

Hit = tuple[str, float]


# --------------------------------------------------------------------------- #
# Analysers                                                                    #
# --------------------------------------------------------------------------- #


def analyze_word(text: str) -> list[str]:
    """Word analyser for the **sparse** arms only.

    Stopword removal and suffix stripping are applied here and **nowhere else**.
    PLAN.md §7.2: dense models keep the full surface form, and the asymmetry is
    deliberate — stemming helps lexical matching and hurts distributional
    representations, so applying it to both would make RQ1 a measurement of
    preprocessing rather than of retrieval.
    """
    return stem_tokens(remove_stopwords(tokenize_lower(text)))


def analyze_char(text: str) -> list[str]:
    """Char 3–5 grams: absorbs Bangla inflection without a morphological analyser."""
    return char_ngrams(text, 3, 5)


def analyze_word_nostem(text: str) -> list[str]:
    """Surface-form words — the form dense arms and the reader use."""
    return tokenize_lower(text)


# --------------------------------------------------------------------------- #
# Protocol                                                                     #
# --------------------------------------------------------------------------- #


@runtime_checkable
class Retriever(Protocol):
    name: str

    def build(self, passages: Sequence[dict]) -> "Retriever":
        """Index ``passages`` (dicts with at least ``pid`` and ``text``)."""

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Return the top ``k`` ``(pid, score)`` pairs, best first."""


class BaseRetriever:
    """Shared bookkeeping: pid table, timing, batch search."""

    name = "base"

    def __init__(self) -> None:
        self.pids: list[str] = []
        self.build_seconds: float = 0.0
        self.index_bytes: int = 0

    # -- helpers ------------------------------------------------------------
    def _start_build(self, passages: Sequence[dict]) -> float:
        self.pids = [p["pid"] for p in passages]
        return time.perf_counter()

    def _end_build(self, t0: float) -> None:
        self.build_seconds = time.perf_counter() - t0

    def search_batch(self, queries: Iterable[str], k: int = 10) -> list[list[Hit]]:
        """Default batch implementation; arms with a faster path override it."""
        return [self.search(q, k) for q in queries]

    def timed_search(self, queries: Sequence[str], k: int = 10) -> tuple[list[list[Hit]], float]:
        """Return ``(results, median_ms_per_query)``.

        Latency is reported beside quality because a retriever that is 40x slower
        for one point of Recall@5 is a different engineering proposition, and the
        report should say so (PLAN.md §7.3).
        """
        t0 = time.perf_counter()
        out = self.search_batch(queries, k)
        elapsed = time.perf_counter() - t0
        return out, 1000.0 * elapsed / max(len(queries), 1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} docs={len(self.pids)}>"
