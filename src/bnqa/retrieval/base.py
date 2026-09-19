"""The ``Retriever`` protocol (PLAN.md §13, task T5).

This file is one of the three interfaces the whole two-tier design rests on.
Every Tier-A arm (TF-IDF word, TF-IDF char, our BM25, the RRF hybrid, query
expansion) and every Tier-B arm (static-dense, bi-encoder, late interaction,
cross-encoder) implements exactly this, so:

* the pipeline, the Streamlit app and every evaluation script consume **only**
  ``build`` / ``search`` and never a concrete class;
* **Tier B is a swap, not a rewrite**, and Tier A is its ablation baseline.

``search`` returns ``(pid, score)`` pairs, highest first.  Scores are
retriever-specific and are never compared across arms — only ranks are.
"""

from __future__ import annotations

import pickle
import time
from typing import Protocol, Sequence, runtime_checkable

from ..preprocess.normalize import normalize
from ..preprocess.stem import stem_tokens
from ..preprocess.stopwords import remove_stopwords
from ..preprocess.tokenize import tokenize_lower

Hit = tuple[str, float]


@runtime_checkable
class Retriever(Protocol):
    """Build once over a passage list, then answer queries."""

    name: str

    def build(self, passages: Sequence[dict]) -> "Retriever": ...

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


# --------------------------------------------------------------------------- #
# The shared sparse analyser                                                   #
# --------------------------------------------------------------------------- #
#
# Stemming and stopword removal are applied **here and only here**, which is how
# PLAN.md §7.2's asymmetry ("sparse models only") is enforced rather than merely
# intended: a dense arm that never calls this analyser cannot accidentally
# inherit it.  ``tests/test_preprocess_asymmetry.py`` asserts the rule.


def sparse_terms(text: str, *, do_stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    """Word terms as the sparse retrievers see them."""
    toks = tokenize_lower(text)
    if drop_stopwords:
        toks = remove_stopwords(toks)
    if do_stem:
        toks = stem_tokens(toks)
    return [t for t in toks if t]


def sparse_analyzer(text: str) -> list[str]:
    """scikit-learn ``analyzer=`` callable for the word vectoriser."""
    return sparse_terms(text)


# --------------------------------------------------------------------------- #
# Shared bookkeeping                                                           #
# --------------------------------------------------------------------------- #


class BaseRetriever:
    """Common plumbing: pid bookkeeping, latency timing and index size.

    ``index.py`` logs latency and size **alongside** quality because a retriever
    that is 40x slower for one point of Recall@5 is a different engineering
    proposition and PLAN.md §7.3 says the report has to say so.
    """

    name: str = "base"

    def __init__(self) -> None:
        self.pids: list[str] = []
        self.texts: list[str] = []
        self._build_seconds: float = 0.0

    # -- construction -------------------------------------------------------
    def _ingest(self, passages: Sequence[dict]) -> list[str]:
        self.pids = [p["pid"] for p in passages]
        self.texts = [normalize(p["text"]) for p in passages]
        return self.texts

    def build(self, passages: Sequence[dict]) -> "BaseRetriever":  # pragma: no cover
        raise NotImplementedError

    # -- query --------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> list[Hit]:  # pragma: no cover
        raise NotImplementedError

    def search_batch(self, queries: Sequence[str], k: int = 10) -> list[list[Hit]]:
        """Default: one at a time.  Vectorised arms override this."""
        return [self.search(q, k) for q in queries]

    # -- reporting ----------------------------------------------------------
    @property
    def build_seconds(self) -> float:
        return self._build_seconds

    def size_mb(self) -> float:
        """Serialised size of the index structures, in MB.

        Measured by pickling the fitted state, which is what "how big is this
        index" means operationally: it is what a deployment would have to ship.
        """
        try:
            blob = pickle.dumps(self._state_for_size(), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # pragma: no cover - never let reporting break a run
            return float("nan")
        return len(blob) / (1024 * 1024)

    def _state_for_size(self) -> object:
        """What counts as "the index" for :meth:`size_mb`.  Text is excluded."""
        skip = {"texts", "_build_seconds"}
        return {k: v for k, v in self.__dict__.items() if k not in skip}

    def timed_build(self, passages: Sequence[dict]) -> "BaseRetriever":
        t0 = time.perf_counter()
        out = self.build(passages)
        self._build_seconds = time.perf_counter() - t0
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r} n={len(self.pids):,d}>"
