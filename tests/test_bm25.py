"""BM25 against a hand-worked example (PLAN.md T5 "Done when", §18.1).

A subtly wrong BM25 still returns plausible rankings, and it is the baseline
every other retrieval arm is measured against — so it is checked against numbers
computed by hand from the Okapi formula rather than against its own output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bnqa.retrieval.bm25 import BM25Retriever

K1, B = 1.2, 0.75

# Three documents over a five-word vocabulary; analyser is identity-on-split so
# the arithmetic below is fully determined.
DOCS = [
    {"pid": "d1", "text": "ka ka kha ga"},      # len 4
    {"pid": "d2", "text": "ka kha"},            # len 2
    {"pid": "d3", "text": "gha nga nga nga"},   # len 4
]
AVGDL = (4 + 2 + 4) / 3


def _analyzer(text: str) -> list[str]:
    return text.split()


def idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def term_weight(tf: int, doc_len: int, n_docs: int, df: int) -> float:
    if tf == 0:
        return 0.0
    denom = tf + K1 * (1.0 - B + B * doc_len / AVGDL)
    return idf(n_docs, df) * tf * (K1 + 1.0) / denom


@pytest.fixture(scope="module")
def retriever() -> BM25Retriever:
    return BM25Retriever(k1=K1, b=B, analyzer=_analyzer).build(DOCS)


def test_idf_matches_hand_computation(retriever: BM25Retriever) -> None:
    # "ka" appears in d1 and d2 -> df = 2 of 3 documents
    assert retriever.idf is not None
    got = retriever.idf[retriever.vocab["ka"]]
    assert got == pytest.approx(idf(3, 2), rel=1e-12)
    # "gha" appears only in d3
    got = retriever.idf[retriever.vocab["gha"]]
    assert got == pytest.approx(idf(3, 1), rel=1e-12)


def test_single_term_scores_match_hand_computation(retriever: BM25Retriever) -> None:
    scores = retriever.score("ka")
    expected = np.array([
        term_weight(tf=2, doc_len=4, n_docs=3, df=2),  # d1 has "ka" twice
        term_weight(tf=1, doc_len=2, n_docs=3, df=2),  # d2 once
        0.0,                                           # d3 not at all
    ])
    assert scores == pytest.approx(expected, rel=1e-12)


def test_multi_term_query_is_additive(retriever: BM25Retriever) -> None:
    got = retriever.score("ka kha")
    expected = retriever.score("ka") + retriever.score("kha")
    assert got == pytest.approx(expected, rel=1e-12)


def test_shorter_document_wins_on_equal_term_frequency(retriever: BM25Retriever) -> None:
    """Length normalisation: d2 mentions 'kha' once in 2 words, d1 once in 4."""
    scores = retriever.score("kha")
    assert scores[1] > scores[0] > 0


def test_repeated_query_term_counts_twice(retriever: BM25Retriever) -> None:
    assert retriever.score("ka ka") == pytest.approx(2 * retriever.score("ka"), rel=1e-12)


def test_search_orders_by_score_and_drops_zeros(retriever: BM25Retriever) -> None:
    hits = retriever.search("ka kha", k=3)
    assert [pid for pid, _ in hits] == ["d2", "d1"]  # d3 scores 0 and is dropped
    assert hits[0][1] >= hits[1][1]


def test_unknown_term_returns_no_hits(retriever: BM25Retriever) -> None:
    assert retriever.search("zzz", k=3) == []
