"""T5's acceptance: **our BM25 matches a hand-worked example** (PLAN.md §7.3).

PLAN.md §0.2 claims BM25 is ours rather than a library call.  A claim like that
is worth exactly as much as the test behind it, so this file works the scoring
function out by hand on a three-document corpus and asserts the implementation
agrees to six decimal places.

The hand computation, for the record — three documents, query ``ক``:

    d0 = "ক খ গ"        |d0| = 3
    d1 = "ক ক খ"        |d1| = 3
    d2 = "ঘ ঙ চ"        |d2| = 3       avgdl = 3

    n(ক) = 2, N = 3
    idf(ক) = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6)

With k1 = 1.2, b = 0.75 and |d| = avgdl the length normalisation term is
k1 exactly, so

    score(d0) = ln(1.6) · (1 · 2.2) / (1 + 1.2) = ln(1.6)
    score(d1) = ln(1.6) · (2 · 2.2) / (2 + 1.2) = ln(1.6) · 1.375
    score(d2) = 0
"""

from __future__ import annotations

import math

import pytest

from bnqa.retrieval.bm25 import BM25Retriever

K1, B = 1.2, 0.75
IDF_KA = math.log(1.0 + (3 - 2 + 0.5) / (2 + 0.5))


@pytest.fixture()
def toy():
    """Three single-letter documents, so stopwords and stemming cannot interfere."""
    return [
        {"pid": "d0", "text": "ক খ গ"},
        {"pid": "d1", "text": "ক ক খ"},
        {"pid": "d2", "text": "ঘ ঙ চ"},
    ]


def test_idf_matches_hand_calculation(toy):
    r = BM25Retriever(K1, B).build(toy)
    assert r.idf[r.vocab["ক"]] == pytest.approx(IDF_KA, abs=1e-6)


def test_document_lengths_and_avgdl(toy):
    r = BM25Retriever(K1, B).build(toy)
    assert list(r.doc_len) == [3.0, 3.0, 3.0]
    assert r.avgdl == pytest.approx(3.0)


def test_scores_match_hand_calculation(toy):
    r = BM25Retriever(K1, B).build(toy)
    scores = r.score("ক")

    expected_d0 = IDF_KA * (1 * (K1 + 1)) / (1 + K1)
    expected_d1 = IDF_KA * (2 * (K1 + 1)) / (2 + K1)

    assert scores[0] == pytest.approx(expected_d0, abs=1e-6)
    assert scores[1] == pytest.approx(expected_d1, abs=1e-6)
    assert scores[2] == pytest.approx(0.0, abs=1e-9)


def test_ranking_prefers_higher_term_frequency(toy):
    r = BM25Retriever(K1, B).build(toy)
    hits = r.search("ক", 3)
    assert [pid for pid, _ in hits] == ["d1", "d0"]   # d2 scores 0 and is dropped


def test_repeated_query_term_counts_twice(toy):
    """BM25 sums over the query *multiset*, so "ক ক" doubles the contribution."""
    r = BM25Retriever(K1, B).build(toy)
    once = r.score("ক")
    twice = r.score("ক ক")
    assert twice[0] == pytest.approx(2 * once[0], abs=1e-6)


def test_b_zero_disables_length_normalisation(toy):
    docs = [{"pid": "short", "text": "ক খ"},
            {"pid": "long", "text": "ক খ গ ঘ ঙ চ ছ জ"}]
    r = BM25Retriever(1.2, 0.0, retain_counts=True).build(docs)
    s = r.score("ক")
    assert s[0] == pytest.approx(s[1], abs=1e-6), "with b=0 length must not matter"

    r.set_params(1.2, 1.0)
    s = r.score("ক")
    assert s[0] > s[1], "with b=1 the shorter document must win"


def test_set_params_reweights_without_rebuilding(toy):
    r = BM25Retriever(K1, B, retain_counts=True).build(toy)
    before = r.score("ক").copy()
    r.set_params(2.0, 0.3)
    assert not (r.score("ক") == before).all()
    r.set_params(K1, B)
    assert r.score("ক") == pytest.approx(before, abs=1e-6)


def test_unknown_term_scores_zero(toy):
    r = BM25Retriever(K1, B).build(toy)
    assert r.search("অজানা", 3) == []
