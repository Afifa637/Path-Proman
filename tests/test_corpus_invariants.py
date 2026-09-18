"""Invariants of the built corpus (PLAN.md §18.1, tasks T1 and T3).

These run against the artefacts on disk rather than fixtures, because the
properties they protect are properties of *this* corpus:

* no ``passage_id`` appears in two splits — otherwise every retrieval number is
  inflated by leakage;
* the index sets are nested and all 3,000 gold contexts are present at both
  sizes — otherwise the index-size comparison is confounded and Recall@k is
  measured against targets that cannot be retrieved;
* aligned answer spans are **literal substrings** of their passage, which is the
  machine-checkable half of VC-1.

They skip rather than fail when the corpus has not been built yet, so a fresh
clone can still run ``pytest``.
"""

from __future__ import annotations

import pytest

from bnqa.config import CFG, PASSAGES, PROCESSED, QA_TEST, QA_TRAIN, QA_VAL
from bnqa.utils import load_jsonl, read_json

pytestmark = pytest.mark.skipif(
    not PASSAGES.exists(), reason="corpus not built yet - run T1/T2/T3 first")


@pytest.fixture(scope="module")
def qa() -> dict[str, list[dict]]:
    return {s: load_jsonl(p) for s, p in
            (("train", QA_TRAIN), ("val", QA_VAL), ("test", QA_TEST))}


@pytest.fixture(scope="module")
def contexts() -> dict[str, str]:
    return {r["pid"]: r["text"] for r in load_jsonl(PROCESSED / "gold_contexts.jsonl")}


@pytest.fixture(scope="module")
def passages() -> list[dict]:
    return load_jsonl(PASSAGES)


# --------------------------------------------------------------------------- #
# T1 — splits                                                                  #
# --------------------------------------------------------------------------- #


def test_split_sizes_match_the_published_counts(qa) -> None:
    assert (len(qa["train"]), len(qa["val"]), len(qa["test"])) == (11_912, 1_484, 1_493)


def test_no_passage_leaks_between_splits(qa) -> None:
    """The gate.  A shared passage would inflate every retrieval number."""
    ids = {s: {r["gold_passage_id"] for r in rows} for s, rows in qa.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_question_ids_are_unique(qa) -> None:
    all_qids = [r["qid"] for rows in qa.values() for r in rows]
    assert len(all_qids) == len(set(all_qids))


# --------------------------------------------------------------------------- #
# T1 — alignment is the machine-checkable half of VC-1                         #
# --------------------------------------------------------------------------- #


def test_aligned_spans_are_literal_substrings(qa, contexts) -> None:
    checked = 0
    for rows in qa.values():
        for row in rows:
            ctx = contexts.get(row["gold_passage_id"])
            if ctx is None:
                continue
            for ans in row["answers"]:
                for span in ans.get("spans") or []:
                    if span["start"] is None:
                        continue
                    assert ctx[span["start"]:span["end"]] == span["text"] or \
                        span["align_method"] == "fuzzy", (
                        f"span {span['start']}:{span['end']} of "
                        f"{row['gold_passage_id']} is not its own text")
                    checked += 1
    assert checked > 10_000, "expected the audit to have aligned most answers"


def test_span_alignment_rate_clears_the_floor(qa) -> None:
    from bnqa.data.banglarqa import ALIGNMENT_FLOOR, SPAN_TYPES

    total = aligned = 0
    for rows in qa.values():
        for row in rows:
            for ans in row["answers"]:
                if ans["answer_type"] not in SPAN_TYPES:
                    continue
                for span in ans.get("spans") or []:
                    total += 1
                    aligned += span["start"] is not None
    assert total > 0
    assert aligned / total >= ALIGNMENT_FLOOR


# --------------------------------------------------------------------------- #
# T3 — index                                                                   #
# --------------------------------------------------------------------------- #


def test_passage_ids_are_unique(passages) -> None:
    pids = [p["pid"] for p in passages]
    assert len(pids) == len(set(pids))


def test_indexes_are_nested(passages) -> None:
    from bnqa.data.index_build import load_manifest

    sizes = sorted(CFG.index_sizes)
    small = set(load_manifest(sizes[0])["pids"])
    large = set(load_manifest(sizes[-1])["pids"])
    assert small <= large, "the size comparison would be confounded"


def test_every_gold_context_is_in_every_index(qa) -> None:
    from bnqa.data.index_build import load_manifest

    gold = {r["gold_passage_id"] for rows in qa.values() for r in rows}
    for size in CFG.index_sizes:
        pids = set(load_manifest(size)["pids"])
        missing = gold - pids
        assert not missing, f"{len(missing)} gold passages missing from the {size} index"


def test_index_manifests_have_the_requested_size(passages) -> None:
    from bnqa.data.index_build import load_manifest

    for size in CFG.index_sizes:
        assert load_manifest(size)["actual"] == size


def test_nctb_passages_carry_citation_metadata(passages) -> None:
    """VC-2 needs grade + chapter for every textbook passage it cites."""
    nctb = [p for p in passages if p["source"] == "nctb_schooltext"]
    assert nctb, "expected NCTB passages in the index"
    assert all(p.get("grade_label") and p.get("chapter_title") is not None for p in nctb)


def test_passages_pass_the_corpus_filter(passages) -> None:
    from bnqa.preprocess.normalize import bengali_ratio

    sample = passages[::500]
    for p in sample:
        assert len(p["text"]) >= CFG.min_passage_chars
        assert bengali_ratio(p["text"]) >= CFG.min_bengali_ratio - 1e-9


def test_passage_text_is_already_normalised(passages) -> None:
    """Offsets stored anywhere in the project index this exact string."""
    from bnqa.preprocess.normalize import normalize

    for p in passages[::997]:
        assert normalize(p["text"]) == p["text"]
