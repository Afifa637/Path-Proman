"""The corpus invariants T1-T3 promise (PLAN.md §14 "Done when").

Every one of these is something a later stage silently depends on:

* the **leakage gate** — a passage in two splits makes every retrieval number
  meaningless;
* **nesting** — the index-size comparison is confounded without it;
* **gold coverage at every size** — Recall@k has no denominator otherwise;
* the **length artefact** — TASKS.md finding #8, where passage length alone
  predicted "is gold" at AUC 0.795 and BM25's ``b`` quietly learned it.

They skip cleanly when the corpus has not been built, so a fresh clone passes
``pytest`` before anything is downloaded (T0's acceptance criterion).
"""

from __future__ import annotations

import pytest

from bnqa.config import CFG, INDEX_MANIFEST, PASSAGES, QA_TEST, QA_TRAIN, QA_VAL
from bnqa.utils import load_jsonl, read_json

needs_corpus = pytest.mark.skipif(
    not PASSAGES.exists(), reason="corpus not built — run the T1-T3 pipeline")
needs_qa = pytest.mark.skipif(
    not QA_TRAIN.exists(), reason="BanglaRQA not ingested — run bnqa.data.banglarqa")


@needs_qa
def test_split_counts_match_the_published_release():
    expected = {QA_TRAIN: 11_912, QA_VAL: 1_484, QA_TEST: 1_493}
    for path, n in expected.items():
        assert len(load_jsonl(path)) == n, f"{path.name} should hold {n} questions"


@needs_qa
def test_no_passage_appears_in_two_splits():
    """The leakage gate.  A failure here invalidates every number downstream."""
    by_split = {}
    for name, path in (("train", QA_TRAIN), ("val", QA_VAL), ("test", QA_TEST)):
        by_split[name] = {r["gold_passage_id"] for r in load_jsonl(path)}
    names = list(by_split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = by_split[a] & by_split[b]
            assert not overlap, f"{len(overlap)} passages shared by {a} and {b}"


@needs_qa
def test_question_ids_are_unique_within_a_split():
    """T1 disambiguates the 7 reused upstream ids; receipts are keyed by qid."""
    for path in (QA_TRAIN, QA_VAL, QA_TEST):
        qids = [r["qid"] for r in load_jsonl(path)]
        assert len(qids) == len(set(qids)), f"duplicate qids in {path.name}"


@needs_corpus
def test_pids_are_unique():
    pids = [p["pid"] for p in load_jsonl(PASSAGES)]
    assert len(pids) == len(set(pids))


@needs_corpus
def test_indexes_are_nested():
    sizes = sorted(CFG.index_sizes)
    small = set(read_json(INDEX_MANIFEST[sizes[0]])["pids"])
    large = set(read_json(INDEX_MANIFEST[sizes[-1]])["pids"])
    assert small <= large, "the index-size comparison would be confounded"


@needs_corpus
def test_every_gold_passage_is_present_at_every_index_size():
    passages = {p["pid"]: p for p in load_jsonl(PASSAGES)}
    gold = {pid for pid, p in passages.items() if p.get("source") == "gold"}
    for size in CFG.index_sizes:
        pids = set(read_json(INDEX_MANIFEST[size])["pids"])
        missing = gold - pids
        assert not missing, f"{len(missing)} gold passages missing from the {size} index"


@needs_corpus
def test_index_sizes_are_what_the_manifest_claims():
    for size in CFG.index_sizes:
        manifest = read_json(INDEX_MANIFEST[size])
        assert len(manifest["pids"]) == manifest["actual"] == size


@needs_corpus
def test_citable_passages_carry_full_citation_metadata():
    """VC-2: a citation line must never be invented for an uncitable source."""
    for p in load_jsonl(PASSAGES):
        if p.get("chapter_title"):
            assert p.get("subject"), f"{p['pid']} has a chapter but no subject"
            assert p.get("grade") or p.get("grade_label"), f"{p['pid']} has no grade"


@needs_corpus
def test_wiki_distractors_are_length_matched_to_gold():
    """TASKS.md finding #8 — the artefact that once tuned BM25's ``b`` for us.

    Only the Wikipedia pool is asserted.  Its chunk lengths are sampled from
    the gold distribution, so chance separability is a property we control and
    a regression in the chunker must fail here.  NCTB-SchoolText keeps its own
    chunking because that chunking carries the chapter metadata VC-2 cites, so
    it is shorter by construction; that number is reported in the audit rather
    than asserted, because the only way to "fix" it is to destroy the citation
    unit.
    """
    from bnqa.data.audit import length_artefact

    result = length_artefact(load_jsonl(PASSAGES))
    wiki = result["auc_by_source"].get("wiki")
    assert wiki is not None
    assert wiki < 0.60, (
        f"passage length separates gold from Wikipedia distractors at AUC {wiki} "
        "— the length-matched sampling has regressed and BM25's b will learn it")
