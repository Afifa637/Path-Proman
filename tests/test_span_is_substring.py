"""VC-1, the test PLAN.md §11 names by file — **extractive by construction**.

The claim is that the answer is *always* a literal character span of a corpus
passage: ``passages[pid].text[start:end] == answer``.  It is a property of the
architecture rather than of the model, so it is asserted structurally here and
over every real prediction when the corpus is built.

The corpus-dependent tests skip cleanly when ``data/processed`` is absent, so
``pytest`` passes on a fresh clone before anything has been downloaded — T0's
acceptance criterion requires exactly that.
"""

from __future__ import annotations

import pytest

from bnqa.config import PASSAGES, QA_VAL
from bnqa.preprocess.normalize import normalize
from bnqa.reader.candidates import generate
from bnqa.utils import load_jsonl

needs_corpus = pytest.mark.skipif(
    not PASSAGES.exists(), reason="corpus not built — run the T1-T3 pipeline")


# --------------------------------------------------------------------------- #
# Structural: candidate generation can only produce literal slices             #
# --------------------------------------------------------------------------- #


def test_every_generated_candidate_is_a_literal_slice():
    passages = [{
        "pid": "p1",
        "text": "সালোকসংশ্লেষণ প্রক্রিয়ায় উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে। "
                "এটি ১৯৭১ সালে আবিষ্কৃত হয়নি।",
    }]
    cands = generate("উদ্ভিদ কোন গ্যাস গ্রহণ করে?", passages)
    assert cands, "the generator produced nothing to check"

    text = normalize(passages[0]["text"])
    for c in cands:
        assert text[c.char_start:c.char_end] == c.text, (
            f"candidate {c.text!r} is not text[{c.char_start}:{c.char_end}]")


def test_candidate_offsets_are_within_the_passage():
    passages = [{"pid": "p1", "text": "ক খ গ ঘ ঙ চ ছ জ ঝ ঞ।"}]
    text = normalize(passages[0]["text"])
    for c in generate("ক কী?", passages):
        assert 0 <= c.char_start < c.char_end <= len(text)


def test_answer_verify_substring_helper():
    from bnqa.reader.base import Answer

    text = "উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে"
    start = text.index("কার্বন")
    end = start + len("কার্বন ডাই-অক্সাইড")
    ans = Answer(text=text[start:end], pid="p", char_start=start, char_end=end,
                 sentence=text, sentence_start=0, sentence_end=len(text))
    assert ans.verify_substring(text)

    ans.text = "অক্সিজেন"
    assert not ans.verify_substring(text)


# --------------------------------------------------------------------------- #
# Against the real corpus                                                      #
# --------------------------------------------------------------------------- #


@needs_corpus
def test_passages_are_stored_normalised():
    """Offsets are meaningless unless the stored text is already canonical."""
    for p in load_jsonl(PASSAGES)[:500]:
        assert p["text"] == normalize(p["text"]), (
            f"{p['pid']} is not stored in normalised form; every span offset "
            "into it would be wrong")


@needs_corpus
def test_gold_spans_are_literal_slices_of_their_context():
    """T1's alignment must have produced offsets, not approximations."""
    from bnqa.config import PROCESSED

    contexts = {r["pid"]: r["text"]
                for r in load_jsonl(PROCESSED / "gold_contexts.jsonl")}
    checked = 0
    for row in load_jsonl(QA_VAL)[:400]:
        text = contexts.get(row["gold_passage_id"])
        if text is None:
            continue
        for ans in row.get("answers", []):
            for span in ans.get("spans") or []:
                if span.get("start") is None:
                    continue
                assert text[span["start"]:span["end"]] == span["text"], (
                    "a gold span's text is not the passage's own characters — "
                    "this is finding #2 in TASKS.md and it breaks VC-1")
                checked += 1
    assert checked > 0, "no aligned gold spans were available to check"


@needs_corpus
def test_receipt_roundtrip_passes_its_own_check():
    from bnqa.verify.receipt import build_receipt, check_receipt

    passage = load_jsonl(PASSAGES)[0]
    text = normalize(passage["text"])
    start, end = 0, min(12, len(text))
    receipt = build_receipt(
        qid="test", question="?", answer=text[start:end], pid=passage["pid"],
        char_start=start, char_end=end, passage_text=text,
        evidence=text[:80], confidence=0.9, signals={}, supported=True)
    report = check_receipt(receipt, {passage["pid"]: text})
    assert report["passed"], report["checks"]


@needs_corpus
def test_tampered_passage_fails_the_receipt_check():
    """VC-3's hash must actually bind the receipt to a corpus state."""
    from bnqa.verify.receipt import build_receipt, check_receipt

    passage = load_jsonl(PASSAGES)[0]
    text = normalize(passage["text"])
    receipt = build_receipt(
        qid="test", question="?", answer=text[:10], pid=passage["pid"],
        char_start=0, char_end=10, passage_text=text, evidence=text[:80],
        confidence=0.9, signals={}, supported=True)

    report = check_receipt(receipt, {passage["pid"]: "সম্পূর্ণ ভিন্ন একটি অনুচ্ছেদ"})
    assert not report["passed"]
