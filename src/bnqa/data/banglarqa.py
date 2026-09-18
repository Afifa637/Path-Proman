"""BanglaRQA ingest, leakage gate and answer-alignment audit (PLAN.md task T1).

Two facts verified on 17 Sep 2026 shape this module (PLAN.md §4.1):

1. **The repo ships a loader script.**  ``datasets`` v3+ refuses to execute
   dataset scripts, so ``load_dataset("sartajekram/BanglaRQA")`` fails.  We read
   the three raw JSON files instead.  ``datasets`` is not a dependency.
2. **There is no ``answer_start`` field.**  The schema is::

       data[] -> {passage_id, context, title,
                  qas[] -> {question_id, question_text, is_answerable,
                            question_type, answers:{answer_text[], answer_type[]}}}

   Span supervision therefore requires locating each answer string inside its
   context ourselves.  That is the most under-appreciated risk in the project,
   so the alignment is a **measured, reported quantity**, not an assumption:
   :func:`align_answer` runs a four-step cascade and every item records which
   step succeeded.  If the overall rate falls below ``ALIGNMENT_FLOOR`` the
   reader falls back to sentence-level answers (PLAN.md §17) — VC-1 still holds,
   the span is just longer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from ..config import (CFG, QA_TEST, QA_TRAIN, QA_VAL, RAW_BANGLARQA, SOURCES,
                      ensure_dirs)
from ..preprocess.normalize import normalize, normalize_for_match
from ..utils import log_result, save_table, write_jsonl

SPLIT_FILES = {"train": "Train.json", "val": "Validation.json", "test": "Test.json"}
SPLIT_OUT = {"train": QA_TRAIN, "val": QA_VAL, "test": QA_TEST}
EXPECTED_COUNTS = {"train": 11_912, "val": 1_484, "test": 1_493}

#: below this, T10 switches to sentence-level answers (PLAN.md §17).  Measured
#: over **span-typed answers only** — see :func:`build` for why.
ALIGNMENT_FLOOR = 0.85
#: fuzzy alignment is allowed this much relative edit distance, and is logged separately
FUZZY_MAX_RATIO = 0.10

#: BanglaRQA encodes a ``multiple spans`` answer as one semicolon-delimited
#: string.  Discovered empirically in T1: 91% of multi-span alignment failures
#: are recovered by splitting on it, because the *parts* occur verbatim in the
#: context while the joined string never does.
MULTISPAN_SEP = ";"

#: Answer types that are spans of the passage.  ``yes/no`` is not one of them:
#: হ্যাঁ/না are *generated* labels that by construction do not occur as spans, so
#: including them in the alignment denominator measures the wrong thing.  They
#: are routed to the answer-type head and the verifier's polarity check instead
#: (PLAN.md §7.6, §10.7), and reported separately below.
SPAN_TYPES = {"single span", "multiple spans"}

_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Answer alignment                                                             #
# --------------------------------------------------------------------------- #


def align_answer(context: str, answer: str) -> tuple[int, int, str] | None:
    """Locate ``answer`` inside ``context``; return ``(start, end, method)``.

    Offsets index the **normalised** context — the same string stored in
    ``passages.jsonl`` — so a span can be re-checked byte-for-byte later by
    ``verify_receipt`` (VC-1/VC-3).

    The cascade, strictest first:

    ``exact``       the answer occurs verbatim
    ``normalized``  it occurs after NFC/zero-width/digit folding
    ``whitespace``  it occurs ignoring internal whitespace runs
    ``fuzzy``       best window within 10% edit distance — **logged separately**,
                    never silently pooled with the exact matches
    """
    if not answer or not context:
        return None

    idx = context.find(answer)
    if idx >= 0:
        return idx, idx + len(answer), "exact"

    n_ans = normalize(answer)
    if n_ans:
        idx = context.find(n_ans)
        if idx >= 0:
            return idx, idx + len(n_ans), "normalized"

    # whitespace-insensitive: build a regex that lets any run of spaces vary
    parts = [re.escape(p) for p in _WS.split(n_ans) if p]
    if parts:
        pattern = r"\s+".join(parts)
        m = re.search(pattern, context)
        if m:
            return m.start(), m.end(), "whitespace"

    return _fuzzy_align(context, n_ans)


def align_answer_parts(context: str, answer: str, answer_type: str | None) -> list[dict]:
    """Align one gold answer, splitting ``multiple spans`` into its parts.

    A ``multiple spans`` answer arrives as ``"ডাইভিং; সিনক্রোনাইজড সাঁতার; ওয়াটার
    পোলো"`` — the joined string occurs nowhere in the passage, but each part
    occurs verbatim.  Returning a list of spans is also exactly what list-type
    questions need downstream, where the answer is scored as a *set*.
    """
    answer = answer.strip()
    if not answer:
        return []

    parts = [answer]
    if answer_type == "multiple spans" and MULTISPAN_SEP in answer:
        split = [p.strip() for p in answer.split(MULTISPAN_SEP) if p.strip()]
        if len(split) > 1:
            parts = split

    out: list[dict] = []
    for part in parts:
        hit = align_answer(context, part)
        # ``text`` is the PASSAGE's own characters for this span, not the gold
        # answer string.  They differ whenever alignment was whitespace-tolerant
        # or fuzzy, and it is the passage's text that VC-1 asserts and that
        # verify_receipt re-checks — so storing the gold string here would make
        # the substring property false for exactly the spans where it matters.
        # The gold string is kept alongside for scoring.
        out.append({
            "text": context[hit[0]:hit[1]] if hit else part,
            "gold_text": part,
            "start": hit[0] if hit else None,
            "end": hit[1] if hit else None,
            "align_method": hit[2] if hit else None,
        })
    return out


def _fuzzy_align(context: str, answer: str) -> tuple[int, int, str] | None:
    """Best same-length window within ``FUZZY_MAX_RATIO`` edit distance.

    Anchored on a rare token from the answer so this stays linear in practice
    rather than scanning every offset of a 4,000-character context.
    """
    if len(answer) < 4:
        return None

    anchors = sorted(set(_WS.split(answer)), key=len, reverse=True)[:3]
    positions: list[int] = []
    for anchor in anchors:
        if len(anchor) < 3:
            continue
        start = 0
        while len(positions) < 64:
            hit = context.find(anchor, start)
            if hit < 0:
                break
            positions.append(hit)
            start = hit + 1
    if not positions:
        return None

    best: tuple[float, int, int] | None = None
    span = len(answer)
    for pos in positions:
        for lo in range(max(0, pos - span), min(len(context) - 1, pos + 1)):
            hi = min(len(context), lo + span)
            window = context[lo:hi]
            if not window:
                continue
            ratio = SequenceMatcher(None, window, answer).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, lo, hi)
    if best and best[0] >= 1.0 - FUZZY_MAX_RATIO:
        return best[1], best[2], "fuzzy"
    return None


# --------------------------------------------------------------------------- #
# Parsing                                                                      #
# --------------------------------------------------------------------------- #


def _read_split(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["data"]


def parse_split(split: str, raw_dir: Path | None = None) -> tuple[list[dict], dict[str, str], Counter]:
    """Return ``(qa_rows, passages_by_id, alignment_counter)``."""
    raw_dir = raw_dir or RAW_BANGLARQA
    records = _read_split(raw_dir / SPLIT_FILES[split])

    qa_rows: list[dict] = []
    passages: dict[str, str] = {}
    methods: Counter = Counter()
    # BanglaRQA reuses a handful of question_ids for *different* questions within
    # a split (7 across the release).  Receipts and per-question bookkeeping are
    # keyed by qid, so collisions are disambiguated here and counted, rather than
    # silently overwriting each other downstream.
    seen_qids: dict[str, int] = {}

    for rec in records:
        pid = rec["passage_id"]
        context = normalize(rec["context"])
        passages[pid] = context
        title = normalize(rec.get("title") or "")

        for qa in rec.get("qas", []):
            answers_blob = qa.get("answers") or {}
            texts = [normalize(a) for a in (answers_blob.get("answer_text") or []) if a]
            types = list(answers_blob.get("answer_type") or [])
            is_answerable = bool(qa.get("is_answerable"))

            aligned: list[dict] = []
            for i, ans in enumerate(texts):
                atype = types[i] if i < len(types) else (types[0] if types else None)
                spans = align_answer_parts(context, ans, atype)
                ok = bool(spans) and all(s["start"] is not None for s in spans)
                for s in spans:
                    methods[(atype, s["align_method"] or "unaligned")] += 1
                aligned.append({
                    "text": ans,
                    "answer_type": atype,
                    "spans": spans,
                    "aligned": ok,
                    # convenience mirrors of the first span, for the single-span path
                    "start": spans[0]["start"] if spans else None,
                    "end": spans[0]["end"] if spans else None,
                    "align_method": spans[0]["align_method"] if spans else None,
                })

            raw_qid = qa["question_id"]
            n_seen = seen_qids.get(raw_qid, 0)
            seen_qids[raw_qid] = n_seen + 1
            if n_seen:
                methods[("_meta", "duplicate_qid")] += 1
            qid = raw_qid if not n_seen else f"{raw_qid}#{n_seen + 1}"

            qa_rows.append({
                "qid": qid,
                "source_qid": raw_qid,
                "question": normalize(qa["question_text"]),
                "question_type": qa.get("question_type"),
                "is_answerable": is_answerable,
                "gold_passage_id": pid,
                "passage_title": title,
                "answers": aligned,
                "answer_types": types,
                "split": split,
            })

    return qa_rows, passages, methods


# --------------------------------------------------------------------------- #
# Leakage gate                                                                 #
# --------------------------------------------------------------------------- #


def leakage_report(passages_by_split: dict[str, dict[str, str]]) -> dict:
    """Assert no ``passage_id`` appears in two splits (PLAN.md T1).

    If the official split were question-level, the same passage could sit in both
    train and test and every retrieval number would be inflated.  This is the
    gate that catches it; ``tests/test_splits.py`` asserts the result.
    """
    ids = {s: set(p) for s, p in passages_by_split.items()}
    overlaps = {
        f"{a}|{b}": sorted(ids[a] & ids[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return {
        "counts": {s: len(v) for s, v in ids.items()},
        "overlap_sizes": {k: len(v) for k, v in overlaps.items()},
        "clean": all(len(v) == 0 for v in overlaps.values()),
        "examples": {k: v[:5] for k, v in overlaps.items() if v},
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def build() -> dict:
    ensure_dirs()
    print("T1  BanglaRQA ingest")

    qa_by_split: dict[str, list[dict]] = {}
    passages_by_split: dict[str, dict[str, str]] = {}
    methods_by_split: dict[str, Counter] = {}

    for split in ("train", "val", "test"):
        rows, passages, methods = parse_split(split)
        qa_by_split[split] = rows
        passages_by_split[split] = passages
        methods_by_split[split] = methods
        n = write_jsonl(SPLIT_OUT[split], rows)
        expected = EXPECTED_COUNTS[split]
        flag = "OK" if n == expected else f"!! expected {expected}"
        print(f"  {split:5s}  {n:6,d} questions  {len(passages):5,d} passages   {flag}")

    # ---- leakage gate ----------------------------------------------------
    leak = leakage_report(passages_by_split)
    print(f"  leakage gate: {'CLEAN - no passage_id in two splits' if leak['clean'] else 'LEAK'}")
    if not leak["clean"]:
        print(f"    overlaps: {leak['overlap_sizes']}")
    save_table("splits", [
        {"split": s, "questions": len(qa_by_split[s]), "passages": leak["counts"][s],
         "expected_questions": EXPECTED_COUNTS[s],
         "matches_published": len(qa_by_split[s]) == EXPECTED_COUNTS[s]}
        for s in ("train", "val", "test")
    ])

    # ---- alignment audit -------------------------------------------------
    # Reported per answer_type, because the denominators mean different things.
    # yes/no answers are labels, not spans: they cannot align by construction and
    # pooling them into one rate understates span supervision by ~8 points.  The
    # floor is therefore checked against SPAN_TYPES only, and the yes/no figure is
    # printed beside it rather than dropped.
    total: Counter = Counter()
    for m in methods_by_split.values():
        total.update(m)

    methods = ("exact", "normalized", "whitespace", "fuzzy", "unaligned")
    dup_qids = total.pop(("_meta", "duplicate_qid"), 0)
    by_type: dict[str, Counter] = {}
    for (atype, method), count in total.items():
        by_type.setdefault(atype or "(blank)", Counter())[method] += count

    rows: list[dict] = []
    print("  answer alignment (the release has no answer_start, so we locate spans ourselves):")
    print(f"    {'answer_type':16s} {'aligned':>16s}   " + "".join(f"{m[:5]:>8s}" for m in methods))
    for atype, counter in sorted(by_type.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(counter.values())
        ok = n - counter["unaligned"]
        print(f"    {atype:16s} {ok:7,d}/{n:7,d} {100 * ok / max(n, 1):5.1f}%  "
              + "".join(f"{counter[m]:8,d}" for m in methods))
        rows.append({"answer_type": atype, "parts": n, "aligned": ok,
                     "pct": round(100 * ok / max(n, 1), 2),
                     **{m: counter[m] for m in methods},
                     "counted_toward_floor": atype in SPAN_TYPES})

    span_counter: Counter = Counter()
    for atype in SPAN_TYPES:
        span_counter.update(by_type.get(atype, Counter()))
    n_span = sum(span_counter.values())
    span_ok = n_span - span_counter["unaligned"]
    strict_ok = span_counter["exact"] + span_counter["normalized"] + span_counter["whitespace"]
    rate = span_ok / n_span if n_span else 0.0
    strict_rate = strict_ok / n_span if n_span else 0.0
    viable = rate >= ALIGNMENT_FLOOR

    yn = by_type.get("yes/no", Counter())
    n_yn = sum(yn.values())

    print(f"  -> span-typed answers: strict {100 * strict_rate:.1f}%  any {100 * rate:.1f}%  "
          f"(floor {100 * ALIGNMENT_FLOOR:.0f}%)  "
          f"{'PASS - span-level reader is viable' if viable else 'BELOW FLOOR - see PLAN 17'}")
    print(f"  -> yes/no answers ({n_yn:,d}) are labels, not spans: reported separately, "
          f"routed to the answer-type head + polarity check")
    if dup_qids:
        print(f"  -> {dup_qids} question_ids are reused upstream for different questions; "
              f"disambiguated with a #n suffix (source_qid keeps the original)")

    save_table("answer_alignment", rows)

    payload = {
        "questions": {s: len(v) for s, v in qa_by_split.items()},
        "passages": leak["counts"],
        "leakage_clean": leak["clean"],
        "span_parts_total": n_span,
        "span_alignment_rate_any": round(rate, 4),
        "span_alignment_rate_strict": round(strict_rate, 4),
        "yes_no_answers": n_yn,
        "alignment_by_type": {t: dict(c) for t, c in by_type.items()},
        "alignment_floor": ALIGNMENT_FLOOR,
        "span_reader_viable": viable,
        "multispan_separator": MULTISPAN_SEP,
        "duplicate_qids_disambiguated": dup_qids,
    }
    log_result("t1_banglarqa", "ingest", payload)

    # gold contexts are needed by T3; keep them beside the QA files
    from ..config import PROCESSED

    all_passages = {pid: text for p in passages_by_split.values() for pid, text in p.items()}
    write_jsonl(PROCESSED / "gold_contexts.jsonl",
                ({"pid": pid, "text": text} for pid, text in sorted(all_passages.items())))
    print(f"  gold contexts: {len(all_passages):,d} -> data/processed/gold_contexts.jsonl")

    from .fetch import update_rows

    for split in ("train", "val", "test"):
        update_rows(f"banglarqa_{'val' if split == 'val' else split}",
                    len(qa_by_split[split]), note=f"{leak['counts'][split]} passages")
    return payload


if __name__ == "__main__":
    build()
