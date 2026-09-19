"""VC-7 — the Counterfactual Corpus Test (PLAN.md §11, task V3).

The most persuasive thing in the project, and a **reportable metric** rather
than a demo trick.  It answers the examiner's real question — *"how do we know
it isn't just a language model guessing?"* — with a number.

Procedure, for *N* questions the system answers correctly with high confidence:

1. **perturb** — in a *copy* of the gold passage, replace the answer string
   with a plausible same-type alternative (১৯৭১→১৯৬৯, a name → another name
   from the corpus).  Only that passage changes.
2. **re-ask** the identical question against the tampered corpus.
3. **score** — did the answer follow the corpus?

**CAR — Corpus Attribution Rate** = fraction of cases where the answer changes
to the perturbed value.  A genuinely retrieval-grounded extractive system
scores ≈1.0.  A model answering from memorised parameters cannot.

**AoRR — Abstention-on-Removal Rate** = delete the gold passage entirely; the
system should **abstain**, not fabricate.

Both numbers are only meaningful on items the system got *right* to begin
with, so the eligible set is filtered first and its size is reported — a CAR
computed over items the system never answered correctly would measure nothing.
"""

from __future__ import annotations

import random
from typing import Sequence

from ..config import CFG, COUNTERFACTUAL
from ..preprocess.normalize import normalize
from ..utils import write_jsonl
from ..verify.banglaverify import ValuePools
from ..verify.constraints import years

TAMPERED_PATH = COUNTERFACTUAL / "passages_tampered.jsonl"


# --------------------------------------------------------------------------- #
# Perturbation                                                                 #
# --------------------------------------------------------------------------- #


def perturb_passage(passage: dict, answer: str, pools: ValuePools,
                    rng: random.Random) -> tuple[dict, str] | None:
    """A copy of ``passage`` with ``answer`` replaced by a same-type alternative."""
    text = normalize(passage["text"])
    if answer not in text:
        return None

    ys = years(answer)
    if ys:
        replacement = pools.other(pools.years, ys[0])
        if replacement is None:
            return None
        new_answer = answer.replace(ys[0], replacement, 1)
    elif any(ch.isdigit() for ch in answer):
        import re

        nums = re.findall(r"\d+", answer)
        replacement = pools.other(pools.numbers, nums[0]) if nums else None
        if replacement is None:
            return None
        new_answer = answer.replace(nums[0], replacement, 1)
    else:
        candidates = [e for e in pools.entities
                      if e != answer and e not in text and 2 < len(e) < 40]
        if not candidates:
            return None
        new_answer = rng.choice(candidates)

    if new_answer == answer:
        return None
    return {**passage, "text": text.replace(answer, new_answer, 1),
            "tampered": True, "original_answer": answer,
            "tampered_answer": new_answer}, new_answer


def tampered_corpus(corpus: Sequence[dict], pid: str, replacement: dict) -> list[dict]:
    """The corpus with exactly one passage swapped.  Everything else is identical."""
    return [replacement if p["pid"] == pid else p for p in corpus]


def corpus_without(corpus: Sequence[dict], pid: str) -> list[dict]:
    return [p for p in corpus if p["pid"] != pid]


# --------------------------------------------------------------------------- #
# The experiment                                                               #
# --------------------------------------------------------------------------- #


def run(pipeline_factory, corpus: Sequence[dict], questions: Sequence[dict], *,
        n: int | None = None, min_confidence: float = 0.0, verbose: bool = True) -> dict:
    """``pipeline_factory(corpus) -> BanglaQA``; returns CAR, AoRR and the rows.

    A factory rather than a pipeline, because each arm runs against a
    *different corpus* and rebuilding the retriever is the only honest way to
    do that — a cached index would answer from passages that no longer exist.
    """
    n = CFG.counterfactual_n if n is None else n
    rng = random.Random(CFG.seed)

    base = pipeline_factory(list(corpus))
    pools = ValuePools([p["text"] for p in corpus[:5000]], seed=CFG.seed)

    rows: list[dict] = []
    tampered_records: list[dict] = []
    considered = 0

    for q in questions:
        if len(rows) >= n:
            break
        considered += 1
        golds = [a["text"] for a in q.get("answers", []) if a.get("text")]
        if not golds:
            continue

        result = base.ask(q["question"], qid=q["qid"])
        if result.abstained or not result.answer:
            continue
        from .qa_metrics import token_f1

        if max(token_f1(result.answer, g) for g in golds) < 0.9:
            continue          # only items the system already gets right
        if result.confidence < min_confidence:
            continue

        passage = next((p for p in corpus if p["pid"] == result.pid), None)
        if passage is None:
            continue
        made = perturb_passage(passage, result.answer, pools, rng)
        if made is None:
            continue
        swapped, new_answer = made
        tampered_records.append(swapped)

        # ---- arm 1: the perturbed corpus --------------------------------
        cf = pipeline_factory(tampered_corpus(corpus, result.pid, swapped))
        cf_result = cf.ask(q["question"], qid=q["qid"])
        followed = normalize(cf_result.answer).strip() == normalize(new_answer).strip()

        # ---- arm 2: the passage removed ---------------------------------
        gone = pipeline_factory(corpus_without(corpus, result.pid))
        gone_result = gone.ask(q["question"], qid=q["qid"])
        abstained = gone_result.abstained or not gone_result.answer

        rows.append({
            "qid": q["qid"], "question": q["question"],
            "original_answer": result.answer, "original_confidence": round(result.confidence, 4),
            "pid": result.pid,
            "perturbed_to": new_answer,
            "counterfactual_answer": cf_result.answer,
            "counterfactual_confidence": round(cf_result.confidence, 4),
            "followed_corpus": followed,
            "removal_answer": gone_result.answer,
            "abstained_on_removal": abstained,
        })
        if verbose:
            mark = "OK " if followed else "NO "
            print(f"    [{mark}] {result.answer!r} -> corpus says {new_answer!r}, "
                  f"system said {cf_result.answer!r}"
                  f"{'  | abstained on removal' if abstained else '  | still answered'}")

    car = sum(1 for r in rows if r["followed_corpus"]) / max(len(rows), 1)
    aorr = sum(1 for r in rows if r["abstained_on_removal"]) / max(len(rows), 1)

    if tampered_records:
        write_jsonl(TAMPERED_PATH, tampered_records)

    return {
        "n": len(rows),
        "considered": considered,
        "car": round(car, 4),
        "aorr": round(aorr, 4),
        "rows": rows,
        "interpretation": (
            "CAR ~ 1.0 means the answer is read out of the corpus, not recalled "
            "from parameters; a generative model cannot do this."),
    }
