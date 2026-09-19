"""BanglaVerify — an auto-constructed minimal-pair support set (§10.3, T11b).

S3 needs negatives that teach **contradiction**, not topic mismatch.  Random
negatives teach nothing about contradiction: a classifier trained on them
learns "is this about the same subject?", which is precisely the similarity
detector §10.7 says we must not ship.  So the CONTRADICTED half is built by
**minimal-pair perturbation** — swap exactly one critical token, keep every
other token identical — with one generator per veto category.

Every item is a **(claim, evidence-passage)** pair.  The claim is a gold
sentence; the evidence is a passage.  Pairing a claim against its own sentence
would make SUPPORTED a string-equality test — see :func:`_evidence` for the
bug that caused and why it scored 1.0000.

===============  =============================================  =====
label            construction                                   ~size
===============  =============================================  =====
SUPPORTED        gold sentence + its own containing passage      ~8k
CONTRADICTED     six minimal-pair generators, one per category   ~8k
NEUTRAL          same claim + a different passage                ~8k
===============  =============================================  =====

Quality control, because this is constructed data and an examiner will press
exactly here:

* a **300-item stratified sample** is written to
  ``reports/tables/banglaverify_sample.csv`` for hand verification, and
  ``banglaverify_quality.csv`` reports **label precision per generator**.  A
  generator below ~90% is fixed or dropped;
* a **500-pair contrast set** is held out and **never used in training**, and
  scored separately — near-chance accuracy there means the verifier is just a
  similarity detector, and the report has to say so.

**License:** derivative of BanglaRQA, therefore CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from ..config import CFG, PROCESSED, QA_TRAIN
from ..preprocess.normalize import normalize
from ..preprocess.tokenize import sentence_spans, tokenize
from ..utils import load_jsonl, log_result, save_table, set_seed, write_jsonl
from .constraints import MONTHS, NEGATION, UNITS, entities, years

SUPPORTED, CONTRADICTED, NEUTRAL = "SUPPORTED", "CONTRADICTED", "NEUTRAL"
LABELS = (SUPPORTED, CONTRADICTED, NEUTRAL)

GENERATORS = ("numeral", "date", "unit", "entity", "polarity", "relation_order")

OUT_TRAIN = PROCESSED / "banglaverify_train.jsonl"
OUT_VAL = PROCESSED / "banglaverify_val.jsonl"
OUT_CONTRAST = PROCESSED / "banglaverify_contrast.jsonl"

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class Item:
    evidence: str
    statement: str
    label: str
    generator: str       # which rule made it; "gold" for SUPPORTED
    qid: str
    pid: str
    swapped_from: str = ""
    swapped_to: str = ""
    # The unperturbed claim, kept only on CONTRADICTED items.  auto_quality
    # scores each generator against *this*, not against the whole passage,
    # because that is the comparison the veto layer actually performs in the
    # pipeline (answer vs its evidence sentence).  Checking a negated sentence
    # against a long passage is meaningless: almost every passage contains a
    # negation somewhere, so the polarity rule never fires.
    source_statement: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Type-matched value pools, drawn from the corpus itself                       #
# --------------------------------------------------------------------------- #


class ValuePools:
    """Same-type replacement values, collected from the evidence sentences.

    Drawing the replacement **from the corpus** rather than from a fixed list
    is what makes the negative plausible: ১৯৭১ -> ১৯৬৯ is a contradiction a
    student could actually believe, while ১৯৭১ -> ৪৭৩২৯ is one no model needs
    a contradiction detector to reject.
    """

    def __init__(self, sentences: Iterable[str], seed: int = CFG.seed) -> None:
        from ..preprocess.stopwords import content_tokens

        self.rng = random.Random(seed)
        self.years: list[str] = []
        self.numbers: list[str] = []
        self.units: list[str] = []
        self.entities: list[str] = []
        term_counts: Counter = Counter()
        seen_e: set[str] = set()
        for sent in sentences:
            ys = years(sent)
            self.years.extend(ys)
            self.numbers.extend(n for n in _NUMBER_RE.findall(sent) if n not in ys)
            self.units.extend(t.lower() for t in tokenize(sent) if t.lower() in UNITS)
            for e in entities(sent):
                if e not in seen_e and len(e) > 2:
                    seen_e.add(e)
                    self.entities.append(e)
            term_counts.update(t for t in content_tokens(tokenize(sent)) if len(t) > 3)
        self.years = sorted(set(self.years))
        self.numbers = sorted(set(self.numbers))
        self.units = sorted(set(self.units))
        # Content terms for the entity generator's fallback.  Restricted to
        # terms seen at least twice, so a typo cannot become a "plausible
        # alternative" — the negative has to be one a student could believe.
        self.terms = sorted(t for t, n in term_counts.items() if n >= 2)

    def other(self, pool: list[str], current: str) -> str | None:
        choices = [v for v in pool if v != current]
        return self.rng.choice(choices) if choices else None


# --------------------------------------------------------------------------- #
# The six generators — each swaps exactly one token                            #
# --------------------------------------------------------------------------- #


def gen_date(statement: str, pools: ValuePools) -> tuple[str, str, str] | None:
    found = years(statement)
    if not found:
        return None
    old = found[0]
    new = pools.other(pools.years, old)
    if new is None:
        return None
    return statement.replace(old, new, 1), old, new


def gen_numeral(statement: str, pools: ValuePools) -> tuple[str, str, str] | None:
    ys = set(years(statement))
    nums = [n for n in _NUMBER_RE.findall(statement) if n not in ys]
    if not nums:
        return None
    old = nums[0]
    new = pools.other(pools.numbers, old)
    if new is None:
        return None
    return statement.replace(old, new, 1), old, new


def gen_unit(statement: str, pools: ValuePools) -> tuple[str, str, str] | None:
    toks = tokenize(statement)
    present = [t for t in toks if t.lower() in UNITS]
    if not present:
        return None
    old = present[0]
    new = pools.other(pools.units, old.lower())
    if new is None:
        return None
    return re.sub(rf"\b{re.escape(old)}\b", new, statement, count=1), old, new


def gen_entity(statement: str, pools: ValuePools) -> tuple[str, str, str] | None:
    """Swap a named entity, or — when the rule-based proxy finds none — a
    salient content term.

    The fallback is not a weakening.  The canonical contradiction in PLAN.md
    §10.1 is অক্সিজেন for কার্বন ডাই-অক্সাইড, and neither is a proper noun, so
    a generator restricted to entities would never produce the pair the veto
    layer most needs to be trained against.
    """
    ents = sorted(entities(statement))
    if ents:
        old = ents[0]
        new = pools.other(pools.entities, old)
        if new is not None:
            return statement.replace(old, new, 1), old, new

    from ..preprocess.stopwords import content_tokens

    terms = [t for t in content_tokens(tokenize(statement)) if len(t) > 3]
    if not terms:
        return None
    old = pools.rng.choice(terms)
    new = pools.other(pools.terms, old)
    if new is None or new in statement:
        return None
    import re as _re

    swapped = _re.sub(rf"\b{_re.escape(old)}\b", new, statement, count=1)
    return (swapped, old, new) if swapped != statement else None


def gen_polarity(statement: str, pools: ValuePools) -> tuple[str, str, str] | None:
    """Insert or delete a negation — the smallest possible meaning reversal."""
    toks = tokenize(statement)
    present = [t for t in toks if t in NEGATION]
    if present:
        old = present[0]
        return re.sub(rf"\s*\b{re.escape(old)}\b", "", statement, count=1).strip(), old, ""
    if len(toks) < 3:
        return None
    # Bangla negation is clause-final, so appending before the dari is the
    # grammatical placement rather than a token dropped anywhere.
    body = statement.rstrip("।").rstrip()
    return f"{body} না।" if statement.rstrip().endswith("।") else f"{body} না", "", "না"


def gen_relation_order(statement: str, pools: ValuePools) -> tuple[str, str, str] | None:
    """Reverse a directional relation.  Identical token set, opposite truth."""
    toks = tokenize(normalize(statement))
    for i, tok in enumerate(toks):
        if tok in ("থেকে", "হতে") and 0 < i < len(toks) - 1:
            left, right = toks[i - 1], toks[i + 1]
            if left == right:
                continue
            swapped = re.sub(rf"\b{re.escape(left)}\b\s+{re.escape(tok)}\s+\b{re.escape(right)}\b",
                             f"{right} {tok} {left}", statement, count=1)
            if swapped != statement:
                return swapped, f"{left}->{right}", f"{right}->{left}"
    return None


GENERATOR_FNS = {
    "date": gen_date,
    "numeral": gen_numeral,
    "unit": gen_unit,
    "entity": gen_entity,
    "polarity": gen_polarity,
    "relation_order": gen_relation_order,
}


# --------------------------------------------------------------------------- #
# Construction                                                                 #
# --------------------------------------------------------------------------- #


def _evidence_sentence(context: str, start: int, end: int) -> tuple[str, int, int] | None:
    """The sentence of ``context`` containing ``[start, end)``."""
    for sent, s, e in sentence_spans(context):
        if s <= start and end <= e:
            return sent, s, e
    return None


MAX_EVIDENCE_CHARS = 1_200


def _evidence(passage: str) -> str:
    """What the claim is judged *against* — the passage, not the sentence.

    **This is the fix for a construction flaw that made S3 meaningless.**  The
    first version paired each SUPPORTED statement with the very sentence it was
    copied from, so 6,950 of 20,429 training pairs were byte-identical strings
    and the classifier scored 1.0000 on val *and* on the held-out contrast set
    by learning ``statement == evidence``.  A perfect score on a contrast set
    designed to be hard is a bug report, not a result: it meant S3 had learned
    string equality, which the pipeline never presents it with, because there
    the statement is an extracted span and the evidence is a retrieved passage.

    Pairing the sentence against its **containing passage** restores the task
    §10.3 actually describes: SUPPORTED is verbatim-contained, CONTRADICTED
    differs from something in the passage by exactly one token, and NEUTRAL is
    a claim the passage does not discuss.  The SUPPORTED/CONTRADICTED boundary
    is then the minimal-pair signal the model is supposed to learn.
    """
    return passage[:MAX_EVIDENCE_CHARS]


def build(limit: int | None = None) -> dict:
    set_seed()
    rng = random.Random(CFG.seed)
    print("T11b  BanglaVerify construction")

    qa = load_jsonl(QA_TRAIN)
    contexts = {r["pid"]: r["text"] for r in load_jsonl(PROCESSED / "gold_contexts.jsonl")}
    if limit:
        qa = qa[:limit]

    # ---- SUPPORTED -------------------------------------------------------
    supported: list[Item] = []
    for row in qa:
        if not row.get("is_answerable"):
            continue
        context = contexts.get(row["gold_passage_id"])
        if not context:
            continue
        for ans in row.get("answers", []):
            if not ans.get("aligned") or ans.get("answer_type") == "yes/no":
                continue
            span = (ans.get("spans") or [{}])[0]
            if span.get("start") is None:
                continue
            found = _evidence_sentence(context, span["start"], span["end"])
            if not found:
                continue
            sent = found[0]
            supported.append(Item(evidence=_evidence(context), statement=sent,
                                  label=SUPPORTED, generator="gold", qid=row["qid"],
                                  pid=row["gold_passage_id"]))
            break
    print(f"  SUPPORTED   {len(supported):,d} from aligned gold spans")

    pools = ValuePools([it.statement for it in supported])
    print(f"  value pools: {len(pools.years):,d} years, {len(pools.numbers):,d} numbers, "
          f"{len(pools.units):,d} units, {len(pools.entities):,d} entities")

    # ---- CONTRADICTED ----------------------------------------------------
    contradicted: list[Item] = []
    attempts: Counter = Counter()
    for it in supported:
        order = list(GENERATORS)
        rng.shuffle(order)
        for gname in order:
            attempts[gname] += 1
            made = GENERATOR_FNS[gname](it.statement, pools)
            if made is None:
                continue
            new_statement, old, new = made
            if normalize(new_statement) == normalize(it.statement):
                continue
            contradicted.append(Item(evidence=it.evidence, statement=new_statement,
                                     label=CONTRADICTED, generator=gname, qid=it.qid,
                                     pid=it.pid, swapped_from=old, swapped_to=new,
                                     source_statement=it.statement))
            break
    by_gen = Counter(i.generator for i in contradicted)
    print(f"  CONTRADICTED {len(contradicted):,d} minimal pairs  {dict(by_gen)}")

    # ---- NEUTRAL ---------------------------------------------------------
    # A topically similar passage that does *not* contain the answer.  Drawn
    # from another question's evidence rather than at random, so the pair stays
    # in-domain and the model cannot separate it on vocabulary alone.
    neutral: list[Item] = []
    pool = [it for it in supported]
    for i, it in enumerate(supported):
        other = pool[(i + len(pool) // 2) % len(pool)]
        if other.pid == it.pid:
            continue
        neutral.append(Item(evidence=other.evidence, statement=it.statement,
                            label=NEUTRAL, generator="cross_passage", qid=it.qid,
                            pid=other.pid))
        # (evidence is another passage, so the claim is one this passage does
        #  not discuss — topically close, never contradicted)
    print(f"  NEUTRAL     {len(neutral):,d} cross-passage pairs")

    items = supported + contradicted + neutral
    rng.shuffle(items)

    # ---- the held-out contrast set --------------------------------------
    # Stratified over generators so every veto category is represented, and
    # removed from the pool *before* the train/val split so it can never leak.
    contrast = _stratified_contrast(items, CFG.contrast_set_size, rng)
    contrast_keys = {(i.qid, i.label, i.statement) for i in contrast}
    rest = [i for i in items if (i.qid, i.label, i.statement) not in contrast_keys]

    n_val = max(int(0.1 * len(rest)), 1)
    val, train = rest[:n_val], rest[n_val:]

    write_jsonl(OUT_TRAIN, (i.as_dict() for i in train))
    write_jsonl(OUT_VAL, (i.as_dict() for i in val))
    write_jsonl(OUT_CONTRAST, (i.as_dict() for i in contrast))
    print(f"  split: train {len(train):,d} · val {len(val):,d} · "
          f"contrast {len(contrast):,d} (held out, never trained on)")

    # ---- the hand-verification sample -----------------------------------
    sample = _stratified_sample(items, CFG.banglaverify_verify_sample, rng)
    save_table("banglaverify_sample", [
        {"i": n, "label": i.label, "generator": i.generator,
         "swapped_from": i.swapped_from, "swapped_to": i.swapped_to,
         "statement": i.statement, "evidence": i.evidence,
         "label_correct": ""}   # <- the column a human fills in
        for n, i in enumerate(sample, start=1)])
    print(f"  hand-verification sample -> reports/tables/banglaverify_sample.csv "
          f"({len(sample)} items, 'label_correct' left blank for the annotator)")

    quality = auto_quality(contradicted)
    save_table("banglaverify_quality", quality)
    for row in quality:
        print(f"    {row['generator']:16s} n={row['n']:5,d}  "
              f"veto detects {row['veto_precision']:.3f}")

    payload = {
        "counts": {SUPPORTED: len(supported), CONTRADICTED: len(contradicted),
                   NEUTRAL: len(neutral)},
        "by_generator": dict(by_gen),
        "generator_attempts": dict(attempts),
        "splits": {"train": len(train), "val": len(val), "contrast": len(contrast)},
        "auto_quality": quality,
        "hand_sample": len(sample),
        "license": "CC BY-NC-SA 4.0 (derivative of BanglaRQA)",
    }
    log_result("t11b_banglaverify", "construction", payload)
    return payload


def _stratified_contrast(items: Sequence[Item], n: int, rng: random.Random) -> list[Item]:
    """Equal-ish coverage of every label and every generator."""
    buckets: defaultdict[str, list[Item]] = defaultdict(list)
    for it in items:
        buckets[f"{it.label}/{it.generator}"].append(it)
    keys = sorted(buckets)
    per = max(n // max(len(keys), 1), 1)
    out: list[Item] = []
    for k in keys:
        group = buckets[k]
        rng.shuffle(group)
        out.extend(group[:per])
    rng.shuffle(out)
    return out[:n]


def _stratified_sample(items: Sequence[Item], n: int, rng: random.Random) -> list[Item]:
    return _stratified_contrast(items, n, rng)


def auto_quality(contradicted: Sequence[Item]) -> list[dict]:
    """What fraction of each generator's output the **veto layer** detects.

    This is an automatic lower bound on label precision, not a substitute for
    the 300-item hand check.  It answers a narrower question — "is this
    perturbation detectable by the mechanism it was built to exercise?" — and a
    generator that scores badly here is producing pairs that are not
    contradictions at all, which is worth catching before a human spends 50
    minutes on them.
    """
    from .constraints import check

    by_gen: defaultdict[str, list[Item]] = defaultdict(list)
    for it in contradicted:
        by_gen[it.generator].append(it)

    rows: list[dict] = []
    for gen in GENERATORS:
        group = by_gen.get(gen, [])
        if not group:
            continue
        hits = sum(1 for it in group
                   if check(it.statement, it.source_statement or it.evidence).fired)
        rows.append({"generator": gen, "n": len(group),
                     "veto_detected": hits,
                     "veto_precision": round(hits / len(group), 4),
                     "meets_90pct_floor": hits / len(group) >= 0.90})
    return rows


def load(split: str = "train") -> list[dict]:
    path = {"train": OUT_TRAIN, "val": OUT_VAL, "contrast": OUT_CONTRAST}[split]
    return load_jsonl(path)


if __name__ == "__main__":
    build()
