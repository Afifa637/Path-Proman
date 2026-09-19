"""The answer-equivalence metric ladder (PLAN.md §9, tasks T6b and T10).

A correct answer phrased differently from the gold string is scored **wrong** by
Exact Match, and in Bangla that is severe for three compounding reasons —
inflection (উদ্ভিদ / উদ্ভিদের / উদ্ভিদগুলো), synonymy (বায়ুমণ্ডল / আবহমণ্ডল) and
script variation (CO2 / কার্বন ডাই-অক্সাইড).  So four tiers are reported, never
one:

==== ====================================== =========
tier metric                                 available
==== ====================================== =========
1    strict EM — the honest floor           A
2    **token F1** — THE HEADLINE NUMBER     A
3    stem + synonym F1                      A
4    soft-embedding F1 + BERTScore          B (X4)
==== ====================================== =========

**The two guards from §9 are enforced in code, not in prose.**

*Guard 1 — tiers 3+ are gated by the veto layer.*  A paraphrase and a
contradiction look nearly identical to any soft metric, so ``tier3_f1`` calls
:func:`bnqa.verify.constraints.check` first and returns the tier-2 score
unchanged when a veto fires.  Granting synonym credit without that gate makes
evaluation *worse*, not better, and it does so invisibly.

*Guard 2 — no tier is ever quoted alone.*  :func:`score_all` returns every
available tier together, which is what the tables print.  Softer tiers are
labelled upper bounds in the report, and the headline stays tier 2.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from ..preprocess.normalize import normalize, normalize_for_match
from ..preprocess.stem import stem
from ..preprocess.tokenize import tokenize_lower
from ..resources.loader import load_synonyms, load_variants
from ..verify.constraints import check as veto_check

HEADLINE_TIER = "tier2_token_f1"


# --------------------------------------------------------------------------- #
# Tier 1 — strict exact match                                                  #
# --------------------------------------------------------------------------- #


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_for_match(pred) == normalize_for_match(gold))


# --------------------------------------------------------------------------- #
# Tier 2 — token F1 after Bangla normalisation (the headline)                  #
# --------------------------------------------------------------------------- #


def _f1_from_counts(pred_toks: Sequence[str], gold_toks: Sequence[str]) -> float:
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def token_f1(pred: str, gold: str) -> float:
    return _f1_from_counts(tokenize_lower(pred), tokenize_lower(gold))


# --------------------------------------------------------------------------- #
# Tier 3 — stem + synonym F1, gated by the veto layer                          #
# --------------------------------------------------------------------------- #


def _canonical(tokens: Iterable[str]) -> list[str]:
    """Map each token to a class representative: variant -> synonym -> stem."""
    lex = load_synonyms()
    variants = load_variants()
    out: list[str] = []
    for tok in tokens:
        forms = {tok} | variants.get(tok, set())
        rep = min(forms)
        cls = lex.expand(rep)
        if len(cls) > 1:
            rep = min(cls)
        out.append(stem(rep))
    return out


def tier3_f1(pred: str, gold: str, *, evidence: str | None = None) -> float:
    """Stem + synonym F1.  **Never exceeds tier 2 when a veto fires.**

    ``evidence`` is the supporting sentence.  When it is supplied and the veto
    layer finds a contradiction between the prediction and that evidence, the
    soft credit is withheld: the prediction is not a paraphrase of the gold, it
    is a statement the corpus contradicts, and tier 3 must not reward it.
    """
    base = token_f1(pred, gold)
    if evidence is not None and veto_check(pred, evidence).fired:
        return base
    soft = _f1_from_counts(_canonical(tokenize_lower(pred)),
                           _canonical(tokenize_lower(gold)))
    # Soft tiers are upper bounds on tier 2, never corrections downward.
    return max(base, soft)


# --------------------------------------------------------------------------- #
# Multi-answer and multi-span handling                                         #
# --------------------------------------------------------------------------- #


def best_over_golds(fn, pred: str, golds: Sequence[str], **kwargs) -> float:
    """BanglaRQA ships several acceptable answers; the best one counts."""
    if not golds:
        return 0.0
    return max(fn(pred, g, **kwargs) for g in golds)


def set_f1(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Set-level F1 for ``multiple spans`` answers (PLAN.md §7.6).

    Greedy one-to-one matching on token F1: each predicted span is paired with
    its best unused gold, so a system is neither rewarded for repeating one
    correct span nor punished for ordering.
    """
    if not preds or not golds:
        return float(not preds and not golds)
    remaining = list(golds)
    total = 0.0
    for p in preds:
        if not remaining:
            break
        scores = [token_f1(p, g) for g in remaining]
        best = max(range(len(scores)), key=lambda i: scores[i])
        total += scores[best]
        remaining.pop(best)
    precision = total / len(preds)
    recall = total / len(golds)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


# --------------------------------------------------------------------------- #
# The full ladder                                                              #
# --------------------------------------------------------------------------- #


def score_all(pred: str, golds: Sequence[str], *, evidence: str | None = None) -> dict:
    """Every available tier at once — guard 2, enforced by the return type."""
    pred = normalize(pred or "")
    golds = [normalize(g or "") for g in golds]
    return {
        "tier1_em": best_over_golds(exact_match, pred, golds),
        "tier2_token_f1": best_over_golds(token_f1, pred, golds),
        "tier3_stem_syn_f1": best_over_golds(tier3_f1, pred, golds, evidence=evidence),
    }


def aggregate(rows: Sequence[dict]) -> dict:
    """Mean of each tier, plus HasAns / NoAns F1 (T10's acceptance criterion).

    ``rows`` carry ``is_answerable``, ``abstained`` and the per-item tiers.
    Abstention is scored as it is *used*: correct on an unanswerable question,
    a zero on an answerable one.  Reporting one pooled F1 would let a system
    that abstains on everything look respectable.
    """
    if not rows:
        return {}
    tiers = ("tier1_em", "tier2_token_f1", "tier3_stem_syn_f1")
    out: dict = {"n": len(rows)}
    for t in tiers:
        out[t] = sum(r.get(t, 0.0) for r in rows) / len(rows)

    has = [r for r in rows if r.get("is_answerable")]
    no = [r for r in rows if not r.get("is_answerable")]
    out["n_hasans"] = len(has)
    out["n_noans"] = len(no)
    if has:
        out["hasans_f1"] = sum(r.get("tier2_token_f1", 0.0) for r in has) / len(has)
    if no:
        # For an unanswerable question the only correct behaviour is abstention.
        out["noans_f1"] = sum(1.0 for r in no if r.get("abstained")) / len(no)
    out["abstention_rate"] = sum(1.0 for r in rows if r.get("abstained")) / len(rows)
    return out
