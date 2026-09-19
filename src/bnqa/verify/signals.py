"""The five verification signals (PLAN.md §10.4, task T11).

Each signal is a function with a **fixed signature** — ``fn(SignalContext) ->
float`` in ``[0, 1]`` — which is the third of the three interfaces the two-tier
design rests on.  Tier B swaps S2's lexical grounding for FastText cosine
(X4), S3's GBDT for a cross-encoder (X10) and S5's bagged rankers for a 3-seed
neural ensemble (X9b) without anything downstream noticing.

=== ==================================== ===========================
id  what it measures                     Tier B replacement
=== ==================================== ===========================
S1  reader confidence (aleatoric)        neural span logits
S2  evidence grounding                   + FastText cosine
S3  support classifier                   cross-encoder (X10)
S4  multi-passage consistency            unchanged
S5  ensemble disagreement (epistemic)    3-seed neural ensemble (X9b)
=== ==================================== ===========================

**S2 is gated by the veto layer, and that gate is the whole point.**  Grounding
is measured *through* the T6b synonym and suffix resources so a correct
paraphrase is not judged ungrounded — but tolerant matching is exactly what
turns ১৯৫২ into ১৯৭১.  So :func:`s2_grounding` runs :func:`constraints.check`
first and returns a capped value when a contradiction fires, before any soft
matching happens.

S1 and S5 are deliberately both kept.  S1 is *aleatoric* — how peaked this
model's distribution is — and S5 is *epistemic* — how much models that saw
different data disagree.  They are different quantities and empirically
complementary; a system with only S1 is confidently wrong in exactly the cases
S5 catches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..preprocess.stem import stem
from ..preprocess.stopwords import content_tokens
from ..preprocess.tokenize import tokenize_lower
from ..reader.base import Answer
from ..reader.candidates import trigram_cosine
from ..resources.loader import expand_term
from .constraints import VETO_SUPPORT_CAP, check as veto_check

SIGNAL_NAMES: tuple[str, ...] = ("s1_reader", "s2_grounding", "s3_support",
                                 "s4_consistency", "s5_ensemble")


@dataclass
class SignalContext:
    """Everything any signal is allowed to see.  Fixed, so the arms stay swappable."""

    question: str
    answer: Answer
    evidence: str
    passages: Sequence[dict] = field(default_factory=list)
    alt_answers: Sequence[Answer] = field(default_factory=list)   # S4: top-k rereads
    ensemble_answers: Sequence[Answer] = field(default_factory=list)  # S5: bagged seeds
    support_model: object | None = None                            # S3: fitted classifier
    gazetteer: frozenset[str] | None = None


# --------------------------------------------------------------------------- #
# S1 — reader confidence                                                       #
# --------------------------------------------------------------------------- #


def s1_reader(ctx: SignalContext) -> float:
    """Normalised span score, best-vs-second margin, null margin, passage score.

    The reader already softmaxes within the question, so ``score`` is
    comparable across questions; the margin is added because a peaked
    distribution over two near-identical spans is not the same kind of
    confidence as a peaked distribution over one.
    """
    a = ctx.answer
    if not a.text:
        return 0.0
    conf = 0.55 * a.score + 0.30 * max(a.margin, 0.0) + 0.15 * _squash(a.null_margin)
    return _clip(conf)


def s1_components(ctx: SignalContext) -> dict[str, float]:
    a = ctx.answer
    return {"s1_score": float(a.score), "s1_margin": float(a.margin),
            "s1_null_margin": float(a.null_margin),
            "s1_passage_score": float(a.passage_score),
            "s1_passage_rank_inv": 1.0 / (1 + a.passage_rank)}


# --------------------------------------------------------------------------- #
# S2 — evidence grounding, gated by the veto layer                             #
# --------------------------------------------------------------------------- #


def _expanded_forms(token: str) -> set[str]:
    forms = {token, stem(token)}
    for t in expand_term(token):
        forms.add(t.lower())
        forms.add(stem(t.lower()))
    return forms


def containment(answer: str, evidence: str) -> float:
    """Fraction of answer content tokens present in the evidence, through T6b."""
    a_toks = content_tokens(tokenize_lower(answer)) or tokenize_lower(answer)
    if not a_toks:
        return 0.0
    e_toks = set(tokenize_lower(evidence))
    e_stems = {stem(t) for t in e_toks}
    hits = 0
    for tok in a_toks:
        if tok in e_toks or stem(tok) in e_stems or (_expanded_forms(tok) & (e_toks | e_stems)):
            hits += 1
    return hits / len(a_toks)


def s2_grounding(ctx: SignalContext) -> float:
    """Token containment + char-3gram overlap + stem match. **Veto-gated.**"""
    if not ctx.answer.text:
        return 0.0
    veto = veto_check(ctx.answer.text, ctx.evidence, gazetteer=ctx.gazetteer)
    if veto.fired:
        # A contradiction is not a grounding failure to be averaged away; it is
        # a cap.  Soft matching never runs on a vetoed pair.
        return VETO_SUPPORT_CAP
    cont = containment(ctx.answer.text, ctx.evidence)
    cos = trigram_cosine(ctx.answer.text, ctx.evidence)
    literal = float(ctx.answer.text in ctx.evidence)
    return _clip(0.6 * cont + 0.2 * cos + 0.2 * literal)


def s2_components(ctx: SignalContext) -> dict[str, float]:
    veto = veto_check(ctx.answer.text, ctx.evidence, gazetteer=ctx.gazetteer)
    return {"s2_containment": containment(ctx.answer.text, ctx.evidence),
            "s2_trigram_cos": trigram_cosine(ctx.answer.text, ctx.evidence),
            "s2_literal": float(bool(ctx.answer.text) and ctx.answer.text in ctx.evidence),
            "s2_veto_fired": float(veto.fired)}


# --------------------------------------------------------------------------- #
# S3 — the support classifier (trained on BanglaVerify)                        #
# --------------------------------------------------------------------------- #


def s3_support(ctx: SignalContext) -> float:
    """P(SUPPORTED) from the BanglaVerify classifier, or a neutral 0.5."""
    model = ctx.support_model
    if model is None or not ctx.answer.text:
        return 0.5
    return float(model.predict_support(ctx.answer.text, ctx.evidence,
                                       question=ctx.question))


# --------------------------------------------------------------------------- #
# S4 — multi-passage consistency                                               #
# --------------------------------------------------------------------------- #


def s4_consistency(ctx: SignalContext) -> float:
    """Agreement between answers read independently from different passages.

    **Disagreement across independent passages is a strong unsupported
    signal** (§10.4): if the corpus really supports one answer, several
    passages about the topic should not point at different strings.
    """
    others = [a for a in ctx.alt_answers if a.pid != ctx.answer.pid and a.text]
    if not others or not ctx.answer.text:
        return 0.5
    sims = [_agreement(ctx.answer.text, o.text) for o in others]
    # Weighted towards the best agreement: one corroborating passage is
    # evidence, while a long tail of unrelated passages is not counter-evidence.
    best = max(sims)
    mean = sum(sims) / len(sims)
    return _clip(0.6 * best + 0.4 * mean)


def _agreement(a: str, b: str) -> float:
    a_n, b_n = a.strip().lower(), b.strip().lower()
    if a_n == b_n:
        return 1.0
    return trigram_cosine(a_n, b_n)


# --------------------------------------------------------------------------- #
# S5 — ensemble disagreement                                                   #
# --------------------------------------------------------------------------- #


def s5_ensemble(ctx: SignalContext) -> float:
    """Fraction of bagged rankers that chose the same span. Epistemic uncertainty."""
    if not ctx.ensemble_answers or not ctx.answer.text:
        return 0.5
    same = sum(1 for a in ctx.ensemble_answers
               if a.pid == ctx.answer.pid and a.char_start == ctx.answer.char_start)
    return same / len(ctx.ensemble_answers)


# --------------------------------------------------------------------------- #
# Registry and feature assembly                                                #
# --------------------------------------------------------------------------- #

SIGNALS: dict[str, Callable[[SignalContext], float]] = {
    "s1_reader": s1_reader,
    "s2_grounding": s2_grounding,
    "s3_support": s3_support,
    "s4_consistency": s4_consistency,
    "s5_ensemble": s5_ensemble,
}


def compute_all(ctx: SignalContext) -> dict[str, float]:
    """The five bars the UI draws, in a fixed order."""
    return {name: float(fn(ctx)) for name, fn in SIGNALS.items()}


def fusion_features(ctx: SignalContext) -> dict[str, float]:
    """Signals + their components + the veto flags — the input to ``fusion.py``."""
    from .constraints import veto_features

    feats = compute_all(ctx)
    feats.update(s1_components(ctx))
    feats.update(s2_components(ctx))
    feats.update(veto_features(ctx.answer.text, ctx.evidence, gazetteer=ctx.gazetteer))
    feats["answer_tokens"] = float(len(tokenize_lower(ctx.answer.text)))
    feats["evidence_tokens"] = float(len(tokenize_lower(ctx.evidence)))
    return feats


FUSION_FEATURE_NAMES: tuple[str, ...] = (
    "s1_reader", "s2_grounding", "s3_support", "s4_consistency", "s5_ensemble",
    "s1_score", "s1_margin", "s1_null_margin", "s1_passage_score", "s1_passage_rank_inv",
    "s2_containment", "s2_trigram_cos", "s2_literal", "s2_veto_fired",
    "veto_numeral", "veto_date", "veto_unit", "veto_entity", "veto_polarity",
    "veto_relation_order", "veto_any", "veto_count", "polarity_match",
    "year_in_evidence", "answer_has_year", "nums_in_evidence", "answer_has_number",
    "answer_tokens", "evidence_tokens",
)


def to_matrix(rows: Sequence[dict[str, float]]):
    import numpy as np

    return np.asarray([[r.get(n, 0.0) for n in FUSION_FEATURE_NAMES] for r in rows],
                      dtype=np.float32)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _squash(x: float) -> float:
    """Map a signed margin into ``[0, 1]`` without a hard cutoff."""
    return 0.5 * (1.0 + x / (1.0 + abs(x)))
