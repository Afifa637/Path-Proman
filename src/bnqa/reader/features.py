"""Span features (PLAN.md §7.6, task T10).

Six groups, every one of them nameable in a viva and printable as a bar chart
in the UI.  **The feature-importance table is a deliverable, not a by-product**
(``reports/tables/reader_features.csv``): "why did the system pick this span?"
has to be answerable, and when Tier B's transformer arrives this is the
baseline it must beat — which turns its win or loss into a number instead of an
assumption.

One group deserves its name explained.  The **negative** features encode the
two mistakes a lexical span ranker makes constantly: echoing a question term
back as the answer ("কোন গ্যাস...?" -> "গ্যাস"), and selecting a well-positioned
run of function words.  Both score *well* on overlap, so they have to be
represented explicitly or the ranker learns them as positives.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

from ..preprocess.normalize import BENGALI_BLOCK
from ..preprocess.stopwords import STOPWORDS, content_tokens
from ..preprocess.tokenize import tokenize_lower
from ..verify.constraints import UNITS, years
from .base import Candidate
from .candidates import IdfTable, trigram_cosine
from .qtype import NUMBER, PERSON, PLACE, PROCESS, REASON, THING, TIME, span_matches_class

_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"[0-9]")
_BENGALI_RE = re.compile(f"[{BENGALI_BLOCK}]")

FEATURE_NAMES: tuple[str, ...] = (
    # lexical
    "sent_idf_overlap", "sent_overlap_frac", "q_content_matched_frac", "min_dist_to_match",
    # span-internal
    "span_tokens", "span_chars", "span_idf_mean", "span_idf_max",
    "is_numeral", "is_year", "has_latin", "has_digit", "in_parentheses", "is_unit_bearing",
    # type match
    "class_match", "class_is_time", "class_is_number", "class_is_person",
    "class_is_place", "class_is_clause", "type_prior_len",
    # similarity
    "cos_q_span", "cos_q_sentence",
    # structural
    "passage_rank_inv", "passage_score", "sentence_rank_inv",
    "span_pos_frac", "relation_order_flag",
    # negative
    "span_in_question", "stopword_frac", "starts_with_stopword",
)


def extract(question: str, cand: Candidate, *, idf: IdfTable, expected_class: str,
            question_type: str = "factoid") -> dict[str, float]:
    """The feature vector for one candidate."""
    q_tokens = tokenize_lower(question)
    q_content = set(content_tokens(q_tokens))
    q_all = set(q_tokens)
    s_tokens = tokenize_lower(cand.sentence)
    span_tokens = tokenize_lower(cand.text)

    # ---- lexical --------------------------------------------------------
    matched = q_content & set(s_tokens)
    denom = idf.weight(sorted(q_content)) or 1.0
    sent_idf_overlap = idf.weight(sorted(matched)) / denom
    sent_overlap_frac = len(matched) / max(len(set(s_tokens)), 1)
    q_matched_frac = len(matched) / max(len(q_content), 1)
    min_dist = _min_distance(s_tokens, span_tokens, matched)

    # ---- span-internal --------------------------------------------------
    span_idfs = [idf(t) for t in span_tokens] or [0.0]
    is_year = bool(years(cand.text))
    is_numeral = bool(span_tokens) and all(t.replace(".", "").replace(",", "").isdigit()
                                           for t in span_tokens)
    before = cand.sentence[:max(cand.char_start - cand.sentence_start, 0)]
    in_paren = before.count("(") > before.count(")")

    # ---- type match -----------------------------------------------------
    class_match = float(span_matches_class(cand.text, expected_class))

    # ---- structural -----------------------------------------------------
    sent_len = max(len(s_tokens), 1)
    pos = (cand.char_start - cand.sentence_start) / max(len(cand.sentence), 1)

    feats = {
        "sent_idf_overlap": sent_idf_overlap,
        "sent_overlap_frac": sent_overlap_frac,
        "q_content_matched_frac": q_matched_frac,
        "min_dist_to_match": min_dist,

        "span_tokens": float(len(span_tokens)),
        "span_chars": float(len(cand.text)),
        "span_idf_mean": sum(span_idfs) / len(span_idfs),
        "span_idf_max": max(span_idfs),
        "is_numeral": float(is_numeral),
        "is_year": float(is_year),
        "has_latin": float(bool(_LATIN_RE.search(cand.text))),
        "has_digit": float(bool(_DIGIT_RE.search(cand.text))),
        "in_parentheses": float(in_paren),
        "is_unit_bearing": float(any(t in UNITS for t in span_tokens)),

        "class_match": class_match,
        "class_is_time": float(expected_class == TIME),
        "class_is_number": float(expected_class == NUMBER),
        "class_is_person": float(expected_class == PERSON),
        "class_is_place": float(expected_class == PLACE),
        "class_is_clause": float(expected_class in (REASON, PROCESS)),
        # Answer-type prior given the question type: causal and list answers are
        # long, factoid answers are short.  Encoded as the length the type
        # expects, so the ranker can penalise deviation rather than length.
        "type_prior_len": _type_prior_len(question_type, expected_class, len(span_tokens)),

        "cos_q_span": trigram_cosine(question, cand.text),
        "cos_q_sentence": trigram_cosine(question, cand.sentence),

        "passage_rank_inv": 1.0 / (1 + cand.passage_rank),
        "passage_score": float(cand.passage_score),
        "sentence_rank_inv": 1.0 / (1 + cand.sentence_rank),
        "span_pos_frac": pos,
        "relation_order_flag": float(any(t in ("থেকে", "হতে", "চেয়ে") for t in s_tokens)),

        # ---- negative ----
        "span_in_question": float(bool(span_tokens) and set(span_tokens) <= q_all),
        "stopword_frac": (sum(1 for t in span_tokens if t in STOPWORDS)
                          / max(len(span_tokens), 1)),
        "starts_with_stopword": float(bool(span_tokens) and span_tokens[0] in STOPWORDS),
    }
    return feats


def _min_distance(sent_tokens: Sequence[str], span_tokens: Sequence[str],
                  matched: set[str]) -> float:
    """Token distance from the span to the nearest matched question term.

    Normalised into ``[0, 1]`` by sentence length and inverted, so "close to
    the evidence" is a large value and an absent match is 0 rather than a
    sentinel the model has to learn around.
    """
    if not matched or not span_tokens:
        return 0.0
    try:
        start = sent_tokens.index(span_tokens[0])
    except ValueError:
        return 0.0
    best = None
    for i, tok in enumerate(sent_tokens):
        if tok in matched:
            d = abs(i - start)
            if best is None or d < best:
                best = d
    if best is None:
        return 0.0
    return 1.0 / (1.0 + best)


def _type_prior_len(question_type: str, expected_class: str, n_tokens: int) -> float:
    """How well the span's length fits what this question type usually answers."""
    expect = {
        "confirmation": 1.0, "factoid": 2.5, "list": 3.0, "causal": 8.0,
    }.get(question_type, 2.5)
    if expected_class in (REASON, PROCESS):
        expect = max(expect, 8.0)
    elif expected_class in (TIME, NUMBER):
        expect = 1.5
    elif expected_class in (PERSON, PLACE, THING):
        expect = 2.5
    return math.exp(-abs(n_tokens - expect) / max(expect, 1.0))


def to_matrix(feature_dicts: Sequence[dict[str, float]]):
    """Stack feature dicts into the ``(n, d)`` array the models consume."""
    import numpy as np

    return np.asarray([[d.get(name, 0.0) for name in FEATURE_NAMES]
                       for d in feature_dicts], dtype=np.float32)
