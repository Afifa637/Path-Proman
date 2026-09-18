"""Rule-based word tokenisation and char n-grams (PLAN.md §7.2, task T4).

Tier A needs **no learned tokenizer**: TF-IDF and BM25 consume words and
character n-grams directly.  Tier B's SentencePiece model (X2) replaces this
behind the same call signature, which is what makes the swap an ablation rather
than a rewrite.

A token is a maximal run of one of three kinds:

* Bengali letters and marks — ``উদ্ভিদের``
* Latin letters optionally followed by digits — ``CO2``, ``pH``
* digits, with internal separators — ``1971``, ``3.14``, ``২৫-৩০`` (after the
  digit folding in :mod:`bnqa.preprocess.normalize`)
"""

from __future__ import annotations

import re
from functools import lru_cache

from .normalize import BENGALI_BLOCK, normalize

TOKEN_RE = re.compile(
    f"[{BENGALI_BLOCK}]+"          # Bangla word
    r"|[A-Za-z]+[0-9]*"            # Latin word, optionally with trailing digits
    r"|[0-9]+(?:[.,:/][0-9]+)*"    # number
)

SENT_SPLIT_RE = re.compile(r"(?<=[।?!])\s+|\n+")


def tokenize(text: str, *, do_normalize: bool = True) -> list[str]:
    """Word tokens of ``text``."""
    if do_normalize:
        text = normalize(text)
    return TOKEN_RE.findall(text)


def tokenize_lower(text: str, *, do_normalize: bool = True) -> list[str]:
    """Tokens with Latin case folded — the form the sparse retrievers index."""
    return [t.lower() for t in tokenize(text, do_normalize=do_normalize)]


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """``(token, char_start, char_end)`` over **already-normalised** text.

    The reader needs these to turn a chosen token span back into the character
    offsets that VC-1 prints and ``verify_receipt`` re-checks, so this must run
    on exactly the string stored in ``passages.jsonl``.
    """
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def char_ngrams(text: str, lo: int = 3, hi: int = 5, *, do_normalize: bool = True) -> list[str]:
    """Character n-grams over whitespace-padded words.

    Padding with a boundary marker keeps prefix/suffix n-grams distinguishable,
    which is what lets the char vectoriser absorb Bangla inflection: ``উদ্ভিদ``
    and ``উদ্ভিদের`` share every n-gram of the stem but differ at the tail.
    """
    toks = tokenize(text, do_normalize=do_normalize)
    out: list[str] = []
    for tok in toks:
        padded = f"<{tok}>"
        for n in range(lo, hi + 1):
            if len(padded) < n:
                continue
            out.extend(padded[i:i + n] for i in range(len(padded) - n + 1))
    return out


def sentences(text: str, *, min_chars: int = 2) -> list[str]:
    """Split on the dari, ``?``/``!`` and newlines.

    Deliberately simple: Bangla has no abbreviation-dot ambiguity of the English
    kind, because the sentence terminator ``।`` is not used inside words.
    """
    text = normalize(text)
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    return [p for p in parts if len(p) >= min_chars]


def sentence_spans(text: str, *, min_chars: int = 2) -> list[tuple[str, int, int]]:
    """Sentences with character offsets into ``text`` (assumed normalised).

    Used by the reader to highlight the supporting sentence *inside* its passage
    without ever re-deriving the passage text.
    """
    spans: list[tuple[str, int, int]] = []
    pos = 0
    for part in SENT_SPLIT_RE.split(text):
        if part is None:
            continue
        idx = text.find(part, pos) if part else -1
        if idx == -1:
            pos += len(part or "")
            continue
        stripped = part.strip()
        if len(stripped) >= min_chars:
            lead = len(part) - len(part.lstrip())
            start = idx + lead
            spans.append((stripped, start, start + len(stripped)))
        pos = idx + len(part)
    return spans


@lru_cache(maxsize=100_000)
def _cached_tokens(text: str) -> tuple[str, ...]:
    return tuple(tokenize_lower(text))


def cached_tokens(text: str) -> tuple[str, ...]:
    """Memoised tokenisation for the inner loops of retrieval evaluation."""
    return _cached_tokens(text)
