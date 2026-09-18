"""Bangla text normalisation (PLAN.md §7.2, task T4).

Five classes of difference make two "identical" Bangla strings compare unequal.
All five are handled here, and the module is deliberately the *only* place that
touches them, so the veto layer (T11) can rely on a single definition.

1. **Zero-width joiners.**  ZWJ ``\\u200d`` and ZWNJ ``\\u200c`` are invisible and
   scattered through web-scraped Bangla.  They are the single most common source
   of strings that look identical and hash differently.
2. **Nukta composition.**  ``ড় ঢ় য়`` each have two encodings: the precomposed
   character (U+09DC/U+09DD/U+09DF) and base + nukta (U+09BC).  Those three are
   Unicode *composition exclusions*, so NFC **decomposes** them — which is
   exactly what we want, because it makes both spellings converge.
3. **Digits.**  ``১৯৭১`` and ``1971`` are the same year.  Bengali digits are
   folded to ASCII so numeric veto terms compare by value.
4. **Punctuation.**  Curly quotes, en/em dashes, NBSP.
5. **Whitespace.**

``normalize`` is **idempotent** — ``normalize(normalize(x)) == normalize(x)`` —
and is asserted so in ``tests/test_normalize.py``, because everything downstream
(dedup keys, passage hashes in evidence receipts, span offsets) assumes it.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Character classes                                                            #
# --------------------------------------------------------------------------- #

BENGALI_BLOCK = "ঀ-৿"
BENGALI_DIGITS = "০১২৩৪৫৬৭৮৯"
ASCII_DIGITS = "0123456789"
DIGIT_MAP = str.maketrans(BENGALI_DIGITS, ASCII_DIGITS)
DIGIT_MAP_REVERSE = str.maketrans(ASCII_DIGITS, BENGALI_DIGITS)

DARI = "।"        # ।  Devanagari danda, used as the Bangla full stop
DOUBLE_DARI = "॥"  # ॥

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "…": "...",
    DOUBLE_DARI: DARI,
}
_PUNCT_TRANS = {ord(k): v for k, v in _PUNCT_MAP.items()}

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([।,;:?!)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")

BENGALI_CHAR_RE = re.compile(f"[{BENGALI_BLOCK}]")
NON_SPACE_RE = re.compile(r"\S")


# --------------------------------------------------------------------------- #
# Core                                                                         #
# --------------------------------------------------------------------------- #


def normalize(text: str, *, fold_digits: bool = True) -> str:
    """Canonical form used everywhere in the project.

    ``fold_digits=False`` preserves Bengali numerals — used only when rendering
    text back to a human, never for matching.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(ZERO_WIDTH)
    text = text.translate(_PUNCT_TRANS)
    if fold_digits:
        text = text.translate(DIGIT_MAP)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    """Aggressive form for *matching only*: lowercase Latin, drop punctuation.

    Never used for anything the user sees and never used to compute a span
    offset — only to decide whether two strings denote the same thing.
    """
    text = normalize(text).lower()
    text = re.sub(f"[^{BENGALI_BLOCK}a-z0-9\\s]", " ", text)
    return _WS_RE.sub(" ", text).strip()


def collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text.replace("\n", " ")).strip()


# --------------------------------------------------------------------------- #
# Script statistics — used by the corpus filter (T3) and the audit             #
# --------------------------------------------------------------------------- #


def bengali_ratio(text: str) -> float:
    """Fraction of non-space characters that are in the Bengali block."""
    chars = NON_SPACE_RE.findall(text)
    if not chars:
        return 0.0
    bengali = sum(1 for c in chars if "ঀ" <= c <= "৿")
    return bengali / len(chars)


def to_bengali_digits(text: str) -> str:
    """For display only — the UI shows ১৯৭১, the index stores 1971."""
    return text.translate(DIGIT_MAP_REVERSE)
