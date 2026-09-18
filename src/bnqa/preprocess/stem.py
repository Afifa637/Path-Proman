"""Rule-based Bangla suffix stripper (PLAN.md §7.2, task T4).

**Applied to sparse models only.**  Dense models keep the full surface form.
The asymmetry is deliberate and reported (PLAN.md §7.2): stemming helps lexical
matching and hurts distributional representations, so applying it to both would
contaminate RQ1 — the comparison would be measuring preprocessing rather than
the retriever.

The suffix inventory lives in ``resources/suffixes.txt`` (T6b) because Tier B's
MorphSpan-MLM needs the same stem/suffix boundary.

Why this is not just "strip the longest suffix"
-----------------------------------------------
Bangla suffixation is genuinely ambiguous without a lexicon.  ``উদ্ভিদের``
("of the plant") is ``উদ্ভিদ`` + ``ের``, but it also *ends with* the plural
genitive ``দের``, and longest-match-first therefore yields ``উদ্ভি`` — which is
not a word, and worse, does **not** equal ``stem("উদ্ভিদ") = উদ্ভিদ``.  The point
of stemming here is to make inflected forms **converge**; a stemmer that splits
them is worse than none.

So the stemmer is **corpus-validated**.  It proposes every candidate strip and
accepts the longest one whose result actually occurs as a standalone token in
our own corpus (installed by :func:`set_vocabulary` once the index is built in
T5).  ``উদ্ভি`` never occurs; ``উদ্ভিদ`` occurs constantly, so the right split
wins on evidence rather than on a rule.  With no vocabulary installed the
stemmer falls back to a conservative tier order — case suffixes before plural
ones — which is right far more often than longest-match.

No external lexicon is involved, so Tier A's "nothing downloaded" claim
(PLAN.md §0.2) still holds: the evidence is our own corpus.
"""

from __future__ import annotations

from pathlib import Path

from ..config import RESOURCES

HASANT = "্"  # ্  — a stem may never end here
MIN_STEM_CHARS = 3
MAX_STRIPS = 2  # উদ্ভিদগুলোর -> উদ্ভিদগুলো -> উদ্ভিদ

# Tier order used when no corpus vocabulary is installed.  Case suffixes are the
# least ambiguous, so they are tried first.
TIER_ORDER = ("case", "classifier", "plural", "adverbial")

_suffix_tiers: dict[str, tuple[str, ...]] | None = None
_vocabulary: frozenset[str] | None = None
_cache: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Resource loading                                                             #
# --------------------------------------------------------------------------- #


def load_suffix_tiers(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Parse ``resources/suffixes.txt`` into tiers keyed by its section headers."""
    global _suffix_tiers
    if _suffix_tiers is not None and path is None:
        return _suffix_tiers

    path = path or RESOURCES / "suffixes.txt"
    tiers: dict[str, list[str]] = {t: [] for t in TIER_ORDER}
    current = "case"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                low = line.lower()
                if "plural" in low or "collective" in low:
                    current = "plural"
                elif "classifier" in low:
                    current = "classifier"
                elif "case" in low or "postposition" in low:
                    current = "case"
                elif "adverbial" in low or "derivational" in low:
                    current = "adverbial"
                continue
            tiers[current].append(line)

    out = {t: tuple(sorted(set(v), key=len, reverse=True)) for t, v in tiers.items()}
    if path == RESOURCES / "suffixes.txt":
        _suffix_tiers = out
    return out


def load_suffixes(path: Path | None = None) -> tuple[str, ...]:
    """All suffixes, longest first (used by Tier B's MorphSpan masking)."""
    tiers = load_suffix_tiers(path)
    flat = [s for tier in TIER_ORDER for s in tiers[tier]]
    return tuple(sorted(set(flat), key=len, reverse=True))


def set_vocabulary(vocab: set[str] | frozenset[str] | None) -> None:
    """Install the corpus vocabulary that validates candidate stems (T5)."""
    global _vocabulary
    _vocabulary = frozenset(vocab) if vocab else None
    _cache.clear()


def vocabulary_size() -> int:
    return len(_vocabulary) if _vocabulary else 0


# --------------------------------------------------------------------------- #
# Stemming                                                                     #
# --------------------------------------------------------------------------- #


def _valid(candidate: str) -> bool:
    return len(candidate) >= MIN_STEM_CHARS and not candidate.endswith(HASANT)


def _strip_once(word: str) -> str:
    """One strip.  Corpus-validated when a vocabulary is installed."""
    tiers = load_suffix_tiers()

    if _vocabulary is not None:
        # Longest strip whose result is an attested standalone token wins.
        best: str | None = None
        best_len = -1
        for tier in TIER_ORDER:
            for suf in tiers[tier]:
                if len(word) > len(suf) and word.endswith(suf):
                    cand = word[: -len(suf)]
                    if _valid(cand) and cand in _vocabulary and len(suf) > best_len:
                        best, best_len = cand, len(suf)
        return best if best is not None else word

    # No vocabulary: conservative tier order, longest within a tier.
    for tier in TIER_ORDER:
        for suf in tiers[tier]:
            if len(word) > len(suf) and word.endswith(suf):
                cand = word[: -len(suf)]
                if _valid(cand):
                    return cand
    return word


def stem(word: str) -> str:
    """Strip up to :data:`MAX_STRIPS` suffixes."""
    cached = _cache.get(word)
    if cached is not None:
        return cached

    out = word
    for _ in range(MAX_STRIPS):
        nxt = _strip_once(out)
        if nxt == out:
            break
        out = nxt

    if len(_cache) < 400_000:
        _cache[word] = out
    return out


def stem_tokens(tokens: list[str]) -> list[str]:
    return [stem(t) for t in tokens]


def same_stem(a: str, b: str) -> bool:
    return stem(a) == stem(b)
