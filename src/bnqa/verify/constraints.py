"""The veto layer (PLAN.md §10.2, task T11) — the core contribution.

Every soft similarity metric scores these pairs at roughly 1.0:

===============================================  ==========================
evidence                                         candidate
===============================================  ==========================
উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে               উদ্ভিদ **অক্সিজেন** গ্রহণ করে
সালোকসংশ্লেষণ দিনে হয়                             সালোকসংশ্লেষণ দিনে হয় **না**
মুক্তিযুদ্ধ **১৯৭১** সালে                          মুক্তিযুদ্ধ **১৯৫২** সালে
===============================================  ==========================

All three are wrong, and each differs from the evidence by exactly one token.
So verification runs this layer **before** any similarity signal, and a
contradiction here **caps support regardless of similarity**.

Six categories, one per generator in BanglaVerify (§10.3): ``numeral``,
``date``, ``unit``, ``entity``, ``polarity``, ``relation_order``.

The interaction bug this module exists to prevent
-------------------------------------------------
Fuzzy matching — edit distance, char n-grams, transliteration — **must never
reach a veto term**.  Tolerant matching over numerals maps ১৯৫২ to ১৯৭১; over
entities it maps অক্সিজেন to অক্সাইড.  Either one silently destroys the only
layer that catches contradiction, and it does so while making every headline
score go *up*, which is why it would survive review.  Everything here therefore
runs on **normalised-but-unfuzzed** text, no function in this module imports a
similarity measure, and ``tests/test_veto_isolation.py`` asserts both.

What ``entity`` means here, stated plainly
------------------------------------------
There is no Bangla NER in Tier A and we do not pretend otherwise.  Entities are
approximated by a **rule-based proper-noun proxy**: Latin capitalised runs,
Bangla tokens adjacent to a class marker (নদী, জেলা, বিশ্ববিদ্যালয়, …), and
high-salience tokens carried by the caller's gazetteer.  It is a recall-limited
proxy, which is the safe direction: a missed entity costs a veto we could have
fired, while a false one would reject a correct answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..preprocess.normalize import normalize
from ..preprocess.tokenize import tokenize

# --------------------------------------------------------------------------- #
# Lexicons — small, hand-authored, and deliberately not downloaded             #
# --------------------------------------------------------------------------- #

NEGATION: frozenset[str] = frozenset({
    "না", "নয়", "নেই", "নাই", "নি", "ছাড়া", "বিনা", "ব্যতীত", "অনুপস্থিত",
    "কখনো", "কোনোদিন", "অসম্ভব",
})

# Units that appear in school science and social-science text.  A unit mismatch
# is a contradiction even when the number agrees: ৫ মিটার is not ৫ কিলোমিটার.
UNITS: frozenset[str] = frozenset({
    "মিটার", "কিলোমিটার", "সেন্টিমিটার", "মিলিমিটার", "গজ", "ফুট", "ইঞ্চি", "মাইল",
    "গ্রাম", "কিলোগ্রাম", "মিলিগ্রাম", "টন", "কেজি", "মণ", "পাউন্ড",
    "লিটার", "মিলিলিটার", "ঘনমিটার", "বর্গমিটার", "বর্গকিলোমিটার", "হেক্টর", "একর",
    "সেকেন্ড", "মিনিট", "ঘণ্টা", "দিন", "সপ্তাহ", "মাস", "বছর", "শতাব্দী", "যুগ",
    "ডিগ্রি", "সেলসিয়াস", "ফারেনহাইট", "কেলভিন",
    "ভোল্ট", "অ্যাম্পিয়ার", "ওয়াট", "জুল", "ক্যালরি", "নিউটন", "প্যাসকেল", "হার্জ",
    "শতাংশ", "ভাগ", "গুণ", "টাকা", "ডলার",
    "m", "km", "cm", "mm", "kg", "g", "mg", "l", "ml", "s", "hz", "kw", "mw",
})

# Markers that make the preceding or following token a named thing.
ENTITY_MARKERS: frozenset[str] = frozenset({
    "নদী", "সাগর", "মহাসাগর", "উপসাগর", "পর্বত", "পাহাড়", "দ্বীপ", "বন", "হ্রদ",
    "জেলা", "উপজেলা", "বিভাগ", "শহর", "নগর", "গ্রাম", "রাজধানী", "বন্দর",
    "বিশ্ববিদ্যালয়", "কলেজ", "বিদ্যালয়", "মাদ্রাসা", "প্রতিষ্ঠান", "সংস্থা", "সরকার",
    "রাষ্ট্র", "দেশ", "সাম্রাজ্য", "রাজ্য", "দল", "সংগঠন", "মন্ত্রণালয়", "কমিশন",
    "যুদ্ধ", "আন্দোলন", "বিপ্লব", "সন্ধি", "চুক্তি", "সম্মেলন",
})

HONORIFICS: frozenset[str] = frozenset({
    "শেখ", "ড", "ডঃ", "ডক্টর", "স্যার", "মি", "মিস্টার", "অধ্যাপক", "কবি", "লেখক",
    "রাষ্ট্রপতি", "প্রধানমন্ত্রী", "সম্রাট", "রাজা", "নবাব", "বীর", "শহীদ", "মহাত্মা",
})

# Directional markers.  "ক থেকে খ" and "খ থেকে ক" have an identical token set
# and opposite truth, so bag-of-words similarity is blind to them *by
# construction* — this is the one veto category no similarity signal can ever
# replace.
DIRECTION_MARKERS: tuple[str, ...] = ("থেকে", "হতে", "চেয়ে", "অপেক্ষা")
DIRECTION_TARGETS: tuple[str, ...] = ("পরিণত", "রূপান্তরিত", "উৎপন্ন", "সৃষ্টি", "তৈরি",
                                      "বড়", "ছোট", "বেশি", "কম", "উঁচু", "নিচু")

MONTHS: frozenset[str] = frozenset({
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট",
    "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর",
    "বৈশাখ", "জ্যৈষ্ঠ", "আষাঢ়", "শ্রাবণ", "ভাদ্র", "আশ্বিন", "কার্তিক", "অগ্রহায়ণ",
    "পৌষ", "মাঘ", "ফাল্গুন", "চৈত্র",
})

YEAR_WORDS: frozenset[str] = frozenset({"সাল", "সালে", "সালের", "খ্রিষ্টাব্দ",
                                        "খ্রিস্টাব্দ", "খ্রিষ্টপূর্ব", "বঙ্গাব্দ"})

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_LATIN_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,})*\b")


# --------------------------------------------------------------------------- #
# The verdict                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Veto:
    """One fired constraint, with the string the UI shows the student."""

    category: str
    reason: str          # Bangla, shown verbatim in the UI
    answer_value: str
    evidence_value: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.category}] {self.reason}"


@dataclass
class VetoResult:
    vetoes: list[Veto] = field(default_factory=list)

    @property
    def fired(self) -> bool:
        return bool(self.vetoes)

    @property
    def categories(self) -> list[str]:
        return [v.category for v in self.vetoes]

    @property
    def reason(self) -> str:
        """The single reason string the UI prints, or ``""`` when none fired."""
        return "; ".join(v.reason for v in self.vetoes)

    def cap(self, support: float) -> float:
        """Apply the cap.  A contradiction bounds support **regardless of similarity**."""
        return min(support, VETO_SUPPORT_CAP) if self.fired else support

    def as_dict(self) -> dict:
        return {"fired": self.fired, "categories": self.categories,
                "reason": self.reason,
                "detail": [v.__dict__ for v in self.vetoes]}


VETO_SUPPORT_CAP = 0.15


# --------------------------------------------------------------------------- #
# Extractors — all on normalised-but-unfuzzed text                             #
# --------------------------------------------------------------------------- #


def numbers(text: str) -> list[str]:
    """Numeric literals.  ``normalize`` has already folded ১৯৭১ to 1971."""
    return _NUMBER_RE.findall(normalize(text))


def years(text: str) -> list[str]:
    """Numbers that denote a year: four digits in a plausible range, or
    any number sitting next to সাল / খ্রিষ্টাব্দ / a month name."""
    norm = normalize(text)
    toks = tokenize(norm)
    out: list[str] = []
    for i, tok in enumerate(toks):
        if not tok.isdigit():
            continue
        neighbours = set(toks[max(0, i - 1):i + 2])
        if (len(tok) == 4 and 1000 <= int(tok) <= 2100) or \
           (neighbours & YEAR_WORDS) or (neighbours & MONTHS):
            out.append(tok)
    return out


def units(text: str) -> list[tuple[str, str]]:
    """``(number, unit)`` pairs — a unit is the token after a number."""
    toks = tokenize(normalize(text))
    out: list[tuple[str, str]] = []
    for i, tok in enumerate(toks[:-1]):
        nxt = toks[i + 1]
        if tok.replace(".", "").replace(",", "").isdigit() and nxt.lower() in UNITS:
            out.append((tok, nxt.lower()))
    return out


def bare_units(text: str) -> set[str]:
    return {t.lower() for t in tokenize(normalize(text)) if t.lower() in UNITS}


def entities(text: str, *, gazetteer: frozenset[str] | None = None) -> set[str]:
    """Rule-based proper-noun proxy.  Recall-limited on purpose (see module doc)."""
    norm = normalize(text)
    out: set[str] = set(m.group(0) for m in _LATIN_PROPER_RE.finditer(norm))
    toks = tokenize(norm)
    for i, tok in enumerate(toks):
        if tok in ENTITY_MARKERS:
            if i > 0 and toks[i - 1] not in ENTITY_MARKERS:
                out.add(toks[i - 1])
        elif tok in HONORIFICS and i + 1 < len(toks):
            out.add(toks[i + 1])
        elif gazetteer and tok in gazetteer:
            out.add(tok)
    # A number is never an entity.  Years and quantities have their own rules
    # above, and letting one through here produces "নাম ভুল" ("wrong name")
    # for a date — which is both wrong and, in the UI, actively confusing.
    # The gazetteer is harvested from passage titles, so four-digit years leak
    # into it easily.
    return {e for e in out if not e.replace(".", "").replace(",", "").isdigit()}


def polarity(text: str) -> bool:
    """``True`` when the statement is affirmative, ``False`` when negated.

    Negation is never a stopword (``stopwords.VETO_SAFE``), so it survives every
    preprocessing step that runs before this one.
    """
    toks = set(tokenize(normalize(text)))
    return not bool(toks & NEGATION)


def _is_number(token: str) -> bool:
    return token.replace(".", "").replace(",", "").isdigit()


MIN_SWAP_TOKENS = 4          # raw tokens, not content tokens — see term_swap
MIN_SWAP_SHARED = 2
SWAP_OVERLAP_FLOOR = 0.60


def term_swap(answer: str, evidence: str) -> tuple[str, str] | None:
    """Detect a **minimal-pair term substitution**: one content word replaced.

    This is what makes row 1 of PLAN.md §10.1 catchable without a Bangla NER.
    *উদ্ভিদ **অক্সিজেন** গ্রহণ করে* against *উদ্ভিদ **কার্বন ডাই-অক্সাইড** গ্রহণ
    করে* shares every other token; the substances are not proper nouns, so no
    entity rule reaches them, and every similarity metric scores the pair at
    ≈1.0.

    The rule is pure set arithmetic over **exactly matched** content tokens —
    no edit distance, no n-gram overlap, no embedding — so it cannot smear one
    veto term into another.  It fires only when the two strings are otherwise
    near-copies of each other, which is the definition of a minimal pair and
    is what keeps it from firing on a bare extracted span that merely fails to
    restate the sentence.

    Returns ``(answer_term, evidence_term)`` — the words the UI names.
    """
    from ..preprocess.stopwords import content_tokens
    from ..resources.loader import equivalent

    # The length gate counts *raw* tokens: "উদ্ভিদ অক্সিজেন গ্রহণ করে" is a full
    # sentence with only three content words, and gating on content tokens
    # would exempt exactly the short factual statements this rule exists for.
    if len(tokenize(answer)) < MIN_SWAP_TOKENS or len(tokenize(evidence)) < MIN_SWAP_TOKENS:
        return None

    a_set = set(content_tokens(tokenize(answer)))
    e_set = set(content_tokens(tokenize(evidence)))
    shared = a_set & e_set
    # Numbers belong to the date and numeral rules, which have already run and
    # produce a better message.  Reporting the same swapped year twice — once
    # as "সাল ভুল" and again as "শব্দ ভুল" — is noise in the UI.
    only_a = sorted(t for t in a_set - e_set if not _is_number(t))
    only_e = sorted(t for t in e_set - a_set if not _is_number(t))

    if len(shared) < MIN_SWAP_SHARED:
        return None
    # near-copy test: most of the answer's content is already in the evidence
    if len(shared) / max(len(a_set), 1) < SWAP_OVERLAP_FLOOR:
        return None
    # a *substitution* means something went out and something came in; one
    # side being empty is an omission or an addition, which is S2's business
    if not only_a or not only_e:
        return None
    # more than a couple of differences is a different sentence, not a swap
    if len(only_a) > 2:
        return None

    # A **paraphrase is not a contradiction.**  §10.4 requires grounding to be
    # matched *through* the T6b resources so a correct rewording is not
    # punished, and that applies here first: if the differing term is in the
    # evidence term's synonym class, nothing was swapped.  This is an exact
    # lookup in a hand-reviewed lexicon, not fuzzy matching, so the isolation
    # guarantee in this module's docstring still holds.
    for a_t in only_a:
        if any(equivalent(a_t, e_t) for e_t in only_e):
            return None
    # Name the longest differing term from the book: when a multi-word compound
    # was replaced ("কার্বন ডাই-অক্সাইড" -> "অক্সিজেন") its longest token reads
    # to a student as the thing that changed, where the alphabetically first
    # one ("অক্সাইড") reads as a fragment.
    return only_a[0], max(only_e, key=len)


def relation_order(text: str) -> list[tuple[str, str]]:
    """``(left, right)`` pairs around a directional marker."""
    toks = tokenize(normalize(text))
    out: list[tuple[str, str]] = []
    for i, tok in enumerate(toks):
        if tok in DIRECTION_MARKERS and 0 < i < len(toks) - 1:
            out.append((toks[i - 1], toks[i + 1]))
    return out


# --------------------------------------------------------------------------- #
# The check                                                                    #
# --------------------------------------------------------------------------- #


def check(answer: str, evidence: str, *, gazetteer: frozenset[str] | None = None,
          check_polarity: bool = True) -> VetoResult:
    """Does ``evidence`` contradict ``answer``?

    Every rule fires only on **conflict**, never on absence.  A value the
    evidence simply does not discuss is a *coverage* question and belongs to the
    grounding signal S2; a value the evidence states **differently** is a
    contradiction and belongs here.  Conflating the two would turn the veto into
    a second, worse similarity measure.
    """
    result = VetoResult()
    a_norm, e_norm = normalize(answer), normalize(evidence)

    # ---- 1. dates -------------------------------------------------------
    a_years, e_years = years(a_norm), years(e_norm)
    for y in a_years:
        if y not in e_years and e_years:
            result.vetoes.append(Veto(
                "date", f"সাল ভুল — বইয়ে {_bn(e_years[0])}, উত্তরে {_bn(y)}", y, e_years[0]))
            break

    # ---- 2. numerals ----------------------------------------------------
    a_nums = [n for n in numbers(a_norm) if n not in a_years]
    e_nums = [n for n in numbers(e_norm) if n not in e_years]
    for n in a_nums:
        if n not in e_nums and e_nums:
            result.vetoes.append(Veto(
                "numeral", f"সংখ্যা ভুল — বইয়ে {_bn(e_nums[0])}, উত্তরে {_bn(n)}", n, e_nums[0]))
            break

    # ---- 3. units -------------------------------------------------------
    a_units, e_units = bare_units(a_norm), bare_units(e_norm)
    extra = a_units - e_units
    if extra and e_units:
        a_u, e_u = sorted(extra)[0], sorted(e_units)[0]
        result.vetoes.append(Veto("unit", f"একক ভুল — বইয়ে {e_u}, উত্তরে {a_u}", a_u, e_u))

    # ---- 4. entities and term swaps -------------------------------------
    a_ents = entities(a_norm, gazetteer=gazetteer)
    e_ents = entities(e_norm, gazetteer=gazetteer)
    unmatched = {e for e in a_ents if e not in e_norm}
    if unmatched and e_ents:
        a_e, e_e = sorted(unmatched)[0], sorted(e_ents)[0]
        result.vetoes.append(Veto("entity", f"নাম ভুল — বইয়ে {e_e}, উত্তরে {a_e}", a_e, e_e))
    else:
        swap = term_swap(a_norm, e_norm)
        if swap is not None:
            a_t, e_t = swap
            result.vetoes.append(Veto(
                "entity", f"শব্দ ভুল — বইয়ে {e_t}, উত্তরে {a_t}", a_t, e_t))

    # ---- 5. polarity ----------------------------------------------------
    # Only meaningful between two *statements*.  A bare extracted span carries no
    # polarity of its own, so a short span is skipped rather than guessed at.
    if check_polarity and len(tokenize(a_norm)) >= 3:
        if polarity(a_norm) != polarity(e_norm):
            said = "হ্যাঁ-বাচক" if polarity(a_norm) else "না-বাচক"
            book = "হ্যাঁ-বাচক" if polarity(e_norm) else "না-বাচক"
            result.vetoes.append(Veto(
                "polarity", f"অর্থ উল্টো — বইয়ে {book}, উত্তরে {said}",
                said, book))

    # ---- 6. relation order ----------------------------------------------
    for left, right in relation_order(a_norm):
        for e_left, e_right in relation_order(e_norm):
            if {left, right} == {e_left, e_right} and (left, right) != (e_left, e_right):
                result.vetoes.append(Veto(
                    "relation_order",
                    f"ক্রম উল্টো — বইয়ে “{e_left} থেকে {e_right}”, "
                    f"উত্তরে “{left} থেকে {right}”",
                    f"{left}->{right}", f"{e_left}->{e_right}"))
                break

    return result


def _bn(text: str) -> str:
    """Bengali digits for display.  Matching always uses the ASCII form."""
    from ..preprocess.normalize import to_bengali_digits

    return to_bengali_digits(text)


def best_evidence_sentence(claim: str, passage: str) -> str:
    """The sentence of ``passage`` a claim should be checked against.

    **The veto layer is a sentence-level instrument.**  Polarity especially:
    almost any passage of school text contains a negation somewhere, so
    comparing an affirmative claim against a whole passage reports a polarity
    contradiction that is not there.  Measured on the BanglaVerify contrast
    set, checking claims against whole passages fired on **29% of SUPPORTED
    claims** — 16 polarity, 3 numeral, 2 date — and dropped supported accuracy
    from 0.936 to 0.694.  The numeral and date rules degrade the same way,
    because a passage holds many numbers and only one of them is the claim's.

    The pipeline never had this problem: it checks an answer against its own
    evidence sentence.  Anything holding a *passage* has to narrow it first,
    and this is that step.

    Selection is exact content-token overlap — set arithmetic, no similarity
    measure — so the isolation guarantee in this module's docstring holds.
    """
    from ..preprocess.stopwords import content_tokens

    sentences = [s for s, _a, _b in _sentence_spans(normalize(passage))]
    if len(sentences) <= 1:
        return passage
    claim_tokens = set(content_tokens(tokenize(claim)))
    if not claim_tokens:
        return passage
    best, best_score = passage, -1.0
    for sent in sentences:
        toks = set(content_tokens(tokenize(sent)))
        if not toks:
            continue
        score = len(claim_tokens & toks) / len(claim_tokens)
        if score > best_score:
            best, best_score = sent, score
    return best


def _sentence_spans(text: str):
    from ..preprocess.tokenize import sentence_spans

    return sentence_spans(text)


def check_passage(claim: str, passage: str, *,
                  gazetteer: frozenset[str] | None = None) -> VetoResult:
    """:func:`check` against the passage's most relevant sentence."""
    return check(claim, best_evidence_sentence(claim, passage), gazetteer=gazetteer)


def veto_features(answer: str, evidence: str, *,
                  gazetteer: frozenset[str] | None = None) -> dict[str, float]:
    """The veto layer as features for S3's classifier (§10.4)."""
    result = check(answer, evidence, gazetteer=gazetteer)
    cats = set(result.categories)
    feats = {f"veto_{c}": float(c in cats) for c in
             ("numeral", "date", "unit", "entity", "polarity", "relation_order")}
    feats["veto_any"] = float(result.fired)
    feats["veto_count"] = float(len(result.vetoes))
    feats["polarity_match"] = float(polarity(answer) == polarity(evidence))
    a_years, e_years = set(years(answer)), set(years(evidence))
    feats["year_in_evidence"] = float(bool(a_years) and a_years <= e_years)
    feats["answer_has_year"] = float(bool(a_years))
    a_nums, e_nums = set(numbers(answer)), set(numbers(evidence))
    feats["nums_in_evidence"] = float(bool(a_nums) and a_nums <= e_nums)
    feats["answer_has_number"] = float(bool(a_nums))
    return feats
