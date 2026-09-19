"""The test PLAN.md §10.2 names by file — **fuzzy matching must never reach a
veto term**.

This failure mode is dangerous precisely because it *improves* every headline
score.  Tolerant matching over numerals maps ১৯৫২ to ১৯৭১; over entities it
maps অক্সিজেন to অক্সাইড.  Either one silently destroys the only layer that
catches contradiction, and F1 goes **up** while the system gets worse — so it
would survive code review, and only a test that asserts the boundary can catch
it.

Two things are asserted here:

1. the veto layer's *behaviour* — the three contradictions from §10.1, each of
   which any soft metric scores at ≈1.0, are all caught;
2. the veto layer's *isolation* — nothing in ``verify.constraints`` imports a
   similarity measure, and S2 returns the capped value rather than a blended
   one whenever a veto fires.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from bnqa.reader.base import Answer
from bnqa.verify import constraints
from bnqa.verify.constraints import VETO_SUPPORT_CAP, check, polarity, years
from bnqa.verify.signals import SignalContext, s2_grounding

# The three rows of PLAN.md §10.1's table.
CONTRADICTIONS = [
    ("উদ্ভিদ অক্সিজেন গ্রহণ করে",
     "উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে", "entity/polarity"),
    ("সালোকসংশ্লেষণ দিনে হয় না",
     "সালোকসংশ্লেষণ দিনে হয়", "polarity"),
    ("মুক্তিযুদ্ধ ১৯৫২ সালে সংঘটিত হয়",
     "মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়", "date"),
]


# --------------------------------------------------------------------------- #
# 1. behaviour                                                                 #
# --------------------------------------------------------------------------- #


def test_year_contradiction_fires():
    result = check("মুক্তিযুদ্ধ ১৯৫২ সালে সংঘটিত হয়",
                   "মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়")
    assert result.fired
    assert "date" in result.categories
    assert "১৯৭১" in result.reason and "১৯৫২" in result.reason


def test_polarity_contradiction_fires():
    result = check("সালোকসংশ্লেষণ দিনে হয় না", "সালোকসংশ্লেষণ দিনে হয়")
    assert result.fired
    assert "polarity" in result.categories


def test_numeral_contradiction_fires():
    result = check("এখানে 7 টি স্তর আছে", "এখানে 5 টি স্তর আছে")
    assert result.fired
    assert "numeral" in result.categories


def test_unit_contradiction_fires():
    result = check("দৈর্ঘ্য 5 কিলোমিটার", "দৈর্ঘ্য 5 মিটার")
    assert result.fired
    assert "unit" in result.categories


def test_relation_order_contradiction_fires():
    """Identical token set, opposite truth — no similarity signal can see this."""
    result = check("অক্সিজেন থেকে কার্বন উৎপন্ন হয়",
                   "কার্বন থেকে অক্সিজেন উৎপন্ন হয়")
    assert result.fired
    assert "relation_order" in result.categories


def test_agreement_does_not_fire():
    assert not check("মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়",
                     "মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়").fired


def test_absence_is_not_contradiction():
    """A value the evidence never discusses is a *coverage* question, not a veto."""
    assert not check("কার্বন ডাই-অক্সাইড", "উদ্ভিদ একটি জীব").fired


def test_veto_caps_support():
    result = check("মুক্তিযুদ্ধ ১৯৫২ সালে", "মুক্তিযুদ্ধ ১৯৭১ সালে")
    assert result.cap(0.99) == pytest.approx(VETO_SUPPORT_CAP)
    assert check("ক", "ক").cap(0.99) == pytest.approx(0.99)


# --------------------------------------------------------------------------- #
# 2. isolation — the part that catches the dangerous bug                       #
# --------------------------------------------------------------------------- #

FORBIDDEN_IMPORTS = {
    "difflib", "Levenshtein", "rapidfuzz", "fuzzywuzzy",
    "sklearn", "numpy", "scipy",
}
FORBIDDEN_NAMES = {"trigram_cosine", "char_trigrams", "containment",
                   "SequenceMatcher", "cosine_similarity", "get_close_matches"}


def test_constraints_imports_no_similarity_machinery():
    """No fuzzy or vector similarity may be importable from the veto layer."""
    source = Path(inspect.getfile(constraints)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
            imported.update(a.name for a in node.names)

    leaked = imported & (FORBIDDEN_IMPORTS | FORBIDDEN_NAMES)
    assert not leaked, (
        f"verify/constraints.py imports {sorted(leaked)} — fuzzy matching must "
        "never reach a veto term (PLAN.md §10.2)")


def test_veto_terms_survive_normalisation():
    """Normalisation may fold digits, but never *change* a veto value."""
    from bnqa.preprocess.normalize import normalize

    assert years(normalize("১৯৭১ সালে")) == ["1971"]
    assert years(normalize("1971 সালে")) == ["1971"]
    assert polarity(normalize("হয় না")) is False
    assert polarity(normalize("হয়")) is True


def test_negation_is_never_a_stopword():
    """Removing না would make a sentence and its negation identical."""
    from bnqa.preprocess.stopwords import STOPWORDS, VETO_SAFE, remove_stopwords

    assert not (VETO_SAFE & STOPWORDS)
    assert "না" in remove_stopwords(["সালোকসংশ্লেষণ", "হয়", "না"])


def test_s2_returns_the_cap_not_a_blend_when_vetoed():
    """S2 must not average a contradiction away with a high similarity."""
    answer = Answer(text="মুক্তিযুদ্ধ ১৯৫২ সালে সংঘটিত হয়", pid="p", char_start=0,
                    char_end=10, sentence="", sentence_start=0, sentence_end=0)
    ctx = SignalContext(question="মুক্তিযুদ্ধ কত সালে?", answer=answer,
                        evidence="মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়")
    assert s2_grounding(ctx) == pytest.approx(VETO_SUPPORT_CAP)


@pytest.mark.parametrize("candidate,evidence,_kind", CONTRADICTIONS)
def test_every_plan_contradiction_is_caught(candidate, evidence, _kind):
    assert check(candidate, evidence).fired, (
        f"{candidate!r} vs {evidence!r} scores ~1.0 on any soft metric and must "
        "be caught by the veto layer")
