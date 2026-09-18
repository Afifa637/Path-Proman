"""Normalisation round-trips and idempotence (PLAN.md §18.1, task T4).

Everything downstream assumes these hold: dedup keys, the passage sha256 in an
evidence receipt (VC-3), and — most importantly — the character offsets that
VC-1 prints and ``verify_receipt`` re-checks.  If ``normalize`` were not
idempotent, a span stored today would not resolve tomorrow.
"""

from __future__ import annotations

import pytest

from bnqa.preprocess.normalize import (DARI, bengali_ratio, normalize,
                                       normalize_for_match, to_bengali_digits)

SAMPLES = [
    "উদ্ভিদ সালোকসংশ্লেষণ প্রক্রিয়ায় কার্বন ডাই-অক্সাইড গ্রহণ করে।",
    "১৯৭১ সালে মুক্তিযুদ্ধ শুরু হয়।",
    "CO2 এবং H2O — “উদ্ধৃতি”  সহ।",
    "শিক্ষার্থীদের‌জন্য‍বিশেষ  নির্দেশনা॥",
    "",
    "   ",
]


@pytest.mark.parametrize("text", SAMPLES)
def test_idempotent(text: str) -> None:
    once = normalize(text)
    assert normalize(once) == once


@pytest.mark.parametrize("text", SAMPLES)
def test_match_form_idempotent(text: str) -> None:
    once = normalize_for_match(text)
    assert normalize_for_match(once) == once


def test_zero_width_joiners_removed() -> None:
    assert "‌" not in normalize("শিক্ষা‌র্থী")
    assert "‍" not in normalize("শিক্ষা‍র্থী")
    assert normalize("উদ্ভিদ‌ের") == normalize("উদ্ভিদের")


def test_nukta_forms_converge() -> None:
    """ড় ঢ় য় are Unicode composition exclusions, so NFC decomposes them.

    Both spellings occur in the wild; if they did not converge, the same word
    would hash to two different dedup keys.
    """
    for precomposed, decomposed in (("ড়", "ড়"),
                                    ("ঢ়", "ঢ়"),
                                    ("য়", "য়")):
        assert normalize(precomposed) == normalize(decomposed)


def test_bengali_digits_folded_to_ascii() -> None:
    assert normalize("১৯৭১") == "1971"
    assert normalize("০১২৩৪৫৬৭৮৯") == "0123456789"


def test_digits_can_be_preserved_for_display() -> None:
    assert normalize("১৯৭১", fold_digits=False) == "১৯৭১"
    assert to_bengali_digits(normalize("১৯৭১")) == "১৯৭১"


def test_double_dari_becomes_single() -> None:
    assert normalize("শেষ॥") == "শেষ" + DARI


def test_curly_quotes_and_dashes_standardised() -> None:
    out = normalize("“উক্তি” — ‘আরেকটি’")
    assert '"উক্তি"' in out
    assert "—" not in out and "‘" not in out


def test_whitespace_collapsed_and_trimmed() -> None:
    assert normalize("  ক   খ \t গ  ") == "ক খ গ"
    assert normalize("ক\n\n\n\n\nখ") == "ক\n\nখ"


def test_space_before_punctuation_removed() -> None:
    assert normalize("বাক্য ।") == "বাক্য।"
    assert normalize("প্রশ্ন ?") == "প্রশ্ন?"


def test_normalize_for_match_strips_punctuation_and_case() -> None:
    assert normalize_for_match("CO2, এবং H2O।") == "co2 এবং h2o"


def test_bengali_ratio() -> None:
    assert bengali_ratio("উদ্ভিদ") == pytest.approx(1.0)
    assert bengali_ratio("plant") == pytest.approx(0.0)
    assert 0.0 < bengali_ratio("উদ্ভিদ plant") < 1.0
    assert bengali_ratio("") == 0.0


def test_empty_input_is_safe() -> None:
    assert normalize("") == ""
    assert normalize_for_match("") == ""
