"""T4's acceptance: normalisation is idempotent and round-trips (PLAN.md §7.2).

Everything downstream assumes this — dedup keys, passage hashes in evidence
receipts, and every span offset the reader returns.  If ``normalize`` were not
idempotent, a receipt written today would fail its own check tomorrow.
"""

from __future__ import annotations

import pytest

from bnqa.preprocess.normalize import (bengali_ratio, normalize, normalize_for_match,
                                       to_bengali_digits)

FIXTURES = [
    "সালোকসংশ্লেষণ প্রক্রিয়ায় উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে।",
    "মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়।",
    "CO2 এবং   H2O   থেকে   গ্লুকোজ তৈরি হয়",
    "উদ্ভিদ‌ের পাতা‍য় ক্লোরোফিল থাকে",          # contains ZWNJ and ZWJ
    "“উদ্ধৃতি” — একটি বাক্য … শেষ॥",
    "",
    "   ",
]


@pytest.mark.parametrize("text", FIXTURES)
def test_idempotent(text):
    once = normalize(text)
    assert normalize(once) == once


@pytest.mark.parametrize("text", FIXTURES)
def test_match_form_idempotent(text):
    once = normalize_for_match(text)
    assert normalize_for_match(once) == once


def test_zero_width_removed():
    with_zwnj = "উদ্ভিদ‌ের"
    with_zwj = "উদ্ভিদ‍ের"
    assert normalize(with_zwnj) == normalize(with_zwj) == normalize("উদ্ভিদের")


def test_bengali_digits_folded():
    assert normalize("১৯৭১") == "1971"
    assert normalize("২০২৬ সালে") == "2026 সালে"


def test_digits_round_trip_for_display():
    assert to_bengali_digits(normalize("১৯৭১")) == "১৯৭১"


def test_double_dari_normalised():
    assert normalize("শেষ॥") == "শেষ।"


def test_whitespace_collapsed():
    assert normalize("ক    খ\t\tগ") == "ক খ গ"


def test_bengali_ratio():
    assert bengali_ratio("উদ্ভিদ") == 1.0
    assert bengali_ratio("abc") == 0.0
    assert 0.0 < bengali_ratio("উদ্ভিদ abc") < 1.0
    assert bengali_ratio("") == 0.0


def test_preserves_digits_when_asked():
    assert "১৯৭১" in normalize("১৯৭১ সাল", fold_digits=False)
