"""Stemmer and tokenizer behaviour (PLAN.md §7.2, task T4).

The stemmer's job is **convergence**, not linguistic correctness: if ``উদ্ভিদ``
and ``উদ্ভিদের`` map to different strings the stemmer has made lexical matching
worse than not stemming at all.  These tests pin that property, including the
ambiguous case that longest-match-first gets wrong.
"""

from __future__ import annotations

import pytest

from bnqa.preprocess import stem as stem_mod
from bnqa.preprocess.stem import MIN_STEM_CHARS, load_suffixes, set_vocabulary, stem
from bnqa.preprocess.stopwords import STOPWORDS, VETO_SAFE, remove_stopwords
from bnqa.preprocess.tokenize import (char_ngrams, sentence_spans, sentences,
                                      token_spans, tokenize)

HASANT = "্"


@pytest.fixture(autouse=True)
def _clean_vocabulary():
    """Each test controls the stemmer's evidence explicitly."""
    set_vocabulary(None)
    yield
    set_vocabulary(None)


# --------------------------------------------------------------------------- #
# Stemmer                                                                      #
# --------------------------------------------------------------------------- #


def test_suffix_list_is_longest_first() -> None:
    sufs = load_suffixes()
    assert len(sufs) > 40
    assert all(len(a) >= len(b) for a, b in zip(sufs, sufs[1:]))


@pytest.mark.parametrize("word,expected", [
    ("উদ্ভিদের", "উদ্ভিদ"),
    ("উদ্ভিদগুলোর", "উদ্ভিদ"),
    ("প্রক্রিয়ার", "প্রক্রিয়া"),
    ("বায়ুমণ্ডলে", "বায়ুমণ্ডল"),
    ("সালোকসংশ্লেষণের", "সালোকসংশ্লেষণ"),
])
def test_common_inflections_strip(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_inflected_forms_converge() -> None:
    forms = ["উদ্ভিদ", "উদ্ভিদের", "উদ্ভিদগুলোর"]
    assert len({stem(f) for f in forms}) == 1


def test_ambiguous_split_is_resolved_by_corpus_evidence() -> None:
    """``শিক্ষার্থীদের`` is ``শিক্ষার্থী``+``দের``, not ``শিক্ষার্থীদ``+``ের``.

    Without evidence the conservative tier order picks the wrong split; with the
    corpus vocabulary installed (as T5 does) the attested stem wins.
    """
    assert stem("শিক্ষার্থীদের") != "শিক্ষার্থী"  # no evidence available
    set_vocabulary({"শিক্ষার্থী"})
    assert stem("শিক্ষার্থীদের") == "শিক্ষার্থী"
    assert stem("শিক্ষার্থী") == "শিক্ষার্থী"


def test_vocabulary_prevents_overstemming() -> None:
    """``উদ্ভিদের`` must not lose its stem-final দ to the plural suffix দের."""
    set_vocabulary({"উদ্ভিদ"})
    assert stem("উদ্ভিদের") == "উদ্ভিদ"
    assert stem("উদ্ভিদের") == stem("উদ্ভিদ")


def test_short_words_are_left_alone() -> None:
    assert stem("তার") == "তার"
    assert all(len(stem(w)) >= min(len(w), MIN_STEM_CHARS) for w in ["কে", "যা", "এর"])


def test_stem_never_ends_on_hasant() -> None:
    for word in ["উদ্ভিদের", "শিক্ষার্থীদের", "বিদ্যুৎের", "রাষ্ট্রের"]:
        assert not stem(word).endswith(HASANT)


def test_stemming_is_deterministic() -> None:
    set_vocabulary({"উদ্ভিদ"})
    assert stem("উদ্ভিদের") == stem("উদ্ভিদের")


# --------------------------------------------------------------------------- #
# Stopwords — the sparse-only rule and the veto guard                          #
# --------------------------------------------------------------------------- #


def test_negation_is_never_a_stopword() -> None:
    """Removing না/নয়/নেই would make a sentence and its negation identical to
    the retriever — the exact failure the veto layer exists to catch."""
    for word in VETO_SAFE:
        assert word not in STOPWORDS
        assert remove_stopwords([word]) == [word]


def test_numerals_survive_stopword_removal() -> None:
    assert remove_stopwords(["1971", "সালে", "এবং"]) == ["1971", "সালে"]


def test_common_function_words_are_removed() -> None:
    assert remove_stopwords(["এবং", "উদ্ভিদ", "থেকে"]) == ["উদ্ভিদ"]


# --------------------------------------------------------------------------- #
# Tokenizer                                                                    #
# --------------------------------------------------------------------------- #


def test_tokenizer_splits_scripts_and_numbers() -> None:
    assert tokenize("সালোকসংশ্লেষণে CO2 ও ১৯৭১ সালের") == [
        "সালোকসংশ্লেষণে", "CO2", "ও", "1971", "সালের"]


def test_token_spans_index_the_normalised_string() -> None:
    from bnqa.preprocess.normalize import normalize

    text = normalize("উদ্ভিদ ১৯৭১ সালে")
    for tok, start, end in token_spans(text):
        assert text[start:end] == tok


def test_sentence_split_on_dari_and_marks() -> None:
    assert sentences("প্রথম বাক্য। দ্বিতীয়? তৃতীয়!") == [
        "প্রথম বাক্য।", "দ্বিতীয়?", "তৃতীয়!"]


def test_sentence_spans_are_exact_substrings() -> None:
    from bnqa.preprocess.normalize import normalize

    text = normalize("প্রথম বাক্য। দ্বিতীয় বাক্য। তৃতীয় বাক্য।")
    spans = sentence_spans(text)
    assert len(spans) == 3
    for sent, start, end in spans:
        assert text[start:end] == sent


def test_char_ngrams_share_the_stem() -> None:
    """The property that lets the char arm absorb inflection without a stemmer."""
    a = set(char_ngrams("উদ্ভিদ", 3, 5))
    b = set(char_ngrams("উদ্ভিদের", 3, 5))
    assert len(a & b) >= 3
