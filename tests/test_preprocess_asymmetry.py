"""T4's second acceptance clause: the suffix stripper is applied to **sparse
models only** (PLAN.md §7.2).

The asymmetry is deliberate and reported: stemming helps lexical matching and
hurts distributional representations.  Applying it to both arms would make RQ1
a comparison of preprocessing rather than of retrievers — the experiment would
still produce a number, and the number would mean nothing.

So the rule is enforced structurally: stemming lives in
``retrieval.base.sparse_terms`` and nowhere else, and this file asserts that
the tokeniser itself leaves surface forms alone.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from bnqa.preprocess import stem as stem_mod
from bnqa.preprocess.stem import set_vocabulary, stem, stem_tokens
from bnqa.preprocess.tokenize import tokenize_lower
from bnqa.retrieval.base import sparse_terms


def test_tokenizer_does_not_stem():
    """The raw tokeniser must return surface forms — dense arms depend on it."""
    assert tokenize_lower("উদ্ভিদের পাতা") == ["উদ্ভিদের", "পাতা"]


def test_sparse_terms_do_stem():
    set_vocabulary({"উদ্ভিদ", "পাতা"})
    try:
        assert "উদ্ভিদ" in sparse_terms("উদ্ভিদের পাতা")
    finally:
        set_vocabulary(None)


def test_stemming_converges_inflected_forms():
    """The point of stemming is convergence.  A stemmer that *splits* is worse
    than none — TASKS.md finding #4."""
    set_vocabulary({"উদ্ভিদ", "উদ্ভিদগুলো"})
    try:
        assert stem("উদ্ভিদের") == stem("উদ্ভিদ") == "উদ্ভিদ"
    finally:
        set_vocabulary(None)


def test_corpus_validation_rejects_an_unattested_stem():
    """Longest-match would give উদ্ভি; corpus validation must refuse it."""
    set_vocabulary({"উদ্ভিদ"})
    try:
        assert stem("উদ্ভিদের") == "উদ্ভিদ"
        assert stem("উদ্ভিদের") != "উদ্ভি"
    finally:
        set_vocabulary(None)


def test_stem_never_ends_on_hasant():
    set_vocabulary(None)
    for word in ("উদ্ভিদের", "শিক্ষকের", "বইগুলোর", "ছাত্রদের"):
        assert not stem(word).endswith(stem_mod.HASANT)


def test_stem_tokens_is_elementwise():
    set_vocabulary(None)
    toks = ["উদ্ভিদের", "পাতা"]
    assert stem_tokens(toks) == [stem(t) for t in toks]


def test_only_the_sparse_path_imports_the_stemmer():
    """Nothing outside retrieval/base.py and the metric ladder may stem.

    ``eval/qa_metrics.py`` is the one other legitimate caller: tier-3 F1 is
    *defined* as stem + synonym matching (PLAN.md §9), and it is an evaluation
    metric rather than a model input, so it cannot contaminate RQ1.
    """
    src = Path(inspect.getfile(stem_mod)).resolve().parents[2]
    allowed = {
        "bnqa/retrieval/base.py",      # the sparse analyser — the intended caller
        "bnqa/eval/qa_metrics.py",     # tier-3 F1, a metric and not a model input
        "bnqa/retrieval/index.py",     # installs the corpus vocabulary
        "bnqa/verify/signals.py",      # S2 matches *through* T6b resources (§10.4)
        "bnqa/verify/support.py",
        "bnqa/preprocess/stem.py",
    }
    offenders: list[str] = []
    for path in src.rglob("bnqa/**/*.py"):
        rel = path.relative_to(src).as_posix()
        if rel in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "stem" in node.module:
                offenders.append(rel)
            elif isinstance(node, ast.Import):
                if any("stem" in a.name for a in node.names):
                    offenders.append(rel)
    assert not offenders, (
        f"{sorted(set(offenders))} import the stemmer; PLAN.md §7.2 confines it "
        "to the sparse path")
