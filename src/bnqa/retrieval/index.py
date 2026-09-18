"""Index loading, the corpus vocabulary, and persistence (PLAN.md §7.3, task T5).

Also the place where the stemmer stops guessing.  ``bnqa.preprocess.stem`` picks
between ambiguous splits (``উদ্ভিদ+ের`` vs ``উদ্ভি+দের``) by asking whether the
candidate stem is an attested token; the attestation comes from **our own
corpus**, built here and installed once before any retriever is fitted.  Nothing
external is consulted, so Tier A's "nothing downloaded" claim holds.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

from ..config import CFG, PROCESSED
from ..preprocess.stem import set_vocabulary, vocabulary_size
from ..preprocess.tokenize import tokenize_lower
from ..utils import read_json, write_json

VOCAB_PATH = PROCESSED / "corpus_vocab.json"
#: a stem must be attested at least this often to be trusted
VOCAB_MIN_COUNT = 3


def build_vocabulary(passages, *, min_count: int = VOCAB_MIN_COUNT) -> set[str]:
    counts: Counter = Counter()
    for p in passages:
        counts.update(tokenize_lower(p["text"]))
    return {w for w, c in counts.items() if c >= min_count}


def load_or_build_vocabulary(passages=None, *, rebuild: bool = False) -> set[str]:
    if VOCAB_PATH.exists() and not rebuild:
        return set(read_json(VOCAB_PATH)["tokens"])
    if passages is None:
        raise ValueError("no cached vocabulary; pass passages to build one")
    vocab = build_vocabulary(passages)
    write_json(VOCAB_PATH, {"min_count": VOCAB_MIN_COUNT, "size": len(vocab),
                            "tokens": sorted(vocab)})
    return vocab


def prepare(size: int | None = None, *, rebuild_vocab: bool = False) -> list[dict]:
    """Load one index and install the corpus vocabulary into the stemmer."""
    from ..data.index_build import load_index

    size = size or CFG.headline_index
    passages = load_index(size)
    vocab = load_or_build_vocabulary(passages, rebuild=rebuild_vocab)
    set_vocabulary(vocab)
    print(f"  index {size // 1000}k: {len(passages):,d} passages, "
          f"stemmer vocabulary {vocabulary_size():,d} tokens")
    return passages


# --------------------------------------------------------------------------- #
# Persistence                                                                  #
# --------------------------------------------------------------------------- #

CACHE = PROCESSED / "retrievers"


def save(retriever, size: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{retriever.name}_{size // 1000}k.pkl"
    with open(path, "wb") as fh:
        pickle.dump(retriever, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(name: str, size: int):
    path = CACHE / f"{name}_{size // 1000}k.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)
