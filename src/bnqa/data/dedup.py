"""Near-duplicate removal across sources (PLAN.md §6.3 step 5, task T3).

This is a **correctness** step, not hygiene.  Two failure modes it prevents:

* a Wikipedia chunk that is a near-copy of a gold context would sit in the index
  as a "distractor" that is actually a second correct answer, so Recall@k would
  be measured against the wrong ground truth;
* near-duplicate leakage between the index and Tier B's ``train_corpus.txt``
  would quietly inflate every retrieval number.

Exact duplicates go first by sha1 of the normalised text (cheap), then MinHash /
Jaccard at 0.85 for the rest.

**Gold passages are protected.**  They are inserted into the LSH index first and
are never dropped, because PLAN.md §6.2B requires all 3,000 to be present at
every index size — a gold passage removed as a "duplicate" would make its
question unanswerable and silently depress Recall@k.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from datasketch import MinHash, MinHashLSH

from ..config import CFG
from ..preprocess.normalize import normalize_for_match

from ..utils import sha1_text


SHINGLE = 3  # word n-gram width


def _shingles(text: str) -> list[bytes]:
    """Word 3-shingles.

    Char 5-grams are the more obvious choice for a morphologically rich language,
    but they are ~7x more numerous per passage and MinHash cost is linear in the
    number of shingles: on this corpus the char version needed hundreds of
    millions of hash updates and did not finish.  Word 3-shingles give the same
    near-duplicate signal at a fraction of the cost — two passages that share
    long word runs are duplicates, which is exactly what we are detecting — and
    the inflection-robustness that char n-grams buy is irrelevant here, because
    a near-duplicate is a *copy*, not a paraphrase.
    """
    words = text.split()
    if len(words) < SHINGLE:
        return [text.encode("utf-8")] if text else []
    return [" ".join(words[i:i + SHINGLE]).encode("utf-8")
            for i in range(len(words) - SHINGLE + 1)]


def _signature(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    grams = set(_shingles(text))
    if grams:
        m.update_batch(list(grams))
    return m


def deduplicate(passages: Iterable[dict], *, threshold: float | None = None,
                num_perm: int | None = None, protect: set[str] | None = None,
                ) -> tuple[list[dict], Counter]:
    """Return ``(kept, stats)``.  ``protect`` holds pids that must survive."""
    threshold = CFG.dedup_jaccard if threshold is None else threshold
    num_perm = CFG.dedup_num_perm if num_perm is None else num_perm
    protect = protect or set()

    passages = list(passages)
    # protected first, so they claim their LSH buckets and can never be the one dropped
    passages.sort(key=lambda p: 0 if p["pid"] in protect else 1)

    stats: Counter = Counter()
    seen_exact: dict[str, str] = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []

    for p in passages:
        stats["seen"] += 1
        is_protected = p["pid"] in protect
        norm = normalize_for_match(p["text"])
        if not norm:
            stats["empty"] += 1
            if not is_protected:
                continue

        key = sha1_text(norm)
        # A protected passage survives BOTH duplicate checks.  BanglaRQA does
        # contain a few contexts that are byte-identical under different
        # passage_ids; dropping either one would leave its questions pointing at
        # a pid that is not in the index, so Recall@k would be measured against
        # a target that cannot be retrieved.
        if key in seen_exact:
            stats["exact_duplicate"] += 1
            stats[f"exact_dup_{p['source']}"] += 1
            if is_protected:
                stats["protected_exact_duplicate"] += 1
            else:
                continue
        else:
            seen_exact[key] = p["pid"]

        sig = _signature(norm, num_perm)
        if lsh.query(sig):
            if is_protected:
                stats["protected_near_duplicate"] += 1
            else:
                stats["near_duplicate"] += 1
                stats[f"near_dup_{p['source']}"] += 1
                continue

        try:
            lsh.insert(p["pid"], sig)
        except ValueError:  # pid already inserted
            stats["pid_collision"] += 1
            if not is_protected:
                continue
        kept.append(p)
        stats["kept"] += 1

    stats["protected_kept"] = sum(1 for p in kept if p["pid"] in protect)
    return kept, stats
