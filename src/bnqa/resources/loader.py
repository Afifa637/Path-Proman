"""Loader for the T6b lexical resources (PLAN.md §7.4).

Three files, three jobs:

* ``suffixes.txt``  — the stemmer and (Tier B) MorphSpan-MLM;
* ``synonyms.tsv``  — tier-3 stem+synonym F1, query expansion (T7), and the
  grading rubric's concept collapsing (§12.3);
* ``variants.tsv``  — the same term written differently.

Synonym classes are closed under symmetry and transitivity, so ``বায়ুমণ্ডল`` and
``আবহমণ্ডল`` land in one class and any member expands to all of them.

**Unreviewed pairs are excluded by default.**  ``reviewed=0`` marks a pair a
human has not checked; embedding neighbourhoods and quick hand lists both put
antonyms and co-hyponyms close together, and letting those into tier-3 F1 would
inflate it in exactly the way PLAN.md §9's second guard warns about.  They are
kept in the file, reported separately in the probe table, and only enabled by
an explicit ``include_unreviewed=True``.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from ..config import RESOURCES


def _read_tsv(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append(line.split("\t"))
    return rows


# --------------------------------------------------------------------------- #
# Synonyms                                                                     #
# --------------------------------------------------------------------------- #


class SynonymLexicon:
    """Union-find over synonym pairs, exposed as classes."""

    def __init__(self, pairs: list[tuple[str, str, bool, str]]) -> None:
        self.pairs = pairs
        self._parent: dict[str, str] = {}
        self.domain: dict[str, str] = {}
        for a, b, reviewed, domain in pairs:
            if not reviewed:
                continue
            self._union(a, b)
            self.domain.setdefault(a, domain)
            self.domain.setdefault(b, domain)

    # -- union-find ---------------------------------------------------------
    def _find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def _union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra

    # -- api ---------------------------------------------------------------
    def expand(self, term: str) -> set[str]:
        """All terms in ``term``'s class (including itself)."""
        term = term.strip()
        if term not in self._parent:
            return {term}
        root = self._find(term)
        return {t for t in self._parent if self._find(t) == root} | {term}

    def same_class(self, a: str, b: str) -> bool:
        a, b = a.strip(), b.strip()
        if a == b:
            return True
        if a not in self._parent or b not in self._parent:
            return False
        return self._find(a) == self._find(b)

    def classes(self) -> list[set[str]]:
        groups: defaultdict[str, set[str]] = defaultdict(set)
        for t in self._parent:
            groups[self._find(t)].add(t)
        return list(groups.values())

    def __len__(self) -> int:
        return len(self._parent)


@lru_cache(maxsize=4)
def load_synonyms(include_unreviewed: bool = False) -> SynonymLexicon:
    pairs: list[tuple[str, str, bool, str]] = []
    for row in _read_tsv(RESOURCES / "synonyms.tsv"):
        if len(row) < 2:
            continue
        term, syn = row[0].strip(), row[1].strip()
        reviewed = (row[2].strip() == "1") if len(row) > 2 else False
        domain = row[3].strip() if len(row) > 3 else ""
        pairs.append((term, syn, reviewed or include_unreviewed, domain))
    return SynonymLexicon(pairs)


def synonym_stats() -> dict:
    raw = _read_tsv(RESOURCES / "synonyms.tsv")
    reviewed = sum(1 for r in raw if len(r) > 2 and r[2].strip() == "1")
    lex = load_synonyms()
    return {
        "pairs_total": len(raw),
        "pairs_reviewed": reviewed,
        "pairs_unreviewed": len(raw) - reviewed,
        "terms_in_classes": len(lex),
        "classes": len(lex.classes()),
        "domains": sorted({r[3].strip() for r in raw if len(r) > 3}),
    }


# --------------------------------------------------------------------------- #
# Variants                                                                     #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def load_variants() -> dict[str, set[str]]:
    """Bidirectional map from a surface form to its other spellings."""
    out: defaultdict[str, set[str]] = defaultdict(set)
    for row in _read_tsv(RESOURCES / "variants.tsv"):
        if len(row) < 2:
            continue
        a, b = row[0].strip(), row[1].strip()
        if a == b:
            continue
        out[a].add(b)
        out[b].add(a)
        out[a.lower()].add(b)
        out[b].add(a.lower())
    return dict(out)


def variant_kinds() -> dict[str, int]:
    kinds: defaultdict[str, int] = defaultdict(int)
    for row in _read_tsv(RESOURCES / "variants.tsv"):
        if len(row) > 2:
            kinds[row[2].strip()] += 1
    return dict(kinds)


# --------------------------------------------------------------------------- #
# Combined expansion                                                           #
# --------------------------------------------------------------------------- #


def expand_term(term: str, *, include_unreviewed: bool = False) -> set[str]:
    """Synonym class ∪ surface variants — the expansion used by T7 and tier-3 F1."""
    out = load_synonyms(include_unreviewed).expand(term)
    variants = load_variants()
    for t in list(out):
        out |= variants.get(t, set())
    out |= variants.get(term, set())
    return out


def equivalent(a: str, b: str, *, include_unreviewed: bool = False) -> bool:
    """True when two surface strings denote the same thing."""
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    return b in expand_term(a, include_unreviewed=include_unreviewed)
