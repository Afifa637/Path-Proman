"""T6b acceptance: precision and recall of the hand-authored lexicon.

PLAN.md §7.4 requires ``reports/tables/synonym_probe.csv`` to report **lexicon
precision on a hand-built probe set**.  Precision matters more than recall
here: a *false* synonym turns tier-3 F1 into nonsense by granting credit for a
wrong answer, whereas a *missing* one only costs a point.

The probe's negatives are deliberately the hard cases — co-hyponyms and
antonyms that any embedding space places close together, several of which are
the very pairs the veto layer exists to keep apart.  So a high score here is
also an early smoke test for §10.2, and it is the baseline Tier B's induced
synonyms (X3) are measured against on the *same* probe.

Two things an examiner will press on, both stated rather than smoothed over:

**Circularity.**  Pairs copied out of the lexicon make recall 1.0 by
construction, so the ``in_lexicon`` row is not evidence of anything.  Only the
*precision* number is meaningful on it — a false merge would show up there.

**The unseen row is a coverage bound, not a failure.**  A pair whose terms the
lexicon has never seen cannot match: ``equivalent`` is a union-find lookup, so
recall on those pairs is 0 **by definition**, and printing 0.000 as though the
lexicon were performing badly would be theatre.  What that row measures is how
much of the real synonym space a hand-authored list of ~56 pairs does not
reach — and that gap is precisely what Tier B's FastText-induced synonyms (X3)
have to close, scored on this same probe.  So it is reported as coverage and
labelled as such.
"""

from __future__ import annotations

from pathlib import Path

from ..config import RESOURCES
from ..resources.loader import (equivalent, load_synonyms, load_variants,
                                synonym_stats, variant_kinds)
from ..utils import log_result, save_table

PROBE = RESOURCES / "synonym_probe.tsv"


def load_probe(path: Path | None = None) -> list[tuple[str, str, int, bool]]:
    """``(a, b, label, held_out)`` rows.

    A pair is held out when neither side appears anywhere in ``synonyms.tsv`` —
    determined from the file rather than declared, so it cannot drift out of
    date when the lexicon grows.
    """
    path = path or PROBE
    lex = load_synonyms()
    variants = load_variants()
    known = set(lex.domain) | set(variants)
    for cls in lex.classes():
        known |= cls

    rows: list[tuple[str, str, int, bool]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            a, b, label = parts[0].strip(), parts[1].strip(), int(parts[2].strip())
            held_out = a not in known and b not in known
            rows.append((a, b, label, held_out))
    return rows


def score(rows: list[tuple[str, str, int, bool]]) -> dict:
    tp = fp = fn = tn = 0
    for a, b, label, _ in rows:
        pred = equivalent(a, b)
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round((tp + tn) / max(len(rows), 1), 4)}


def build() -> dict:
    print("T6b  lexical resource probe")
    rows = load_probe()
    unseen = [r for r in rows if r[3]]
    seen = [r for r in rows if not r[3]]

    table = [
        {"subset": "all", "note": "the headline: precision is the number that matters",
         **score(rows)},
        {"subset": "in_lexicon", "note": "recall is 1.0 by construction — read precision only",
         **score(seen)},
        {"subset": "unseen_pairs", "note": "coverage bound: unreachable by construction, "
                                           "and the gap Tier B (X3) must close",
         **score(unseen)},
    ]
    for row in table:
        print(f"  {row['subset']:13s} n={row['n']:3d}  P={row['precision']:.3f} "
              f"R={row['recall']:.3f}  F1={row['f1']:.3f}  "
              f"(tp {row['tp']}, fp {row['fp']}, fn {row['fn']}, tn {row['tn']})")
        print(f"                 {row['note']}")

    n_unseen_pos = sum(1 for _a, _b, label, _h in unseen if label)
    n_pos = sum(1 for _a, _b, label, _h in rows if label)
    print(f"  -> precision {table[0]['precision']:.3f} with {table[0]['fp']} false merges "
          f"across {table[0]['tn'] + table[0]['fp']} hard negatives (antonyms and "
          f"co-hyponyms the veto layer must also keep apart)")
    print(f"  -> recall {table[0]['recall']:.3f}; {n_unseen_pos}/{n_pos} positives are "
          f"terms the lexicon has never seen, so they are a coverage limit rather "
          f"than an error")

    stats = synonym_stats()
    print(f"  lexicon: {stats['pairs_reviewed']:,d} reviewed pairs, "
          f"{stats['classes']:,d} classes, {stats['terms_in_classes']:,d} terms; "
          f"variants by kind: {variant_kinds()}")

    save_table("synonym_probe", table)
    payload = {"probe": table, "lexicon": stats, "variant_kinds": variant_kinds(),
               "unseen_positives": n_unseen_pos, "positives": n_pos,
               "coverage_gap_is_tier_b_target": True}
    log_result("t6b_lexicon", "probe", payload)
    return payload


if __name__ == "__main__":
    build()
