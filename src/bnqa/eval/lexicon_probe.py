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

A note on circularity, because an examiner will press exactly here: pairs
copied out of the lexicon make recall 1.0 by construction.  Held-out pairs —
genuine Bangla doublets the lexicon has never seen — are therefore marked in
the probe file and reported as a separate row, and that row is the honest one.
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
    held = [r for r in rows if r[3]]
    seen = [r for r in rows if not r[3]]

    table = [
        {"subset": "all", **score(rows)},
        {"subset": "held_out", **score(held)},
        {"subset": "in_lexicon", **score(seen)},
    ]
    for row in table:
        print(f"  {row['subset']:12s} n={row['n']:3d}  P={row['precision']:.3f} "
              f"R={row['recall']:.3f}  F1={row['f1']:.3f}  "
              f"(tp {row['tp']}, fp {row['fp']}, fn {row['fn']}, tn {row['tn']})")
    print("  held_out is the honest row: pairs copied out of the lexicon make "
          "recall 1.0 by construction.")

    stats = synonym_stats()
    print(f"  lexicon: {stats['pairs_reviewed']:,d} reviewed pairs, "
          f"{stats['classes']:,d} classes, {stats['terms_in_classes']:,d} terms; "
          f"variants by kind: {variant_kinds()}")

    save_table("synonym_probe", table)
    payload = {"probe": table, "lexicon": stats, "variant_kinds": variant_kinds()}
    log_result("t6b_lexicon", "probe", payload)
    return payload


if __name__ == "__main__":
    build()
