"""Synonym-lexicon precision on the hand-built probe (PLAN.md task T6b "Done when").

    python -m bnqa.eval.lexicon_probe

Precision is the number that matters.  A **false** synonym silently inflates
tier-3 F1 — it lets a wrong answer score as a paraphrase — whereas a *missing*
synonym only costs a point of recall.  So the probe is weighted toward hard
negatives (co-hyponyms, antonyms, numerals), and the reviewed and unreviewed
halves of the lexicon are scored separately: PLAN.md §7.4 excludes unreviewed
pairs from headline metrics, and this table is the evidence for that decision.
"""

from __future__ import annotations

from pathlib import Path

from ..config import RESOURCES
from ..resources.loader import (equivalent, load_synonyms, load_variants,
                                synonym_stats, variant_kinds)
from ..utils import log_result, save_table


def load_probe() -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    path = RESOURCES / "synonym_probe.tsv"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                rows.append((parts[0].strip(), parts[1].strip(), int(parts[2])))
    return rows


def score(include_unreviewed: bool) -> dict:
    probe = load_probe()
    tp = fp = tn = fn = 0
    mistakes: list[dict] = []
    for a, b, label in probe:
        pred = equivalent(a, b, include_unreviewed=include_unreviewed)
        if label == 1 and pred:
            tp += 1
        elif label == 1 and not pred:
            fn += 1
            mistakes.append({"a": a, "b": b, "gold": 1, "pred": 0, "kind": "missed"})
        elif label == 0 and pred:
            fp += 1
            mistakes.append({"a": a, "b": b, "gold": 0, "pred": 1, "kind": "false_synonym"})
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"variant": "with_unreviewed" if include_unreviewed else "reviewed_only",
            "probe_pairs": len(probe), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "mistakes": mistakes}


def main() -> dict:
    stats = synonym_stats()
    print("T6b Lexical resources")
    print(f"  synonyms : {stats['pairs_total']} pairs "
          f"({stats['pairs_reviewed']} reviewed, {stats['pairs_unreviewed']} unreviewed) "
          f"-> {stats['classes']} classes over {stats['terms_in_classes']} terms")
    print(f"  variants : {sum(variant_kinds().values())} pairs {variant_kinds()}")
    print(f"  suffixes : {len(open(RESOURCES / 'suffixes.txt', encoding='utf-8').readlines())} lines")

    rows: list[dict] = []
    for include in (False, True):
        r = score(include)
        rows.append({k: v for k, v in r.items() if k != "mistakes"})
        print(f"  probe [{r['variant']:16s}] P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f}  (tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']})")
        for m in r["mistakes"][:6]:
            print(f"      {m['kind']:14s} {m['a']} ~ {m['b']}")

    save_table("synonym_probe", rows)
    payload = {"lexicon": stats, "variant_kinds": variant_kinds(),
               "probe": {r["variant"]: r for r in rows}}
    log_result("t6b_resources", "probe", payload)
    return payload


if __name__ == "__main__":
    main()
