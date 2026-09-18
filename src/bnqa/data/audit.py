"""Corpus audit → ``reports/tables/corpus_audit.csv`` (PLAN.md §6.3 step 7, task T3).

Reports what is actually in the index rather than what we hoped would be: per
source counts, passage-length distribution, Bengali-script validity over a
sample, and the gold-coverage check that the retrieval metrics depend on.

    python -m bnqa.data.audit
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from ..config import CFG, PASSAGES
from ..preprocess.normalize import bengali_ratio
from ..preprocess.tokenize import tokenize
from ..utils import load_jsonl, log_result, save_table, set_seed
from .index_build import load_manifest

SAMPLE = 200


def audit() -> dict:
    set_seed()
    passages = load_jsonl(PASSAGES)
    print(f"Corpus audit over {len(passages):,d} passages")

    by_source: dict[str, list[dict]] = {}
    for p in passages:
        by_source.setdefault(p["source"], []).append(p)

    rows: list[dict] = []
    for source, group in sorted(by_source.items()):
        chars = np.array([len(p["text"]) for p in group])
        toks = np.array([len(tokenize(p["text"], do_normalize=False)) for p in group])
        sample = random.Random(CFG.seed).sample(group, min(SAMPLE, len(group)))
        ratios = np.array([bengali_ratio(p["text"]) for p in sample])
        with_citation = sum(1 for p in group if p.get("grade_label") and p.get("chapter_title"))
        rows.append({
            "source": source,
            "passages": len(group),
            "chars_median": int(np.median(chars)),
            "chars_p10": int(np.percentile(chars, 10)),
            "chars_p90": int(np.percentile(chars, 90)),
            "tokens_median": int(np.median(toks)),
            "tokens_p90": int(np.percentile(toks, 90)),
            "bengali_ratio_mean": round(float(ratios.mean()), 4),
            "bengali_ratio_min": round(float(ratios.min()), 4),
            "sample_checked": len(sample),
            "with_citation_metadata": with_citation,
        })
        print(f"  {source:16s} {len(group):6,d} passages  median {int(np.median(chars)):4d} chars "
              f"/ {int(np.median(toks)):3d} tokens  bengali {ratios.mean():.3f}  "
              f"citable {with_citation:,d}")

    # gold coverage — every question must be able to find its passage
    pids = {p["pid"] for p in passages}
    coverage: dict[str, dict] = {}
    for size in sorted(CFG.index_sizes):
        man = set(load_manifest(size)["pids"])
        gold_in = sum(1 for p in passages if p["source"] == "gold" and p["pid"] in man)
        coverage[str(size)] = {"index_size": len(man), "gold_present": gold_in}
        print(f"  index {size // 1000:2d}k: {gold_in:,d} gold passages present "
              f"{'OK' if gold_in == CFG.index_composition[size]['gold'] else 'MISSING'}")

    save_table("corpus_audit", rows)
    payload = {"total_passages": len(passages), "by_source": {r["source"]: r for r in rows},
               "gold_coverage": coverage, "sample_per_source": SAMPLE}
    log_result("t3_index", "audit", payload)
    return payload


if __name__ == "__main__":
    audit()
