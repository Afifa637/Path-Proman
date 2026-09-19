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

    artefact = length_artefact(passages)

    save_table("corpus_audit", rows)
    payload = {"total_passages": len(passages), "by_source": {r["source"]: r for r in rows},
               "gold_coverage": coverage, "sample_per_source": SAMPLE,
               "length_artefact": artefact}
    log_result("t3_index", "audit", payload)
    return payload


def length_artefact(passages: list[dict]) -> dict:
    """Can passage **length alone** tell a gold passage from a distractor?

    This is the distribution artefact PLAN.md §6.2B warns about, and it is not
    cosmetic.  Chunking Wikipedia to a fixed 120-180 tokens while leaving
    BanglaRQA contexts whole once made the two trivially separable at
    **AUC 0.795** — and BM25's ``b`` tunes length normalisation directly, so it
    tuned to ``b=0.3`` and quietly learned "long documents are the answers".

    The fix was to sample each distractor's target length from the gold
    distribution.  This check keeps the fix honest: it re-measures the AUC on
    every build, so a regression in the chunker shows up as a number rather
    than as a suspiciously good retrieval score.  Chance is 0.5.
    """
    from sklearn.metrics import roc_auc_score

    lengths = np.array([len(tokenize(p["text"], do_normalize=False)) for p in passages],
                       dtype=float)
    is_gold = np.array([p["source"] == "gold" for p in passages], dtype=int)
    if is_gold.sum() == 0 or is_gold.sum() == len(is_gold):
        return {"auc": None, "note": "only one class present"}

    def _auc(mask) -> float:
        y = is_gold[mask]
        if y.min() == y.max():
            return float("nan")
        # Direction is irrelevant: separability is separability either way.
        a = float(roc_auc_score(y, lengths[mask]))
        return max(a, 1.0 - a)

    overall = _auc(np.ones(len(is_gold), dtype=bool))
    sources = np.array([p["source"] for p in passages])
    per_source = {}
    for src in sorted(set(sources) - {"gold"}):
        per_source[src] = round(_auc((sources == src) | (sources == "gold")), 4)

    # The distractor pool we *control* is Wikipedia: its chunk lengths are
    # sampled from the gold distribution precisely so this number is chance.
    # NCTB-SchoolText keeps its own chunking (T3 requires it — that chunking is
    # what carries the chapter metadata VC-2 cites), and its passages are
    # natively much shorter, so it separates on length and there is no fix for
    # that which does not also destroy the citation unit.  Both numbers are
    # reported; the wiki one is the one the fix owns.
    wiki_auc = per_source.get("wiki", float("nan"))
    clean = wiki_auc == wiki_auc and wiki_auc < 0.60
    verdict = ("wiki distractors are length-matched (chance); NCTB is shorter by "
               "construction and is reported separately" if clean else
               "SEPARABLE - the length-matched distractor sampling has regressed")
    print(f"  length artefact: is-gold from length alone — overall AUC {overall:.3f}; "
          f"per distractor source {per_source}")
    print(f"    {verdict}")
    return {"auc_overall": round(overall, 4), "auc_by_source": per_source,
            "auc_vs_wiki": wiki_auc, "chance": 0.5, "threshold": 0.60,
            "clean": bool(clean), "verdict": verdict}


if __name__ == "__main__":
    audit()
