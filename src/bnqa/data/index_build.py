"""Nested index assembly at 10k and 50k (PLAN.md §6.2B, task T3).

Index size is an **experiment**, not a setting.  More passages means more
distractors, so Recall@k *falls* as the index grows — a bigger index cannot
flatter the numbers.  Rather than pick a size arbitrarily we measure the effect,
which needs the two index sets to be **nested**: the 50k set is a strict
superset of the 10k set, and **all 3,000 BanglaRQA gold contexts are present at
both sizes**.  Anything else would confound size with composition.

Headline retrieval numbers are reported at 50k (PLAN.md §16.1); the 10k figure
is shown beside them and never quoted alone, because it flatters the system.
"""

from __future__ import annotations

import random
from collections import Counter

from ..config import CFG, INDEX_MANIFEST, PASSAGES, PROCESSED, config_hash, ensure_dirs
from ..utils import load_jsonl, log_result, save_table, set_seed, write_json, write_jsonl
from .dedup import deduplicate


def _gold_passages() -> list[dict]:
    rows = load_jsonl(PROCESSED / "gold_contexts.jsonl")
    return [{"pid": r["pid"], "text": r["text"], "source": "gold",
             "grade": None, "grade_label": None, "subject": None, "subject_bn": None,
             "chapter_no": None, "chapter_title": None, "title": None} for r in rows]


def build() -> dict:
    ensure_dirs()
    set_seed()
    rng = random.Random(CFG.seed)
    print("T3  Index assembly (nested 10k / 50k)")

    gold = _gold_passages()
    wiki = load_jsonl(PROCESSED / "wiki_passages.jsonl")
    nctb_all = load_jsonl(PROCESSED / "nctb_passages.jsonl")
    # Prefer SchoolText for the index: it is the only source carrying the
    # grade/subject/chapter metadata that VC-2's citation line needs.
    nctb = [p for p in nctb_all if p["source"] == "nctb_schooltext"]
    nctb_fallback = [p for p in nctb_all if p["source"] != "nctb_schooltext"]
    print(f"  candidates: {len(gold):,d} gold, {len(wiki):,d} wiki, "
          f"{len(nctb):,d} nctb-schooltext ({len(nctb_fallback):,d} textbook held in reserve)")

    protect = {p["pid"] for p in gold}
    kept, stats = deduplicate(gold + wiki + nctb + nctb_fallback, protect=protect)
    print(f"  dedup: {stats['seen']:,d} -> {stats['kept']:,d} "
          f"({stats['exact_duplicate']:,d} exact, {stats['near_duplicate']:,d} near @ "
          f"Jaccard {CFG.dedup_jaccard})")
    print(f"    all {stats['protected_kept']:,d}/{len(gold):,d} gold passages survived "
          f"{'(required)' if stats['protected_kept'] == len(gold) else '<-- PROBLEM'}")
    if stats["protected_kept"] != len(gold):
        raise RuntimeError("a gold passage was removed by dedup - Recall@k would be wrong")

    pool: dict[str, list[dict]] = {"gold": [], "wiki": [], "nctb": [], "_nctb_reserve": []}
    for p in kept:
        if p["source"] == "gold":
            pool["gold"].append(p)
        elif p["source"] == "wiki":
            pool["wiki"].append(p)
        elif p["source"] == "nctb_schooltext":
            pool["nctb"].append(p)
        else:
            pool["_nctb_reserve"].append(p)

    for k in ("wiki", "nctb", "_nctb_reserve"):
        rng.shuffle(pool[k])
    pool["gold"].sort(key=lambda p: p["pid"])
    # SchoolText first, Bangla-TextBook only to top up: SchoolText is the only
    # source carrying grade/subject/chapter, and a passage without that metadata
    # cannot produce the citation line VC-2 promises.  Shuffling the two together
    # would silently fill the index with uncitable passages.
    pool["nctb"] = pool["nctb"] + pool.pop("_nctb_reserve")

    sizes = sorted(CFG.index_sizes)
    biggest = sizes[-1]
    want = CFG.index_composition[biggest]
    selected: dict[str, list[dict]] = {}
    for stratum, n in want.items():
        have = pool[stratum]
        if len(have) < n:
            print(f"    ! only {len(have):,d} {stratum} passages available, wanted {n:,d}")
        selected[stratum] = have[:n]

    universe = selected["gold"] + selected["wiki"] + selected["nctb"]
    n = write_jsonl(PASSAGES, universe)
    print(f"  corpus universe: {n:,d} passages -> {PASSAGES.name}")

    manifests: list[dict] = []
    for size in sizes:
        comp = CFG.index_composition[size]
        pids: list[str] = []
        for stratum in ("gold", "wiki", "nctb"):
            pids.extend(p["pid"] for p in selected[stratum][: comp[stratum]])
        manifest = {
            "size": size, "requested": size, "actual": len(pids),
            "composition": comp, "nested_in": biggest if size != biggest else None,
            "gold_passages": comp["gold"], "config_hash": config_hash(), "seed": CFG.seed,
            "pids": pids,
        }
        write_json(INDEX_MANIFEST[size], manifest)
        gold_here = sum(1 for p in pids if p.startswith("bn_wiki_"))
        print(f"  index {size // 1000:2d}k: {len(pids):,d} pids "
              f"(gold {comp['gold']:,d}, wiki {comp['wiki']:,d}, nctb {comp['nctb']:,d})")
        manifests.append({"size": size, "actual": len(pids), **comp,
                          "gold_present": gold_here})

    # nesting check — the whole size comparison depends on it
    small = set(load_manifest(sizes[0])["pids"])
    large = set(load_manifest(sizes[-1])["pids"])
    nested = small <= large
    print(f"  nesting: {sizes[0] // 1000}k ⊂ {sizes[-1] // 1000}k = {nested} "
          f"{'(required for the size comparison)' if nested else '<-- PROBLEM'}")
    if not nested:
        raise RuntimeError("index sets are not nested - the size comparison would be confounded")

    save_table("index_manifest", manifests)
    payload = {
        "universe": n,
        "dedup": {k: v for k, v in stats.items() if not k.startswith(("exact_dup_", "near_dup_"))},
        "dedup_by_source": {k: v for k, v in stats.items()
                            if k.startswith(("exact_dup_", "near_dup_"))},
        "sizes": {str(s): CFG.index_composition[s] for s in sizes},
        "nested": nested,
        "gold_at_every_size": True,
    }
    log_result("t3_index", "assembly", payload)
    return payload


def load_manifest(size: int) -> dict:
    from ..utils import read_json

    return read_json(INDEX_MANIFEST[size])


def load_index(size: int) -> list[dict]:
    """Passages of one index, in manifest order."""
    wanted = load_manifest(size)["pids"]
    by_pid = {p["pid"]: p for p in load_jsonl(PASSAGES)}
    return [by_pid[p] for p in wanted if p in by_pid]


if __name__ == "__main__":
    build()
