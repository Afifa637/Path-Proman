"""T5 + T7: fit every sparse arm, tune on val, evaluate at both index sizes.

    python scripts/run_retrieval.py                              # full run
    python scripts/run_retrieval.py --size 10000 --queries 300   # quick pass

Produces ``reports/tables/retrieval.csv`` — Recall@1/5/10, MRR@10, nDCG@10,
latency and index size for every arm at every index size, which is T7's
acceptance criterion and the evidence for RQ1.

Discipline (PLAN.md §16.1): **BM25's k1/b are tuned on val and nothing is tuned
on test.**  Headline numbers are the 50k ones; the 10k column sits beside them
and is never quoted alone, because a smaller index flatters the system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnqa.config import CFG  # noqa: E402
from bnqa.eval.retrieval_metrics import evaluate_retriever, load_queries  # noqa: E402
from bnqa.retrieval.bm25 import BM25Retriever  # noqa: E402
from bnqa.retrieval.hybrid import ExpandedQuery, RRFHybrid  # noqa: E402
from bnqa.retrieval.index import prepare  # noqa: E402
from bnqa.retrieval.tfidf import TfidfRetriever  # noqa: E402
from bnqa.utils import log_result, save_table, set_seed  # noqa: E402

K1_GRID = (0.9, 1.2, 1.5, 2.0)
B_GRID = (0.3, 0.5, 0.75, 0.9)


def tune_bm25(passages, queries, *, verbose: bool = True) -> tuple[float, float, list[dict]]:
    """Grid-search k1/b on **val**.  Never touches test."""
    rows: list[dict] = []
    best = None
    # One build, then re-weight per grid point: k1 and b do not change the term
    # counts, only how they are normalised.
    r = BM25Retriever(retain_counts=True).build(passages)
    for k1 in K1_GRID:
        for b in B_GRID:
            r.set_params(k1, b)
            m = evaluate_retriever(r, queries)
            rows.append({"k1": k1, "b": b, "recall@5": round(m["recall@5"], 4),
                         "mrr@10": round(m["mrr@10"], 4)})
            if best is None or m["recall@5"] > best[2]:
                best = (k1, b, m["recall@5"])
    assert best is not None
    if verbose:
        default = [r for r in rows if r["k1"] == CFG.bm25_k1 and r["b"] == CFG.bm25_b]
        note = f"; default {CFG.bm25_k1}/{CFG.bm25_b} scored {default[0]['recall@5']:.4f}" if default else ""
        print(f"  BM25 tuned on val: k1={best[0]}, b={best[1]} "
              f"(Recall@5 {best[2]:.4f}{note})")
    return best[0], best[1], rows


def arms(passages, k1: float, b: float):
    """Every Tier-A retrieval arm, all behind the same Retriever protocol."""
    bm25 = BM25Retriever(k1=k1, b=b).build(passages)
    word = TfidfRetriever("word").build(passages)
    char = TfidfRetriever("char").build(passages)
    hybrid_bc = RRFHybrid([bm25, char], name="rrf(bm25+char)").build(passages)
    hybrid = RRFHybrid([bm25, word, char]).build(passages)
    qe = ExpandedQuery(BM25Retriever(k1=k1, b=b).build(passages)).build(passages)
    return [bm25, word, char, hybrid_bc, hybrid, qe]


def control(size: int, val: list[dict]) -> list[dict]:
    """The length-artefact control: gold + Wikipedia only, no NCTB.

    The corpus audit measures that passage length separates gold from NCTB
    passages at AUC ~0.97, while the length-matched Wikipedia distractors sit
    at chance.  NCTB text is short because we keep its own chunking — that
    chunking is what carries the chapter metadata VC-2 cites — so the artefact
    cannot be removed without destroying the citation unit.

    What *can* be done is measure what it is worth.  This arm re-runs BM25
    over the same index with the NCTB passages removed, so the headline number
    can be read beside a retrieval task whose distractors are known to be
    distributionally matched.  A large gap means the headline is partly
    measuring our own index construction, and the report says so.
    """
    from bnqa.retrieval.index import prepare as _prepare

    passages = _prepare(size, verbose=False)
    subset = [p for p in passages if p.get("source") != "nctb_schooltext"]
    print(f"  control index: {len(subset):,d} passages (gold + wiki only, "
          f"{len(passages) - len(subset):,d} NCTB removed)")

    rows: list[dict] = []
    for label, pool in (("full", passages), ("gold+wiki", subset)):
        r = BM25Retriever(CFG.bm25_k1, CFG.bm25_b).build(pool)
        m = evaluate_retriever(r, val, with_latency=False)
        m.update({"arm": f"bm25/{label}", "index_size": size,
                  "passages": len(pool), "bm25_k1": CFG.bm25_k1, "bm25_b": CFG.bm25_b})
        rows.append(m)
        print(f"    {m['arm']:18s} R@5={m['recall@5']:.4f}  MRR={m['mrr@10']:.4f}")
    if len(rows) == 2:
        delta = rows[1]["recall@5"] - rows[0]["recall@5"]
        print(f"    removing the distributionally-distinct NCTB distractors moves "
              f"Recall@5 by {delta:+.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None, help="one index size only")
    ap.add_argument("--queries", type=int, default=None, help="cap val queries (quick runs)")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--control-only", action="store_true",
                    help="run only the length-artefact control arm")
    args = ap.parse_args()

    set_seed()
    if args.control_only:
        val = load_queries("val", limit=args.queries)
        rows = control(args.size or CFG.headline_index, val)
        save_table("retrieval_length_control", rows)
        log_result("t7_retrieval", "length_control", {"rows": rows})
        return
    sizes = [args.size] if args.size else sorted(CFG.index_sizes)
    val = load_queries("val", limit=args.queries)
    print(f"T5/T7  sparse retrieval - {len(val):,d} answerable val questions\n")

    table: list[dict] = []
    for size in sizes:
        print(f"=== index {size // 1000}k " + "=" * 46)
        passages = prepare(size)

        if args.no_tune:
            k1, b, grid = CFG.bm25_k1, CFG.bm25_b, []
        else:
            k1, b, grid = tune_bm25(passages, val)
            save_table(f"bm25_tuning_{size // 1000}k", grid)

        for r in arms(passages, k1, b):
            m = evaluate_retriever(r, val)
            m["index_size"] = size
            m["split"] = "val"
            m["bm25_k1"] = k1
            m["bm25_b"] = b
            # Query-expansion coverage arrives via the arm's own counters
            # (evaluate_retriever snapshots them).  It is the number that
            # explains this row: the T6b lexicon is curriculum-domain and
            # hand-authored, BanglaRQA questions are Wikipedia-domain, so
            # expansion can only fire where the two vocabularies overlap.
            # Reporting that overlap turns "query expansion did nothing" into a
            # measured explanation, and sets the baseline Tier B's induced
            # synonyms (X3) have to beat.
            table.append(m)
            print(f"  {m['retriever']:22s} R@1={m['recall@1']:.3f} R@5={m['recall@5']:.3f} "
                  f"R@10={m['recall@10']:.3f} MRR={m['mrr@10']:.3f} nDCG={m['ndcg@10']:.3f} "
                  f"| {m['ms_per_query']:7.2f} ms/q {m['index_mb']:7.1f} MB")
        print()

    save_table("retrieval", table)

    # the index-size comparison, which is the point of building nested indexes
    if len(sizes) > 1:
        print("Index-size effect on Recall@5 (PLAN.md §6.2B - bigger index, harder task):")
        by_arm: dict[str, dict[int, float]] = {}
        for row in table:
            by_arm.setdefault(row["retriever"], {})[row["index_size"]] = row["recall@5"]
        comp: list[dict] = []
        lo, hi = min(sizes), max(sizes)
        for arm, vals in by_arm.items():
            if lo in vals and hi in vals:
                delta = vals[hi] - vals[lo]
                rel = round(-100 * delta / vals[lo], 2) if vals[lo] else None
                comp.append({"retriever": arm, f"recall@5_{lo // 1000}k": round(vals[lo], 4),
                             f"recall@5_{hi // 1000}k": round(vals[hi], 4),
                             "delta": round(delta, 4), "relative_drop_pct": rel})
                print(f"  {arm:22s} {vals[lo]:.3f} -> {vals[hi]:.3f}  "
                      f"({delta:+.3f}, {rel:.1f}% relative drop)")
        save_table("retrieval_index_size", comp)

    headline = [r for r in table if r["index_size"] == CFG.headline_index]
    if headline:
        best = max(headline, key=lambda r: r["recall@5"])
        print(f"\nHeadline ({CFG.headline_index // 1000}k index, val): best arm = "
              f"{best['retriever']}  Recall@5 = {best['recall@5']:.4f}")
    log_result("t7_retrieval", "val", {"rows": table})


if __name__ == "__main__":
    main()
