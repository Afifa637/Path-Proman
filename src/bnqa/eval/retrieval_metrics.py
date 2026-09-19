"""Retrieval metrics: Recall@k, MRR@10, nDCG@10, latency, index size (T5/T7).

One gold passage per question (BanglaRQA's ``gold_passage_id``), so the graded
relevance of nDCG collapses to binary — it is reported anyway because it is the
standard companion to MRR and because Tier B's rerankers (X8) will be compared
on it.

**Only answerable questions are scored.**  An unanswerable question has no gold
passage, so including it would put an undefined value in the denominator of
every recall figure.  Unanswerable items are the *verifier's* evaluation set
(§10), not the retriever's, and they are counted and reported separately so the
exclusion is visible rather than silent.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..config import CFG, QA_TEST, QA_TRAIN, QA_VAL
from ..retrieval.base import BaseRetriever
from ..retrieval.index import measure_latency
from ..utils import load_jsonl

SPLIT_PATH = {"train": QA_TRAIN, "val": QA_VAL, "test": QA_TEST}


# --------------------------------------------------------------------------- #
# Query loading                                                                #
# --------------------------------------------------------------------------- #


def load_queries(split: str, *, limit: int | None = None,
                 answerable_only: bool = True) -> list[dict]:
    """``{qid, question, gold_passage_id}`` rows for one split."""
    rows = load_jsonl(SPLIT_PATH[split])
    if answerable_only:
        rows = [r for r in rows if r.get("is_answerable")]
    if limit:
        rows = rows[:limit]
    return rows


def unanswerable_count(split: str) -> int:
    return sum(1 for r in load_jsonl(SPLIT_PATH[split]) if not r.get("is_answerable"))


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #


def reciprocal_rank(ranked: Sequence[str], gold: str) -> float:
    for i, pid in enumerate(ranked, start=1):
        if pid == gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: str, k: int = 10) -> float:
    """Binary-relevance nDCG.  IDCG is 1 because there is exactly one gold."""
    for i, pid in enumerate(ranked[:k], start=1):
        if pid == gold:
            return 1.0 / math.log2(i + 1)
    return 0.0


def evaluate_retriever(retriever: BaseRetriever, queries: Sequence[dict], *,
                       k: int | None = None, with_latency: bool = True,
                       latency_sample: int = 200) -> dict:
    """Recall@1/5/10, MRR@10, nDCG@10 (+ latency and index size) for one arm."""
    k = max(CFG.recall_at) if k is None else k
    depth = max(k, 10)

    # Arms that count their own behaviour (query expansion) are reset here, so
    # the coverage they report is over *this* query set.  It is also snapshotted
    # before the latency loop below, which would otherwise re-run 200 queries
    # and inflate the denominator past 100%.
    if hasattr(retriever, "reset_counters"):
        retriever.reset_counters()

    hits = {at: 0 for at in CFG.recall_at}
    mrr = ndcg = 0.0
    found = 0
    for q in queries:
        ranked = [pid for pid, _ in retriever.search(q["question"], depth)]
        gold = q["gold_passage_id"]
        for at in CFG.recall_at:
            if gold in ranked[:at]:
                hits[at] += 1
        rr = reciprocal_rank(ranked[:10], gold)
        mrr += rr
        ndcg += ndcg_at_k(ranked, gold, 10)
        found += 1 if rr else 0

    n = max(len(queries), 1)
    out: dict = {"retriever": retriever.name, "n_queries": len(queries)}
    for at in CFG.recall_at:
        out[f"recall@{at}"] = hits[at] / n
    out["mrr@10"] = mrr / n
    out["ndcg@10"] = ndcg / n
    out["found@10"] = found / n

    if hasattr(retriever, "coverage"):
        out.update(retriever.coverage())

    if with_latency:
        sample = [q["question"] for q in queries[:latency_sample]]
        out["ms_per_query"] = round(measure_latency(retriever, sample, k=depth), 3)
        out["index_mb"] = round(retriever.size_mb(), 2)
        out["build_s"] = round(retriever.build_seconds, 2)
    return out


def bootstrap_ci(values: Sequence[float], *, resamples: int | None = None,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI — used wherever a headline number is quoted."""
    import numpy as np

    resamples = CFG.bootstrap_resamples if resamples is None else resamples
    if not len(values):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(CFG.seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(resamples, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
