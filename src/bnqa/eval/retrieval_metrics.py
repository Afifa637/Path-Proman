"""Retrieval metrics (PLAN.md §16.2, tasks T5/T7).

Recall@k, MRR@10 and nDCG@10 over BanglaRQA questions, where the single relevant
document is the question's ``gold_passage_id``.  With one relevant document per
query nDCG reduces to ``1/log2(rank+1)`` and MRR to ``1/rank``; both are computed
explicitly anyway so the code reads the same as the definitions in the report.

Every hyperparameter that any of these numbers depends on is selected on **val**;
the test split is scored once, at the end (PLAN.md §16.1).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from ..config import CFG, QA_TEST, QA_TRAIN, QA_VAL
from ..utils import load_jsonl

SPLIT_FILES = {"train": QA_TRAIN, "val": QA_VAL, "test": QA_TEST}


def load_queries(split: str = "val", *, answerable_only: bool = True,
                 limit: int | None = None) -> list[dict]:
    """Questions with their gold passage id.

    ``answerable_only`` keeps the retrieval task well-posed: an unanswerable
    question still names a passage, but "did we retrieve it" is not the quantity
    the abstention story cares about, and mixing the two would blur RQ1.
    """
    rows = load_jsonl(SPLIT_FILES[split])
    if answerable_only:
        rows = [r for r in rows if r.get("is_answerable")]
    if limit:
        rows = rows[:limit]
    return rows


def evaluate(results: Sequence[Sequence[tuple[str, float]]], golds: Sequence[str],
             *, ks: tuple[int, ...] | None = None) -> dict:
    """Score ranked lists against one relevant pid each."""
    ks = ks or CFG.recall_at
    n = len(golds)
    if n == 0:
        return {}

    recall = {k: 0 for k in ks}
    rr_sum = 0.0
    ndcg_sum = 0.0
    found = 0
    ranks: list[int] = []

    for hits, gold in zip(results, golds):
        pids = [pid for pid, _ in hits]
        rank = pids.index(gold) + 1 if gold in pids else None
        if rank is not None:
            found += 1
            ranks.append(rank)
            for k in ks:
                if rank <= k:
                    recall[k] += 1
            if rank <= 10:
                rr_sum += 1.0 / rank
                ndcg_sum += 1.0 / math.log2(rank + 1)

    out = {f"recall@{k}": recall[k] / n for k in ks}
    out["mrr@10"] = rr_sum / n
    out["ndcg@10"] = ndcg_sum / n
    out["found_any"] = found / n
    out["median_rank_when_found"] = float(np.median(ranks)) if ranks else float("nan")
    out["n_queries"] = n
    return out


def evaluate_retriever(retriever, queries: Sequence[dict], *, k: int = 10) -> dict:
    """Run a retriever over ``queries`` and return metrics plus latency."""
    texts = [q["question"] for q in queries]
    golds = [q["gold_passage_id"] for q in queries]
    results, ms = retriever.timed_search(texts, k=max(k, max(CFG.recall_at)))
    metrics = evaluate(results, golds)
    metrics.update(
        retriever=retriever.name,
        ms_per_query=round(ms, 3),
        build_seconds=round(getattr(retriever, "build_seconds", 0.0), 2),
        index_mb=round(getattr(retriever, "index_bytes", 0) / 1024**2, 1),
    )
    return metrics


def per_question_type(results: Sequence[Sequence[tuple[str, float]]],
                      queries: Sequence[dict], k: int = 5) -> dict[str, float]:
    """Recall@k broken out by BanglaRQA ``question_type``."""
    hit: defaultdict[str, int] = defaultdict(int)
    tot: defaultdict[str, int] = defaultdict(int)
    for hits, q in zip(results, queries):
        qt = q.get("question_type") or "unknown"
        tot[qt] += 1
        if q["gold_passage_id"] in [pid for pid, _ in hits[:k]]:
            hit[qt] += 1
    return {qt: hit[qt] / tot[qt] for qt in sorted(tot)}
