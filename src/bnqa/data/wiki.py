"""Bangla Wikipedia distractors (PLAN.md §6.2B, task T3).

Distractors are drawn from **the same distribution as the gold passages**.  This
is not fussiness: BanglaRQA's contexts are Bangla Wikipedia (their ids are
``bn_wiki_*``), so if every distractor were a textbook passage, a retriever could
partly succeed by learning *"gold passages look like Wikipedia"* and Recall@k
would be inflated by a **distribution artefact** rather than by relevance.

Wikipedia articles are the only source we chunk ourselves — NCTB-SchoolText
arrives pre-chunked into pedagogically coherent units and Bangla-TextBook into
128-word blocks.  Chunks are sentence-aligned, so one never ends mid-sentence,
and their **lengths are sampled from the gold length distribution** rather than
fixed: see :func:`chunk_article_length_matched` for the measured reason why.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from ..config import CFG, PROCESSED, RAW_WIKI, ensure_dirs
from ..preprocess.normalize import bengali_ratio, normalize
from ..preprocess.tokenize import sentences, tokenize
from ..utils import log_result, save_table, write_jsonl


TAIL_KEEP_RATIO = 0.85  # a chunk must reach this fraction of its sampled target


def gold_length_distribution() -> np.ndarray:
    """Character lengths of the BanglaRQA gold contexts."""
    from ..utils import read_jsonl

    path = PROCESSED / "gold_contexts.jsonl"
    return np.array([len(r["text"]) for r in read_jsonl(path)], dtype=np.int64)


def chunk_article_length_matched(text: str, targets: np.ndarray,
                                 rng: np.random.Generator) -> list[str]:
    """Chunk to a **sampled** target length drawn from the gold distribution.

    This exists because of a measured failure.  Chunking Wikipedia to a fixed
    120–180 tokens while leaving BanglaRQA's contexts whole made the two
    trivially distinguishable: gold passages ran to a median 1,378 characters
    against 856 for the distractors, and passage **length alone** predicted
    "is gold" with **AUC 0.795**.

    That is precisely the distribution artefact PLAN.md §6.2B warns about, in a
    form we introduced ourselves.  It is not cosmetic: BM25's ``b`` parameter
    tunes length normalisation directly, and on the broken index it tuned to
    ``b=0.3`` — weak normalisation, which favours long documents, which are the
    gold ones.  Part of Recall@5 would have been "gold passages are longer"
    rather than relevance.

    Sampling each chunk's target length from the gold distribution removes the
    signal at its source.
    """
    sents = sentences(text)
    if not sents:
        return []

    chunks: list[str] = []
    i = 0
    while i < len(sents):
        target = int(targets[rng.integers(len(targets))])
        buf: list[str] = []
        size = 0
        while i < len(sents) and size < target:
            buf.append(sents[i])
            size += len(sents[i]) + 1
            i += 1
        if not buf:
            continue
        # Drop the short tail.  An article that runs out of sentences before
        # reaching its sampled target would otherwise emit a stub, and those
        # stubs are what kept the length signal alive after the first fix
        # (AUC 0.795 -> 0.658 rather than to chance).  Wikipedia supplies far
        # more articles than we need, so being picky costs nothing.
        if size >= TAIL_KEEP_RATIO * target:
            chunks.append(" ".join(buf))
    return chunks


def chunk_article(text: str, *, lo: int | None = None, hi: int | None = None,
                  overlap: int | None = None) -> list[str]:
    """Fixed-width sentence-aligned chunks of ``lo``–``hi`` tokens.

    Retained for the NCTB fallback path and for tests; the index itself uses
    :func:`chunk_article_length_matched`.
    """
    lo = lo or CFG.wiki_chunk_tokens[0]
    hi = hi or CFG.wiki_chunk_tokens[1]
    overlap = CFG.wiki_chunk_overlap if overlap is None else overlap

    sents = sentences(text)
    if not sents:
        return []

    lengths = [len(tokenize(s, do_normalize=False)) for s in sents]
    chunks: list[str] = []
    i = 0
    while i < len(sents):
        buf: list[str] = []
        count = 0
        j = i
        while j < len(sents) and count < hi:
            buf.append(sents[j])
            count += lengths[j]
            j += 1
            if count >= lo:
                break
        if buf:
            chunks.append(" ".join(buf))
        if j >= len(sents):
            break
        # step back far enough to give the next chunk its overlap
        back = 0
        k = j
        while k > i + 1 and back < overlap:
            k -= 1
            back += lengths[k]
        i = max(k, i + 1)
    return chunks


def build_passages(*, target: int, exclude_titles: set[str] | None = None,
                   parquet: Path | None = None) -> tuple[list[dict], Counter]:
    """Chunk Wikipedia articles until ``target`` usable passages exist."""
    import pyarrow.parquet as pq

    parquet = parquet or RAW_WIKI / "train-00000-of-00002.parquet"
    exclude_titles = exclude_titles or set()
    stats: Counter = Counter()
    rows: list[dict] = []
    targets = gold_length_distribution()
    rng = np.random.default_rng(CFG.seed)

    pf = pq.ParquetFile(parquet)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["id", "title", "text"])
        ids = table.column("id").to_pylist()
        titles = table.column("title").to_pylist()
        texts = table.column("text").to_pylist()
        for aid, title, text in zip(ids, titles, texts):
            stats["articles_read"] += 1
            title_n = normalize(str(title or ""))
            if title_n in exclude_titles:
                stats["excluded_gold_title"] += 1
                continue
            body = normalize(str(text or ""))
            if len(body) < CFG.min_passage_chars:
                stats["article_too_short"] += 1
                continue
            for ci, chunk in enumerate(chunk_article_length_matched(body, targets, rng)):
                if len(chunk) < CFG.min_passage_chars:
                    stats["chunk_too_short"] += 1
                    continue
                if bengali_ratio(chunk) < CFG.min_bengali_ratio:
                    stats["chunk_non_bengali"] += 1
                    continue
                if len(chunk) > CFG.max_passage_chars:
                    chunk = chunk[: CFG.max_passage_chars]
                    stats["truncated"] += 1
                rows.append({
                    "pid": f"wiki_{aid}_{ci}",
                    "text": chunk,
                    "source": "wiki",
                    "grade": None, "grade_label": None,
                    "subject": None, "subject_bn": None,
                    "chapter_no": None, "chapter_title": None,
                    "title": title_n,
                })
                stats["kept"] += 1
                if len(rows) >= target:
                    return rows, stats
    return rows, stats


def gold_titles() -> set[str]:
    """Titles of the BanglaRQA gold articles.

    A distractor drawn from the *same article* as a gold passage is not a
    distractor — it is a near-duplicate that would make Recall@k look worse for
    reasons that have nothing to do with the retriever.  Excluding them keeps
    the comparison clean in the opposite direction from the distribution
    argument above.
    """
    from ..config import QA_TEST, QA_TRAIN, QA_VAL
    from ..utils import read_jsonl

    titles: set[str] = set()
    for path in (QA_TRAIN, QA_VAL, QA_TEST):
        if path.exists():
            for row in read_jsonl(path):
                t = (row.get("passage_title") or "").strip()
                if t:
                    titles.add(t)
    return titles


def build(*, target: int | None = None) -> dict:
    """Produce distractor passages sized for the largest index."""
    ensure_dirs()
    target = target or int(max(CFG.index_composition.values(),
                               key=lambda c: c["wiki"])["wiki"] * 1.15)
    print(f"T3a Bangla Wikipedia distractors (target {target:,d})")

    exclude = gold_titles()
    rows, stats = build_passages(target=target, exclude_titles=exclude)

    out = PROCESSED / "wiki_passages.jsonl"
    write_jsonl(out, rows)
    print(f"  {stats['articles_read']:,d} articles read -> {len(rows):,d} chunks kept "
          f"(excluded {stats['excluded_gold_title']:,d} gold-title articles)")
    print(f"  dropped: {stats['chunk_too_short']:,d} short, "
          f"{stats['chunk_non_bengali']:,d} non-Bengali")
    print(f"  -> {out.name}")

    payload = {"articles_read": stats["articles_read"], "kept": len(rows),
               "excluded_gold_title": stats["excluded_gold_title"],
               "chunk_tokens": list(CFG.wiki_chunk_tokens),
               "chunk_overlap": CFG.wiki_chunk_overlap}
    log_result("t3_wiki", "distractors", payload)
    return payload


if __name__ == "__main__":
    build()
