"""Candidate span generation (PLAN.md §7.6, task T10).

For each retrieved passage:

1. split into sentences;
2. score sentences by IDF-weighted question overlap + char-3gram cosine;
3. keep the top ``CFG.cand_sentences``;
4. enumerate token n-grams of length 1..``CFG.cand_ngram_max`` inside them;
5. filter by the expected answer class (§7.5) and drop degenerate spans;
6. dedup by surface form within a passage.

Typical yield is 200-600 candidates per question, which is what the plan
predicts and what the ranker is sized for.

**Character offsets are carried, never re-derived.**  Every span records the
offsets of its first and last token *in the passage text as stored*, so the
answer the reader returns is a literal slice of ``passages.jsonl`` — VC-1 holds
because generation never builds a string that did not come out of the passage.
Re-tokenising the answer later and searching for it would break exactly when
whitespace differs, which is the failure mode the receipt check is meant to
catch rather than to suffer.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from ..config import CFG
from ..preprocess.normalize import normalize
from ..preprocess.stopwords import STOPWORDS, content_tokens
from ..preprocess.tokenize import sentence_spans, token_spans, tokenize_lower
from .base import Candidate
from .qtype import PROCESS, REASON, span_matches_class

# A span consisting only of these is never an answer, whatever it scores.
_TRIVIAL_EDGE = STOPWORDS | {",", ".", "।", ";", ":"}


# --------------------------------------------------------------------------- #
# IDF over the working set                                                     #
# --------------------------------------------------------------------------- #


class IdfTable:
    """Document frequencies over the retrieved passages.

    Computed over the *working set* rather than the whole index on purpose: the
    reader's job is to discriminate between the handful of passages retrieval
    returned, and a term common to all of them carries no information *here*
    even if it is rare corpus-wide.
    """

    def __init__(self, docs: Sequence[str]) -> None:
        self.n = max(len(docs), 1)
        df: Counter = Counter()
        for doc in docs:
            df.update(set(tokenize_lower(doc)))
        self.df = df

    def __call__(self, token: str) -> float:
        return math.log(1.0 + (self.n + 0.5) / (self.df.get(token, 0) + 0.5))

    def weight(self, tokens: Sequence[str]) -> float:
        return sum(self(t) for t in tokens)


# --------------------------------------------------------------------------- #
# Sentence scoring                                                             #
# --------------------------------------------------------------------------- #


def char_trigrams(text: str) -> set[str]:
    t = f" {normalize(text).lower()} "
    return {t[i:i + 3] for i in range(max(len(t) - 2, 0))}


def trigram_cosine(a: str, b: str) -> float:
    A, B = char_trigrams(a), char_trigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / math.sqrt(len(A) * len(B))


def score_sentences(question: str, sentences: Sequence[tuple[str, int, int]],
                    idf: IdfTable) -> list[tuple[float, tuple[str, int, int]]]:
    q_tokens = set(content_tokens(tokenize_lower(question)))
    scored: list[tuple[float, tuple[str, int, int]]] = []
    for sent in sentences:
        s_tokens = set(tokenize_lower(sent[0]))
        overlap = idf.weight(sorted(q_tokens & s_tokens))
        denom = idf.weight(sorted(q_tokens)) or 1.0
        lexical = overlap / denom
        scored.append((lexical + trigram_cosine(question, sent[0]), sent))
    scored.sort(key=lambda kv: -kv[0])
    return scored


# --------------------------------------------------------------------------- #
# Generation                                                                   #
# --------------------------------------------------------------------------- #


def generate(question: str, passages: Sequence[dict], *, expected_class: str = "UNKNOWN",
             ranks: Sequence[tuple[int, float]] | None = None,
             max_sentences: int | None = None,
             max_ngram: int | None = None) -> list[Candidate]:
    """All candidate spans for one question over the retrieved passages."""
    max_sentences = CFG.cand_sentences if max_sentences is None else max_sentences
    max_ngram = CFG.cand_ngram_max if max_ngram is None else max_ngram

    idf = IdfTable([p["text"] for p in passages])
    out: list[Candidate] = []

    for p_idx, passage in enumerate(passages):
        text = normalize(passage["text"])
        rank, p_score = (ranks[p_idx] if ranks else (p_idx + 1, 0.0))
        sents = sentence_spans(text)
        if not sents:
            continue
        top = score_sentences(question, sents, idf)[:max_sentences]

        seen: set[str] = set()
        for s_rank, (_s_score, (sent_text, s_start, s_end)) in enumerate(top, start=1):
            spans = token_spans(text[s_start:s_end])
            n = len(spans)

            # REASON and PROCESS answers are *clauses* (§7.5), and a clause is
            # routinely longer than ``cand_ngram_max`` tokens — 9% of gold
            # spans are, and no n-gram enumeration can ever reach them.  So
            # clause-shaped candidates are added whole for those classes.
            if expected_class in (REASON, PROCESS):
                for cl_start, cl_end in _clause_spans(text, s_start, s_end):
                    surface = text[cl_start:cl_end]
                    key = surface.lower()
                    if key in seen or not surface.strip():
                        continue
                    seen.add(key)
                    out.append(Candidate(
                        text=surface, pid=passage["pid"], char_start=cl_start,
                        char_end=cl_end, sentence=sent_text, sentence_start=s_start,
                        sentence_end=s_end, passage_rank=rank, passage_score=p_score,
                        sentence_rank=s_rank))

            for i in range(n):
                for length in range(1, min(max_ngram, n - i) + 1):
                    first, last = spans[i], spans[i + length - 1]
                    start = s_start + first[1]
                    end = s_start + last[2]
                    surface = text[start:end]
                    if not _acceptable(surface, spans[i:i + length], expected_class):
                        continue
                    key = surface.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(Candidate(
                        text=surface, pid=passage["pid"], char_start=start, char_end=end,
                        sentence=sent_text, sentence_start=s_start, sentence_end=s_end,
                        passage_rank=rank, passage_score=p_score, sentence_rank=s_rank))
    return out


def _acceptable(surface: str, toks: Sequence[tuple[str, int, int]], expected: str) -> bool:
    if not surface.strip():
        return False
    words = [t[0].lower() for t in toks]
    # A span that is *only* stopwords carries no answer.  Spans that merely
    # begin or end on one are kept: rejecting them looked sensible and cost
    # 12% of gold spans outright (measured — ``reports/tables/reader.csv``
    # carries the candidate-recall column), because Bangla answers routinely
    # start with a determiner or end with a case particle.  The ranker already
    # has ``starts_with_stopword`` and ``stopword_frac`` to penalise the bad
    # ones, and a feature it can weigh beats a filter it cannot appeal.
    return not all(w in _TRIVIAL_EDGE for w in words) and span_matches_class(surface, expected)


def _clause_spans(text: str, s_start: int, s_end: int) -> list[tuple[int, int]]:
    """The whole sentence, plus its comma- and conjunction-delimited clauses."""
    sent = text[s_start:s_end]
    out: list[tuple[int, int]] = [(s_start, s_end)]
    pos = 0
    for piece in re.split(r"(?<=[,;:])\s+|\s+(?=কারণ|যেহেতু|ফলে|তাই|অর্থাৎ)\s*", sent):
        if piece is None:
            continue
        idx = sent.find(piece, pos)
        if idx == -1 or len(piece.strip()) < 8:
            pos = max(pos, idx + len(piece)) if idx != -1 else pos
            continue
        lead = len(piece) - len(piece.lstrip())
        start = s_start + idx + lead
        end = start + len(piece.strip())
        if (start, end) != (s_start, s_end):
            out.append((start, end))
        pos = idx + len(piece)
    return out


def gold_candidate_index(cands: Sequence[Candidate], gold_pid: str,
                         gold_start: int, gold_end: int) -> int:
    """Index of the exact gold span among the candidates, or ``-1``.

    Used to build training labels.  Matching on offsets rather than on the
    answer string is deliberate: the same string can occur several times in a
    passage and only one occurrence is the annotated one.
    """
    for i, c in enumerate(cands):
        if c.pid == gold_pid and c.char_start == gold_start and c.char_end == gold_end:
            return i
    return -1
