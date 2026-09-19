"""The ``Reader`` protocol and the ``Answer`` it returns (PLAN.md §13, T10).

The second of the three interfaces the two-tier design rests on.  Tier B's
neural span reader (X9) implements exactly this, which is what makes it an
ablation of the feature-based reader rather than a replacement for the pipeline.

``Answer`` carries **character offsets into a named passage**, not just a
string.  That is VC-1 — "extractive by construction" — expressed in the type
system: an ``Answer`` that did not come out of a passage cannot be built
without lying about ``pid``, and :meth:`Answer.verify_substring` re-checks the
claim against the corpus text at any time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@dataclass
class Candidate:
    """One span under consideration, with the provenance the features need."""

    text: str
    pid: str
    char_start: int
    char_end: int
    sentence: str
    sentence_start: int
    sentence_end: int
    passage_rank: int
    passage_score: float
    sentence_rank: int
    features: dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    @property
    def n_tokens(self) -> int:
        from ..preprocess.tokenize import tokenize

        return len(tokenize(self.text))


@dataclass
class Answer:
    """What the reader returns — and what the receipt (VC-3) serialises."""

    text: str
    pid: str
    char_start: int
    char_end: int
    sentence: str
    sentence_start: int
    sentence_end: int
    score: float = 0.0
    margin: float = 0.0            # best minus second-best: an S1 feature
    null_margin: float = 0.0       # best minus the no-answer score
    route: str = "span"
    expected_class: str = "UNKNOWN"
    question_type: str = "factoid"
    spans: list[tuple[str, int, int]] = field(default_factory=list)  # list answers
    top_candidates: list[dict] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    passage_rank: int = 0
    passage_score: float = 0.0

    # -- VC-1 ---------------------------------------------------------------
    def verify_substring(self, passage_text: str) -> bool:
        """``passages[pid].text[start:end] == answer`` — the assertion the UI prints."""
        return passage_text[self.char_start:self.char_end] == self.text

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["spans"] = [list(s) for s in self.spans]
        return out


NO_ANSWER = Answer(text="", pid="", char_start=0, char_end=0, sentence="",
                   sentence_start=0, sentence_end=0, score=0.0)


@runtime_checkable
class Reader(Protocol):
    """``read(question, passages) -> Answer``."""

    name: str

    def read(self, question: str, passages: Sequence[dict], **kwargs) -> Answer: ...
