"""``BanglaQA`` — the facade used by the CLI, the Streamlit app and notebooks (T12).

preprocess -> retrieve -> read -> verify -> abstain-or-answer, returning the
six-tuple the whole project promises:

**⟨answer, supporting sentence, corpus citation, calibrated confidence,
supported? + veto reason, evidence receipt⟩**

Nothing above this layer knows which retriever or reader is in use — they are
the ``Retriever`` and ``Reader`` protocols — so swapping in a Tier-B arm is a
constructor argument, and Tier A stays its ablation baseline.

Two design points worth stating, because both are load-bearing:

**The corpus is injectable.**  ``BanglaQA.load(corpus=...)`` takes a passage
list, which is what makes VC-7's counterfactual test a *toggle* rather than a
rebuild: the tampered corpus is a different list of dicts and nothing else
changes.  A system that could only run against one hard-coded corpus could not
demonstrate corpus attribution at all.

**Abstention shows its work.**  An abstaining answer still carries the closest
passage, the citation and the five signals.  VC-5's demo is "the system
abstains **and shows you what it looked at**"; a bare refusal is
indistinguishable from a crash to the person watching.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .config import (CALIBRATOR, CFG, CONFORMAL_JSON, FUSION_MODEL, QTYPE_MODEL,
                     READER_MODEL, SUPPORT_MODEL)
from .preprocess.normalize import normalize, to_bengali_digits
from .reader.base import Answer
from .reader.span_ranker import FeatureReader, SpanRanker
from .retrieval.base import BaseRetriever
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import RRFHybrid
from .retrieval.index import prepare
from .retrieval.tfidf import TfidfRetriever
from .utils import read_json, sha256_text
from .verify.conformal import ConformalThreshold
from .verify.constraints import VetoResult, check as veto_check
from .verify.receipt import build_receipt, write_receipt
from .verify.signals import (SIGNAL_NAMES, SignalContext, fusion_features,
                             to_matrix)
from .verify.threshold import Decision, decide

# --------------------------------------------------------------------------- #
# The result object                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class QAResult:
    """The six-tuple, plus everything the UI draws."""

    question: str
    answer: str
    evidence: str
    citation: dict
    confidence: float
    supported: bool
    veto_reason: str
    receipt: dict
    # -- supporting detail --
    pid: str = ""
    char_start: int = 0
    char_end: int = 0
    sentence_start: int = 0
    sentence_end: int = 0
    signals: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    passages: list[dict] = field(default_factory=list)
    top_candidates: list[dict] = field(default_factory=list)
    route: str = "span"
    expected_class: str = "UNKNOWN"
    question_type: str = "factoid"
    yes_no: str | None = None
    abstained: bool = False
    message: str = ""
    latency_ms: float = 0.0

    def citation_line(self) -> str:
        """The line VC-2 promises, in Bangla, or a plain pid when uncitable."""
        c = self.citation
        if not c or not c.get("chapter_title"):
            return f"pid {self.pid} › chars {self.char_start}–{self.char_end}"
        return (f"শ্রেণি {to_bengali_digits(str(c.get('grade_label') or c.get('grade')))} › "
                f"{c.get('subject_bn') or c.get('subject')} › "
                f"অধ্যায় {to_bengali_digits(str(c.get('chapter_no')))}: {c.get('chapter_title')} › "
                f"pid {self.pid} › chars {to_bengali_digits(str(self.char_start))}–"
                f"{to_bengali_digits(str(self.char_end))}")

    def as_dict(self) -> dict:
        out = dict(self.__dict__)
        out["citation_line"] = self.citation_line()
        return out


# --------------------------------------------------------------------------- #
# The facade                                                                   #
# --------------------------------------------------------------------------- #


class BanglaQA:
    """Everything wired together, with every component swappable."""

    def __init__(self, *, corpus: Sequence[dict], retriever: BaseRetriever,
                 reader: FeatureReader, support_model=None, fusion_model=None,
                 calibrator=None, conformal: ConformalThreshold | None = None,
                 policy: str = "conformal", ensemble: Sequence[SpanRanker] = ()) -> None:
        self.corpus = list(corpus)
        self.by_pid = {p["pid"]: p for p in self.corpus}
        self.retriever = retriever
        self.reader = reader
        self.support_model = support_model
        self.fusion_model = fusion_model
        self.calibrator = calibrator
        self.conformal = conformal
        self.policy = policy
        self.ensemble = list(ensemble)
        self.gazetteer = _gazetteer(self.corpus)

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, *, size: int | None = None, corpus: Sequence[dict] | None = None,
             retriever: str = "rrf", policy: str = "conformal",
             models_dir: Path | None = None, verbose: bool = False) -> "BanglaQA":
        """Build from the fitted artefacts on disk.

        Missing artefacts degrade gracefully and *visibly*: the pipeline still
        answers with whatever is present, because PLAN.md §14's degradation
        order says the end-to-end demo must degrade last, never first.
        """
        size = CFG.headline_index if size is None else size
        passages = list(corpus) if corpus is not None else prepare(size, verbose=verbose)

        retr = build_retriever(retriever, passages)

        qtype_clf = _maybe_pickle(QTYPE_MODEL)
        ranker = (SpanRanker.load(READER_MODEL) if READER_MODEL.exists()
                  else SpanRanker("logreg"))

        support = None
        if SUPPORT_MODEL.exists():
            from .verify.support import SupportClassifier

            support = SupportClassifier.load(SUPPORT_MODEL)

        fusion = None
        if FUSION_MODEL.exists():
            from .verify.fusion import FusionModel

            fusion = FusionModel.load(FUSION_MODEL)

        calibrator = _maybe_pickle(CALIBRATOR)
        conformal = None
        if CONFORMAL_JSON.exists():
            blob = read_json(CONFORMAL_JSON)
            conformal = ConformalThreshold(**{k: blob[k] for k in
                                              ConformalThreshold.__dataclass_fields__})

        ensemble = []
        for path in sorted((models_dir or READER_MODEL.parent).glob("reader_bag_*.pkl")):
            try:
                ensemble.append(SpanRanker.load(path))
            except Exception:  # pragma: no cover - a stale bag is not fatal
                pass

        reader = FeatureReader(ranker, qtype_clf=qtype_clf, ensemble=ensemble)
        return cls(corpus=passages, retriever=retr, reader=reader, support_model=support,
                   fusion_model=fusion, calibrator=calibrator, conformal=conformal,
                   policy=policy, ensemble=ensemble)

    # ------------------------------------------------------------------ #
    # Asking                                                             #
    # ------------------------------------------------------------------ #

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        k = CFG.reader_top_k if k is None else k
        hits = self.retriever.search(normalize(question), k)
        out: list[dict] = []
        for rank, (pid, score) in enumerate(hits, start=1):
            p = self.by_pid.get(pid)
            if p is not None:
                out.append({**p, "rank": rank, "retriever_score": float(score)})
        return out

    def ask(self, question: str, *, k: int | None = None, qid: str | None = None,
            write_receipt_file: bool = False, policy: str | None = None) -> QAResult:
        t0 = time.perf_counter()
        question = normalize(question)
        policy = policy or self.policy

        passages = self.retrieve(question, k)
        if not passages:
            return self._empty(question, qid, t0, policy,
                               "কর্পাসে সংশ্লিষ্ট কোনো অনুচ্ছেদ পাওয়া যায়নি।")

        ranks = [(p["rank"], p["retriever_score"]) for p in passages]
        # S5's bagged rankers score the same candidates, so one read produces
        # both the answer and the ensemble's votes.
        answer, ens = self.reader.read_with_ensemble(question, passages, ranks=ranks)

        # S4 needs answers read from *other* passages, independently — these
        # really are separate reads, because that independence is the signal.
        alt = [self.reader.read(question, [p], ranks=[(p["rank"], p["retriever_score"])])
               for p in passages[1:4]]

        evidence = answer.sentence or (passages[0]["text"][:400] if passages else "")
        ctx = SignalContext(question=question, answer=answer, evidence=evidence,
                            passages=passages, alt_answers=alt, ensemble_answers=ens,
                            support_model=self.support_model, gazetteer=self.gazetteer)

        # One pass: the fusion features are a superset of the five signals, so
        # computing them separately would run S3 and S4 twice per question.
        feats = fusion_features(ctx)
        signals = {k: feats[k] for k in SIGNAL_NAMES}
        veto = veto_check(answer.text, evidence, gazetteer=self.gazetteer)
        confidence = self._confidence(feats, signals, veto)
        decision = decide(confidence, veto, policy=policy, threshold=self.conformal,
                          signals=signals)

        return self._result(question, qid, answer, evidence, passages, signals, veto,
                            decision, confidence, t0, write_receipt_file)

    # ------------------------------------------------------------------ #
    # Verifying somebody else's answer (Tab 1's উত্তর যাচাই control)      #
    # ------------------------------------------------------------------ #

    def verify_claim(self, question: str, candidate: str, *,
                     k: int | None = None) -> dict:
        """Does the corpus support this *candidate* answer?  C1 doing its own job.

        The same discrimination the reader's own answers go through, pointed at
        a string a student typed — which is what §18.2 case 4 demonstrates and
        what makes the veto layer's reason string user-facing rather than
        diagnostic.
        """
        question, candidate = normalize(question), normalize(candidate)
        passages = self.retrieve(question, k)
        if not passages:
            return {"supported": False, "reason": "কর্পাসে সংশ্লিষ্ট অনুচ্ছেদ নেই",
                    "evidence": "", "pid": "", "signals": {}, "veto": {}}

        from .preprocess.tokenize import sentence_spans
        from .reader.candidates import trigram_cosine

        best = None
        for p in passages:
            for sent, s, e in sentence_spans(normalize(p["text"])):
                sim = trigram_cosine(candidate, sent)
                if best is None or sim > best[0]:
                    best = (sim, sent, p, s, e)
        assert best is not None
        _sim, evidence, passage, s_start, s_end = best

        veto = veto_check(candidate, evidence, gazetteer=self.gazetteer)
        probe = Answer(text=candidate, pid=passage["pid"], char_start=s_start,
                       char_end=s_end, sentence=evidence, sentence_start=s_start,
                       sentence_end=s_end, score=0.5, margin=0.0,
                       passage_rank=passage["rank"],
                       passage_score=passage["retriever_score"])
        ctx = SignalContext(question=question, answer=probe, evidence=evidence,
                            passages=passages, support_model=self.support_model,
                            gazetteer=self.gazetteer)
        feats = fusion_features(ctx)
        signals = {k: feats[k] for k in SIGNAL_NAMES}
        confidence = self._confidence(feats, signals, veto)
        label = None
        if self.support_model is not None:
            label = self.support_model.predict_label(candidate, evidence)

        return {
            "supported": bool(not veto.fired and confidence >= self._tau()),
            "confidence": confidence,
            "label": label,
            "reason": veto.reason or ("কর্পাস সমর্থন করে" if confidence >= self._tau()
                                      else "কর্পাসে যথেষ্ট প্রমাণ নেই"),
            "veto": veto.as_dict(),
            "evidence": evidence,
            "pid": passage["pid"],
            "citation": citation_of(passage),
            "signals": signals,
        }

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _tau(self) -> float:
        if self.conformal is not None and self.conformal.feasible:
            return self.conformal.tau
        return 0.5

    def _confidence(self, feats: dict, signals: dict, veto: VetoResult) -> float:
        """Fused, calibrated, then capped by the veto layer — in that order.

        The cap is applied **last** on purpose: a calibrator fitted on fused
        scores has no way to represent "and also the book says 1971", so
        letting it run first and capping afterwards is the only ordering in
        which the guarantee survives contact with a contradiction.
        """
        if self.fusion_model is not None:
            import numpy as np

            raw = float(self.fusion_model.predict_proba(to_matrix([feats]))[0])
            if self.calibrator is not None:
                raw = float(self.calibrator(np.asarray([raw]))[0])
        else:
            # No fusion model fitted yet: the mean of the available signals is
            # a defensible stand-in and keeps the demo alive (§14 degradation).
            raw = sum(signals.values()) / max(len(signals), 1)
        return float(veto.cap(raw))

    def _result(self, question, qid, answer: Answer, evidence, passages, signals,
                veto: VetoResult, decision: Decision, confidence, t0,
                write_file: bool) -> QAResult:
        passage = self.by_pid.get(answer.pid, passages[0])
        citation = citation_of(passage)
        yes_no = None
        if answer.route == "polarity":
            # §10.7: polarity agreement between the proposition and the evidence
            # *is* the yes/no decision.  One mechanism, two jobs.
            yes_no = "না" if veto.fired and "polarity" in veto.categories else "হ্যাঁ"

        receipt = build_receipt(
            qid=qid or sha256_text(question)[:12], question=question,
            answer=answer.text, pid=answer.pid, char_start=answer.char_start,
            char_end=answer.char_end, passage_text=normalize(passage["text"]),
            evidence=evidence, confidence=confidence, signals=signals,
            supported=decision.answered, veto_reason=veto.reason, citation=citation,
            extra={"route": answer.route, "expected_class": answer.expected_class,
                   "policy": decision.policy, "tau": decision.tau})
        if write_file:
            write_receipt(receipt)

        return QAResult(
            question=question,
            answer=answer.text if decision.answered else "",
            evidence=evidence, citation=citation, confidence=confidence,
            supported=decision.answered, veto_reason=veto.reason, receipt=receipt,
            pid=answer.pid, char_start=answer.char_start, char_end=answer.char_end,
            sentence_start=answer.sentence_start, sentence_end=answer.sentence_end,
            signals=signals, decision=decision.as_dict(), passages=passages,
            top_candidates=answer.top_candidates, route=answer.route,
            expected_class=answer.expected_class, question_type=answer.question_type,
            yes_no=yes_no, abstained=not decision.answered, message=decision.message,
            latency_ms=1000 * (time.perf_counter() - t0))

    def _empty(self, question, qid, t0, policy, message) -> QAResult:
        return QAResult(question=question, answer="", evidence="", citation={},
                        confidence=0.0, supported=False, veto_reason="",
                        receipt={}, abstained=True, message=message,
                        decision={"answered": False, "policy": policy, "reason": "no passages"},
                        latency_ms=1000 * (time.perf_counter() - t0))


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def build_retriever(kind: str, passages: Sequence[dict]) -> BaseRetriever:
    """``bm25`` · ``tfidf-word`` · ``tfidf-char`` · ``rrf`` (the default)."""
    if kind == "bm25":
        return BM25Retriever().timed_build(passages)
    if kind == "tfidf-word":
        return TfidfRetriever("word").timed_build(passages)
    if kind == "tfidf-char":
        return TfidfRetriever("char").timed_build(passages)
    if kind in ("rrf", "hybrid"):
        bm25 = BM25Retriever().timed_build(passages)
        char = TfidfRetriever("char").timed_build(passages)
        return RRFHybrid([bm25, char], name="rrf(bm25+char)").build(passages)
    raise ValueError(f"unknown retriever {kind!r}")


def citation_of(passage: dict) -> dict:
    """grade -> subject -> chapter -> pid, or ``{}`` when the source cannot cite.

    Only NCTB-SchoolText carries chapter metadata; Wikipedia distractors and
    BanglaRQA gold contexts do not, and inventing a citation for them would be
    the single worst thing this project could do.  An uncitable passage
    returns an empty dict and the UI prints the pid.
    """
    if not passage.get("chapter_title"):
        return {"pid": passage.get("pid"), "source": passage.get("source"),
                "citable": False, "title": passage.get("title")}
    return {
        "pid": passage.get("pid"),
        "source": passage.get("source"),
        "citable": True,
        "grade": passage.get("grade"),
        "grade_label": passage.get("grade_label"),
        "subject": passage.get("subject"),
        "subject_bn": passage.get("subject_bn"),
        "chapter_no": passage.get("chapter_no"),
        "chapter_title": passage.get("chapter_title"),
    }


def _gazetteer(passages: Sequence[dict], limit: int = 20_000) -> frozenset[str]:
    """Entity candidates harvested from passage titles — no external list."""
    out: set[str] = set()
    for p in passages:
        title = (p.get("title") or p.get("chapter_title") or "").strip()
        if title and len(title) > 2:
            out.add(title)
            for part in title.split():
                if len(part) > 3:
                    out.add(part)
        if len(out) >= limit:
            break
    return frozenset(out)


def _maybe_pickle(path: Path):
    if not path.exists():
        return None
    import pickle

    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:  # pragma: no cover
        return None
