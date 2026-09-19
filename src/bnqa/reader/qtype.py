"""Question analysis (PLAN.md §7.5, task T10).

Two layers, and the split between them is the point:

* a **rule table** mapping Bangla interrogatives to an *expected answer class*.
  Every row is individually defensible, which makes it good viva material and
  makes a wrong prediction traceable to one line rather than to a weight matrix;
* a supervised **4-way ``question_type`` classifier** (confirmation / factoid /
  causal / list) trained on BanglaRQA's own labels, with the rule table's
  outputs as features.

The predicted class **routes decoding** (§7.5): POLARITY questions are answered
by the verifier's polarity check rather than by span extraction, LIST questions
go to multi-span set scoring, everything else takes the single-span path.  So
this module is not decoration — a wrong route costs the answer.

Ambiguity is real and is handled explicitly.  ``কি`` is a yes/no particle
sentence-finally but a content word (``কী``, "what") sentence-initially, and the
two are routinely spelled the same way in running text.  Position therefore
beats lexeme here, and :func:`expected_class` documents that choice where it is
made.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..preprocess.normalize import normalize
from ..preprocess.tokenize import tokenize

# --------------------------------------------------------------------------- #
# The rule table (PLAN.md §7.5)                                                #
# --------------------------------------------------------------------------- #

PERSON = "PERSON"
THING = "THING"
PLACE = "PLACE"
TIME = "TIME"
NUMBER = "NUMBER"
REASON = "REASON"
PROCESS = "PROCESS"
POLARITY = "POLARITY"
UNKNOWN = "UNKNOWN"

ANSWER_CLASSES = (PERSON, THING, PLACE, TIME, NUMBER, REASON, PROCESS, POLARITY, UNKNOWN)

RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("কে", "কারা", "কাকে", "কার", "কাদের"), PERSON),
    (("কোথায়", "কোথা", "কোন্‌খানে", "কোনখানে"), PLACE),
    (("কখন", "কবে"), TIME),
    (("কেন", "কীজন্য", "কিজন্য"), REASON),
    (("কীভাবে", "কিভাবে", "কেমনে"), PROCESS),
    (("কত", "কতটি", "কয়টি", "কতগুলো", "কতজন", "কতটুকু", "কতখানি"), NUMBER),
    (("কী", "কি", "কোনটি", "কোনগুলো", "কোন"), THING),
)

# Multi-word cues, checked before the single-token table because they are
# unambiguous where their head token is not: "কত সালে" is a TIME question even
# though "কত" alone is a NUMBER question.
PHRASE_RULES: tuple[tuple[str, str], ...] = (
    ("কত সালে", TIME), ("কোন সালে", TIME), ("কত সালের", TIME),
    ("কত তারিখে", TIME), ("কোন তারিখে", TIME), ("কোন বছর", TIME),
    ("কত শতাব্দী", TIME), ("কোন সময়", TIME),
    ("কী কারণে", REASON), ("কি কারণে", REASON),
    ("কোন স্থানে", PLACE), ("কোন জায়গায", PLACE), ("কোন দেশে", PLACE),
    ("কত সংখ্যক", NUMBER), ("কত শতাংশ", NUMBER),
)

# BanglaRQA's own four labels, which the supervised head predicts.
QUESTION_TYPES = ("confirmation", "factoid", "causal", "list")

TYPE_TO_CLASS = {
    "confirmation": POLARITY,
    "causal": REASON,
    "list": THING,
    "factoid": UNKNOWN,   # the rule table decides; factoid spans every class
}

LIST_CUES: frozenset[str] = frozenset({"কী কী", "কি কি", "কোন কোন", "কারা কারা",
                                       "কয়টি", "কতগুলো", "তালিকা", "নামগুলো"})


@dataclass(frozen=True)
class QuestionAnalysis:
    question: str
    expected_class: str
    rule_fired: str
    question_type: str
    route: str           # "polarity" | "list" | "span"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Rules                                                                        #
# --------------------------------------------------------------------------- #


def expected_class(question: str) -> tuple[str, str]:
    """``(class, the rule that fired)``."""
    q = normalize(question)
    for phrase, cls in PHRASE_RULES:
        if phrase in q:
            return cls, f"phrase:{phrase}"

    toks = tokenize(q)
    if not toks:
        return UNKNOWN, "empty"

    # ``কি`` sentence-finally is the yes/no particle; sentence-initially it is
    # the content word "what".  The two are spelled interchangeably in running
    # Bangla, so position decides rather than the lexeme.
    if toks[-1] in ("কি", "কী") and len(toks) > 1:
        return POLARITY, "final:কি (yes/no particle)"

    for tokens, cls in RULES:
        for i, tok in enumerate(toks):
            if tok in tokens:
                if tok in ("কি", "কী") and i == 0 and len(toks) > 2:
                    # Sentence-initial কি before a full clause reads as polarity
                    # only when no other interrogative follows.
                    rest = set(toks[1:])
                    if not any(t in rest for group, _ in RULES for t in group):
                        return POLARITY, "initial:কি (yes/no)"
                return cls, f"token:{tok}"
    return UNKNOWN, "no-rule"


def is_list_question(question: str) -> bool:
    q = normalize(question)
    return any(cue in q for cue in LIST_CUES)


def route_for(expected: str, question_type: str | None, question: str) -> str:
    """Which decoder handles this question."""
    if expected == POLARITY or question_type == "confirmation":
        return "polarity"
    if question_type == "list" or is_list_question(question):
        return "list"
    return "span"


def analyze(question: str, question_type: str | None = None) -> QuestionAnalysis:
    """The rule layer alone — used when no trained head is available."""
    cls, rule = expected_class(question)
    qtype = question_type or "factoid"
    return QuestionAnalysis(question=question, expected_class=cls, rule_fired=rule,
                            question_type=qtype,
                            route=route_for(cls, question_type, question))


# --------------------------------------------------------------------------- #
# The supervised 4-way head                                                    #
# --------------------------------------------------------------------------- #


class QuestionTypeClassifier:
    """Logistic regression over rule features + char n-grams of the question.

    Trained on BanglaRQA's ``question_type`` labels.  The rule table's own
    prediction is one of the features, so the model can only *correct* the rules
    where the data disagrees with them — which keeps the rules auditable and
    makes the disagreement itself measurable.
    """

    def __init__(self) -> None:
        self.pipeline = None
        self.classes_: list[str] = []

    def _features(self, questions: list[str]) -> list[str]:
        """Rule output prepended to the text, so the linear model can key on it."""
        out = []
        for q in questions:
            cls, rule = expected_class(q)
            out.append(f"__{cls}__ __{rule.split(':')[0]}__ "
                       f"__list{int(is_list_question(q))}__ {normalize(q)}")
        return out

    def fit(self, questions: list[str], labels: list[str]) -> "QuestionTypeClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        from ..config import CFG

        self.pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                            max_features=200_000, sublinear_tf=True),
            LogisticRegression(max_iter=1000, C=4.0, random_state=CFG.seed,
                               class_weight="balanced"),
        )
        self.pipeline.fit(self._features(questions), labels)
        self.classes_ = list(self.pipeline.classes_)
        return self

    def predict(self, questions: list[str]) -> list[str]:
        if self.pipeline is None:
            return ["factoid"] * len(questions)
        return list(self.pipeline.predict(self._features(questions)))

    def analyze_many(self, questions: list[str]) -> list[QuestionAnalysis]:
        preds = self.predict(questions)
        return [analyze(q, t) for q, t in zip(questions, preds)]


# --------------------------------------------------------------------------- #
# Class-consistent span filtering (used by candidate generation)               #
# --------------------------------------------------------------------------- #

_YEARISH = set("0123456789")


def span_matches_class(span: str, expected: str) -> bool:
    """Cheap type filter applied during candidate generation (§7.6).

    Deliberately permissive: this runs *before* ranking and a false reject can
    never be recovered, while a false accept only costs the ranker one more
    candidate to sort through.
    """
    from ..verify.constraints import ENTITY_MARKERS, HONORIFICS, UNITS, years

    toks = tokenize(span)
    if not toks:
        return False
    has_digit = any(c in _YEARISH for c in span)

    if expected == TIME:
        return bool(years(span)) or has_digit
    if expected == NUMBER:
        return has_digit or any(t in UNITS for t in toks)
    if expected == PERSON:
        # No NER in Tier A, so PERSON only excludes the obviously non-personal:
        # a bare number is never a person.
        return not (has_digit and len(toks) == 1)
    if expected == PLACE:
        return not (has_digit and len(toks) == 1) or any(
            t in ENTITY_MARKERS or t in HONORIFICS for t in toks)
    if expected in (REASON, PROCESS):
        return len(toks) >= 3   # a clause, not a word
    return True
