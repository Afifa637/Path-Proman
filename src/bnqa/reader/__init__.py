"""Tier-A extractive reader (PLAN.md §7.5-7.6, task T10).

Feature-based, CPU, and explainable line by line — which is the point, not a
compromise.  A logistic regression with a printed feature-importance table is
something a teacher can audit; a 25M-parameter black box is not.
"""

from .base import Answer, Candidate, Reader  # noqa: F401
from .qtype import QuestionTypeClassifier, analyze  # noqa: F401
from .span_ranker import FeatureReader, SpanRanker  # noqa: F401
