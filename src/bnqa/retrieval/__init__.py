"""Tier-A sparse retrieval (PLAN.md §7.3, tasks T5 and T7).

Every arm implements :class:`~bnqa.retrieval.base.Retriever`, which is what lets
Tier B's neural arms replace them one at a time without touching the pipeline.
"""

from .base import BaseRetriever, Retriever, sparse_terms  # noqa: F401
from .bm25 import BM25Retriever  # noqa: F401
from .hybrid import ExpandedQuery, RRFHybrid  # noqa: F401
from .index import prepare  # noqa: F401
from .tfidf import TfidfRetriever  # noqa: F401
