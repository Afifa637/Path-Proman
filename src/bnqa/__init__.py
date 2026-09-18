"""bnqa — Path-Proman: evidence-grounded Bangla textbook QA.

Tier A (this package's core) runs on CPU with no torch, no transformers and no
gensim.  Tier B modules (embeddings/, encoder/, the neural retrieval and reader
arms) import those libraries lazily and are never required by Tier A.
"""

from .config import CFG, config_hash  # noqa: F401
from .utils import force_utf8_stdout, set_seed  # noqa: F401

__version__ = "0.4.0"

force_utf8_stdout()
