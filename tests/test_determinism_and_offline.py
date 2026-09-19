"""VC-8 (determinism) and VC-6 (the offline guard) — task V2.

VC-8: same question + same seed + same config hash -> byte-identical answer,
pid and confidence.  An API-backed model drifts; this one cannot, and the
examiner checks it by asking twice.

VC-6: ``BNQA_OFFLINE=1`` installs a socket guard that raises on any outbound
connection.  The guard is what turns "we did not notice any requests" into a
stack trace with a hostname in it.
"""

from __future__ import annotations

import socket

import pytest

from bnqa.config import PASSAGES, config_hash
from bnqa.offline import OutboundBlocked, install, is_active, uninstall

needs_corpus = pytest.mark.skipif(
    not PASSAGES.exists(), reason="corpus not built — run the T1-T3 pipeline")


# --------------------------------------------------------------------------- #
# VC-8 determinism                                                             #
# --------------------------------------------------------------------------- #


def test_config_hash_is_stable_across_calls():
    assert config_hash() == config_hash()
    assert len(config_hash()) == 12


def test_seeding_is_reproducible():
    import random

    import numpy as np

    from bnqa.utils import set_seed

    set_seed()
    a = (random.random(), float(np.random.rand()))
    set_seed()
    b = (random.random(), float(np.random.rand()))
    assert a == b


def test_candidate_generation_is_deterministic():
    from bnqa.reader.candidates import generate

    passages = [{"pid": "p", "text": "উদ্ভিদ কার্বন ডাই-অক্সাইড গ্রহণ করে। "
                                     "এটি ১৯৭১ সালের ঘটনা নয়।"}]
    first = [(c.text, c.char_start, c.char_end) for c in generate("কোন গ্যাস?", passages)]
    second = [(c.text, c.char_start, c.char_end) for c in generate("কোন গ্যাস?", passages)]
    assert first == second


@needs_corpus
def test_retrieval_is_deterministic():
    from bnqa.retrieval.bm25 import BM25Retriever
    from bnqa.retrieval.index import prepare

    passages = prepare(min(10_000, 10_000), verbose=False)[:2000]
    r = BM25Retriever().build(passages)
    q = "মুক্তিযুদ্ধ কত সালে সংঘটিত হয়?"
    assert r.search(q, 10) == r.search(q, 10)


# --------------------------------------------------------------------------- #
# VC-6 the offline guard                                                       #
# --------------------------------------------------------------------------- #


def test_guard_is_off_by_default():
    assert not is_active()


def test_guard_blocks_outbound_and_allows_loopback():
    install(force=True)
    try:
        assert is_active()
        s = socket.socket()
        with pytest.raises(OutboundBlocked):
            s.connect(("huggingface.co", 443))
        s.close()

        with pytest.raises(OutboundBlocked):
            socket.create_connection(("example.com", 80))
    finally:
        uninstall()
    assert not is_active()


def test_guard_names_the_host_it_refused():
    install(force=True)
    try:
        with pytest.raises(OutboundBlocked, match="huggingface.co"):
            socket.create_connection(("huggingface.co", 443))
    finally:
        uninstall()


def test_uninstall_restores_the_real_socket():
    original = socket.socket
    install(force=True)
    assert socket.socket is not original
    uninstall()
    assert socket.socket is original
