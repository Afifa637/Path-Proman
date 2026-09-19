"""V3 / VC-7: the Counterfactual Corpus Test.

    python scripts/run_counterfactual.py --n 30

Produces ``reports/tables/counterfactual.csv`` (one row per probed question),
``data/counterfactual/passages_tampered.jsonl`` (which the Streamlit sidebar
toggle reads), and the CAR / AoRR headline numbers in ``results.json``.

This is the experiment that answers *"how do we know it isn't just a language
model guessing?"* with a number.  It is slow — each probe rebuilds the
retriever against a different corpus — and that slowness is the honest cost of
the claim: a cached index would be answering from passages that no longer
exist.

To keep it tractable the probe runs against the **10k** index by default.  The
question it answers is about *attribution*, not about retrieval difficulty, and
attribution does not change with index size; the headline retrieval numbers
stay at 50k where PLAN.md §16.1 puts them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnqa.config import CFG, ensure_dirs  # noqa: E402
from bnqa.eval.counterfactual import run  # noqa: E402
from bnqa.eval.retrieval_metrics import load_queries  # noqa: E402
from bnqa.pipeline import BanglaQA, build_retriever  # noqa: E402
from bnqa.retrieval.index import prepare  # noqa: E402
from bnqa.utils import log_result, save_table, set_seed, timer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=10_000)
    ap.add_argument("--n", type=int, default=CFG.counterfactual_n)
    ap.add_argument("--pool", type=int, default=400,
                    help="how many val questions to consider before giving up")
    ap.add_argument("--retriever", default="bm25",
                    help="bm25 keeps the rebuild-per-probe affordable")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    print(f"V3  counterfactual corpus test (VC-7)  ·  target N = {args.n}")

    corpus = prepare(args.size)
    # The reader and the verifier are corpus-independent, so they are loaded
    # once and shared; only the *retriever* has to be rebuilt per corpus,
    # because that is precisely the thing under test.
    shared = _load_components()

    def factory(passages):
        """A fresh pipeline over a *specific* corpus — the whole point."""
        return BanglaQA(corpus=passages,
                        retriever=build_retriever(args.retriever, passages), **shared)

    questions = [q for q in load_queries("val") if q.get("is_answerable")][:args.pool]
    with timer("counterfactual probe"):
        out = run(factory, corpus, questions, n=args.n)

    print(f"\n  N = {out['n']} (from {out['considered']} candidates considered)")
    print(f"  CAR  (Corpus Attribution Rate)   = {out['car']:.4f}")
    print(f"  AoRR (Abstention on Removal)     = {out['aorr']:.4f}")
    print(f"  {out['interpretation']}")

    save_table("counterfactual", out["rows"])
    log_result("v3_counterfactual", "summary",
               {k: v for k, v in out.items() if k != "rows"})


def _load_components() -> dict:
    """The fitted reader and verifier, straight off disk."""
    import pickle

    from bnqa.config import (CALIBRATOR, CONFORMAL_JSON, FUSION_MODEL, QTYPE_MODEL,
                             READER_MODEL, SUPPORT_MODEL)
    from bnqa.reader.span_ranker import FeatureReader, SpanRanker
    from bnqa.utils import read_json
    from bnqa.verify.conformal import ConformalThreshold

    if not READER_MODEL.exists():
        raise SystemExit("no fitted reader — run scripts/run_reader.py first")

    qtype = pickle.load(open(QTYPE_MODEL, "rb")) if QTYPE_MODEL.exists() else None
    ensemble = [SpanRanker.load(p)
                for p in sorted(READER_MODEL.parent.glob("reader_bag_*.pkl"))]
    reader = FeatureReader(SpanRanker.load(READER_MODEL), qtype_clf=qtype,
                           ensemble=ensemble)

    support = None
    if SUPPORT_MODEL.exists():
        from bnqa.verify.support import SupportClassifier

        support = SupportClassifier.load(SUPPORT_MODEL)

    fusion = None
    if FUSION_MODEL.exists():
        from bnqa.verify.fusion import FusionModel

        fusion = FusionModel.load(FUSION_MODEL)

    calibrator = pickle.load(open(CALIBRATOR, "rb")) if CALIBRATOR.exists() else None
    conformal = None
    if CONFORMAL_JSON.exists():
        blob = read_json(CONFORMAL_JSON)
        conformal = ConformalThreshold(
            **{k: blob[k] for k in ConformalThreshold.__dataclass_fields__})

    return {"reader": reader, "support_model": support, "fusion_model": fusion,
            "calibrator": calibrator, "conformal": conformal, "ensemble": ensemble}


if __name__ == "__main__":
    main()
