"""T11 + T11b: BanglaVerify, S3, fusion, calibration, and the conformal bound.

    python scripts/run_verify.py                  # full run
    python scripts/run_verify.py --eval 150       # quick pass

Produces every artefact T11's acceptance criterion names:

  * ``banglaverify_quality.csv``   per-generator label precision (T11b)
  * ``verify_contrast.csv``        contrast-set accuracy, **veto on vs off**
  * ``fusion.csv``                 Naive Bayes vs LogReg vs GBDT (RQ5)
  * ``calibration.csv``            uncalibrated / temperature / Platt / isotonic
  * ``conformal.csv``              the bound, and the achieved risk on test
  * ``abstention_policies.csv``    RQ8b's four arms with risk-coverage AUC
  * ``figures/reliability.png`` · ``figures/risk_coverage.png``

**Protocol (PLAN.md §16.1), and it is the whole point of the guarantee.**  val
is split in half: the first half fits fusion and the calibrator, the second
half is the *conformal calibration split* used only to choose τ.  Test is
opened exactly once, at the end, to report the achieved risk.  A τ chosen on
data the fusion model was fitted on would carry no guarantee at all.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnqa.config import (CALIBRATOR, CFG, CONFORMAL_JSON, FUSION_MODEL,  # noqa: E402
                         QTYPE_MODEL, READER_MODEL, SUPPORT_MODEL, ensure_dirs)
from bnqa.eval.qa_metrics import token_f1  # noqa: E402
from bnqa.eval.retrieval_metrics import load_queries  # noqa: E402
from bnqa.eval.verify_metrics import (calibration_row, contrast_ablation,  # noqa: E402
                                      correctness, reliability_table)
from bnqa.pipeline import build_retriever  # noqa: E402
from bnqa.preprocess.normalize import normalize  # noqa: E402
from bnqa.reader.span_ranker import FeatureReader, SpanRanker  # noqa: E402
from bnqa.retrieval.index import prepare  # noqa: E402
from bnqa.utils import (log_result, save_table, set_seed, timer,  # noqa: E402
                        write_json)
from bnqa.verify import banglaverify as bv  # noqa: E402
from bnqa.verify.calibration import compare as compare_calibrators  # noqa: E402
from bnqa.verify.calibration import reliability_figure  # noqa: E402
from bnqa.verify.conformal import (compare_policies, risk_coverage_figure,  # noqa: E402
                                   select_threshold, verify as conformal_verify)
from bnqa.verify.constraints import check as veto_check  # noqa: E402
from bnqa.verify.fusion import compare as compare_fusion  # noqa: E402
from bnqa.verify.signals import (SIGNAL_NAMES, SignalContext, fusion_features,  # noqa: E402
                                 to_matrix)
from bnqa.verify.support import SupportClassifier  # noqa: E402


# --------------------------------------------------------------------------- #
# Collecting signals over a split                                              #
# --------------------------------------------------------------------------- #


def collect(rows, by_pid, retriever, reader, ensemble, support, gazetteer, *,
            verbose=True) -> list[dict]:
    """Per-question signals + whether the answer was correct.

    This deliberately does **not** go through ``BanglaQA.ask``: fusion is what
    we are about to fit, so the pipeline's fused confidence does not exist yet.
    Everything upstream of fusion is identical.
    """
    out: list[dict] = []
    for i, row in enumerate(rows):
        hits = retriever.search(row["question"], CFG.reader_top_k)
        passages = []
        for rank, (pid, score) in enumerate(hits, start=1):
            p = by_pid.get(pid)
            if p is not None:
                passages.append({**p, "rank": rank, "retriever_score": float(score)})
        if not passages:
            continue

        ranks = [(p["rank"], p["retriever_score"]) for p in passages]
        ans, ens = reader.read_with_ensemble(row["question"], passages, ranks=ranks,
                                             question_type=row.get("question_type"))
        evidence = ans.sentence or passages[0]["text"][:400]

        alt = [reader.read(row["question"], [p], ranks=[(p["rank"], p["retriever_score"])])
               for p in passages[1:4]]

        ctx = SignalContext(question=row["question"], answer=ans, evidence=evidence,
                            passages=passages, alt_answers=alt, ensemble_answers=ens,
                            support_model=support, gazetteer=gazetteer)
        feats = fusion_features(ctx)
        signals = {k: feats[k] for k in SIGNAL_NAMES}
        veto = veto_check(ans.text, evidence, gazetteer=gazetteer)

        golds = [a["text"] for a in row.get("answers", []) if a.get("text")]
        f1 = max((token_f1(ans.text, g) for g in golds), default=0.0)
        # An unanswerable question has no correct span, so "correct" means the
        # system should not have answered at all.  Encoding it as f1=0 keeps
        # one definition of error across the whole selective-risk computation.
        is_correct = correctness(f1) if row.get("is_answerable") else False

        out.append({
            "qid": row["qid"], "question": row["question"], "answer": ans.text,
            "gold": golds[0] if golds else "", "evidence": evidence, "pid": ans.pid,
            "is_answerable": bool(row.get("is_answerable")),
            "token_f1": round(f1, 4), "correct": bool(is_correct),
            "veto_fired": bool(veto.fired), "veto_reason": veto.reason,
            "features": feats, "signals": signals,
        })
        if verbose and (i + 1) % 100 == 0:
            print(f"    {i + 1:5,d}/{len(rows):,d}")
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=CFG.headline_index)
    ap.add_argument("--eval", type=int, default=CFG.verify_eval_questions)
    ap.add_argument("--bv-questions", type=int, default=CFG.banglaverify_max_questions)
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--skip-banglaverify", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    print(f"T11  verifier  (index {args.size // 1000}k)")

    # ---- T11b: BanglaVerify + S3 ----------------------------------------
    if not args.skip_banglaverify or not SUPPORT_MODEL.exists():
        print("\n=== T11b  BanglaVerify " + "=" * 40)
        bv.build(limit=args.bv_questions)

    bv_train, bv_val = bv.load("train"), bv.load("val")
    bv_contrast = bv.load("contrast")
    print(f"\n  S3: fitting on {len(bv_train):,d} BanglaVerify items")
    with timer("fit S3"):
        support = SupportClassifier().fit(bv_train)
    support.save(SUPPORT_MODEL)

    s3_val = support.evaluate(bv_val)
    s3_contrast = support.evaluate(bv_contrast)
    print(f"    val accuracy      {s3_val['accuracy']:.4f} "
          f"(majority {s3_val['majority_baseline']:.4f})")
    print(f"    contrast accuracy {s3_contrast['accuracy']:.4f} "
          f"(majority {s3_contrast['majority_baseline']:.4f})  <- scored separately, §10.7")
    save_table("s3_support", [
        {"split": "val", **{k: v for k, v in s3_val.items() if k != "by_generator"}},
        {"split": "contrast", **{k: v for k, v in s3_contrast.items()
                                 if k != "by_generator"}},
    ])

    contrast_rows = contrast_ablation(bv_contrast, support)
    save_table("verify_contrast", contrast_rows)
    on = next((r for r in contrast_rows if r["arm"] == "veto_on"), None)
    off = next((r for r in contrast_rows if r["arm"] == "veto_off"), None)
    if on and off:
        print(f"    contrast set, veto OFF {off['accuracy']:.4f} -> "
              f"veto ON {on['accuracy']:.4f}  "
              f"(Δ {on['accuracy'] - off['accuracy']:+.4f})")

    # ---- the pipeline components ----------------------------------------
    print("\n=== signals over val " + "=" * 42)
    passages = prepare(args.size)
    by_pid = {p["pid"]: {**p, "text": normalize(p["text"])} for p in passages}
    with timer("build retriever"):
        retriever = build_retriever(args.retriever, passages)

    if not READER_MODEL.exists():
        raise SystemExit("no fitted reader - run scripts/run_reader.py first")
    ranker = SpanRanker.load(READER_MODEL)
    qtype_clf = pickle.load(open(QTYPE_MODEL, "rb")) if QTYPE_MODEL.exists() else None
    ensemble = [SpanRanker.load(p) for p in sorted(READER_MODEL.parent.glob("reader_bag_*.pkl"))]
    reader = FeatureReader(ranker, qtype_clf=qtype_clf, ensemble=ensemble)
    print(f"  reader {ranker.name} · {len(ensemble)} bagged rankers for S5")

    from bnqa.pipeline import _gazetteer

    gazetteer = _gazetteer(passages)

    val_rows = load_queries("val", answerable_only=False)[:args.eval]
    with timer("collect val signals"):
        val = collect(val_rows, by_pid, retriever, reader, ensemble, support, gazetteer)
    print(f"  {len(val):,d} val items · {sum(r['correct'] for r in val):,d} correct "
          f"({100 * np.mean([r['correct'] for r in val]):.1f}%)")

    # ---- the val split: fusion half / conformal-calibration half ---------
    cut = int(len(val) * CFG.conformal_calibration_frac)
    fit_half, cal_half = val[:cut], val[cut:]
    print(f"  val split: {len(fit_half):,d} for fusion+calibration, "
          f"{len(cal_half):,d} held for the conformal threshold")

    X_fit = to_matrix([r["features"] for r in fit_half])
    y_fit = np.asarray([r["correct"] for r in fit_half], dtype=int)
    X_cal = to_matrix([r["features"] for r in cal_half])
    y_cal = np.asarray([r["correct"] for r in cal_half], dtype=int)

    # ---- RQ5: fusion comparison -----------------------------------------
    print("\n=== RQ5  fusion " + "=" * 47)
    fusion_rows, fitted = compare_fusion(X_fit, y_fit, X_cal, y_cal)
    for row in fusion_rows:
        print(f"    {row['fusion']:14s} AUROC {row['auroc']:.4f} · "
              f"AUPRC {row['auprc']:.4f} · Brier {row['brier']:.4f}")
    save_table("fusion", fusion_rows)
    best_kind = fusion_rows[0]["fusion"]
    fusion = fitted[best_kind]
    fusion.save(FUSION_MODEL)
    print(f"    -> {best_kind} wins on AUROC and is the shipped fusion model")
    if fusion.coefficients():
        save_table("fusion_coefficients", fusion.coefficients())

    # ---- calibration study ----------------------------------------------
    print("\n=== calibration " + "=" * 47)
    p_fit = fusion.predict_proba(X_fit)
    p_cal = fusion.predict_proba(X_cal)
    cal_rows, calibrators = compare_calibrators(p_fit, y_fit, p_cal, y_cal)
    for row in cal_rows:
        print(f"    {row['method']:14s} ECE {row['ece']:.4f} · ACE {row['ace']:.4f} · "
              f"Brier {row['brier']:.4f}")
    save_table("calibration", cal_rows)
    best_cal = cal_rows[0]["method"]
    calibrator = calibrators[best_cal]
    with open(CALIBRATOR, "wb") as fh:
        pickle.dump(calibrator, fh)
    print(f"    -> {best_cal} has the lowest ECE and is the shipped calibrator")

    fig = reliability_figure({m: (calibrators[m](p_cal), y_cal.astype(float))
                              for m in ("uncalibrated", best_cal)})
    print(f"    reliability diagram -> {fig}")
    save_table("reliability", reliability_table(calibrator(p_cal), y_cal.astype(bool)))

    # ---- conformal threshold on the held-out half ------------------------
    print("\n=== RQ8  conformal selective risk " + "=" * 29)
    conf_cal = calibrator(p_cal)
    conf_cal = np.asarray([min(c, 0.15) if r["veto_fired"] else c
                           for c, r in zip(conf_cal, cal_half)])
    threshold, curve = select_threshold(conf_cal, [r["correct"] for r in cal_half])
    save_table("conformal_curve", curve)
    if threshold.feasible:
        print(f"    τ = {threshold.tau:.4f} · coverage {threshold.coverage:.4f} · "
              f"empirical risk {threshold.empirical_risk:.4f} · "
              f"UCB {threshold.risk_upper_bound:.4f} ≤ α {threshold.alpha}")
    else:
        print(f"    NO FEASIBLE τ: the ≤{100 * threshold.alpha:.0f}% promise cannot be "
              f"met on this calibration split, and the report says so.")
    write_json(CONFORMAL_JSON, threshold.as_dict())

    # ---- RQ8b: the four abstention policies ------------------------------
    policies = {
        "a_reader_only": ([r["signals"]["s1_reader"] for r in cal_half],
                          [r["correct"] for r in cal_half]),
        "b_fused": (list(calibrator(p_cal)), [r["correct"] for r in cal_half]),
        "c_fused_veto": (list(conf_cal), [r["correct"] for r in cal_half]),
    }
    policy_rows, curves = compare_policies(policies)
    for row in policy_rows:
        print(f"    {row['policy']:16s} RC-AUC {row['rc_auc']:.4f} · "
              f"risk@80% coverage {row['risk_at_80pct_coverage']}")

    # ---- test, opened exactly once ---------------------------------------
    print("\n=== test (opened once) " + "=" * 40)
    test_rows = load_queries("test", answerable_only=False)[:args.eval]
    with timer("collect test signals"):
        test = collect(test_rows, by_pid, retriever, reader, ensemble, support, gazetteer)
    X_test = to_matrix([r["features"] for r in test])
    p_test = calibrator(fusion.predict_proba(X_test))
    p_test = np.asarray([min(c, 0.15) if r["veto_fired"] else c
                         for c, r in zip(p_test, test)])
    y_test = [r["correct"] for r in test]

    achieved = conformal_verify(threshold, p_test, y_test)
    print(f"    {achieved['claim']}")
    print(f"    achieved: selective risk {achieved['achieved_selective_risk']:.4f} at "
          f"coverage {achieved['coverage']:.4f} — bound "
          f"{'HELD' if achieved['bound_held'] else 'VIOLATED'}")
    save_table("conformal", [{**threshold.as_dict(), **achieved}])

    policy_rows_test, curves_test = compare_policies({
        "d_conformal(test)": (list(p_test), y_test), **{
            "b_fused(test)": (list(calibrator(fusion.predict_proba(X_test))), y_test)}})
    save_table("abstention_policies", policy_rows + policy_rows_test)
    fig2 = risk_coverage_figure({**curves, **curves_test}, tau_point=achieved)
    print(f"    risk–coverage figure -> {fig2}")

    cal_table = [calibration_row("val(calibrated)", calibrator(p_cal), y_cal.astype(bool)),
                 calibration_row("test(calibrated)", p_test, np.asarray(y_test))]
    save_table("verify_calibration_summary", cal_table)
    for row in cal_table:
        print(f"    {row['arm']:18s} ECE {row['ece']:.4f} · ACE {row['ace']:.4f} · "
              f"AUROC(conf vs correct) {row['auroc_conf_vs_correct']:.4f}")

    log_result("t11_verify", "summary", {
        "s3": {"val": s3_val, "contrast": s3_contrast},
        "contrast_ablation": contrast_rows,
        "fusion": fusion_rows, "fusion_chosen": best_kind,
        "calibration": cal_rows, "calibrator_chosen": best_cal,
        "conformal": {**threshold.as_dict(), **achieved},
        "policies": policy_rows + policy_rows_test,
        "calibration_summary": cal_table,
        "n_val": len(val), "n_test": len(test),
    })
    print("\nT11 complete.")


if __name__ == "__main__":
    main()
