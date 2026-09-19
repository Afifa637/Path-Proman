"""T10: train the span reader, evaluate it, and write the feature table.

    python scripts/run_reader.py                      # full run
    python scripts/run_reader.py --train 300 --eval 100   # quick pass

Produces
  * ``reports/tables/reader.csv``           — metric-ladder tiers 1-3 + HasAns/NoAns F1
  * ``reports/tables/reader_features.csv``  — the feature-importance table (§7.6)
  * ``reports/tables/qtype.csv``            — the 4-way question-type head
  * ``data/processed/models/reader_*.pkl``  — the fitted rankers and the S5 bag

Discipline (PLAN.md §16.1): the reader trains on **train**, every number below
is measured on **val**, and test is not opened by this script at all.

Training uses **gold-passage teacher forcing** — the gold passage is added to
the candidate set even when retrieval missed it.  Without it the reader would
only ever be trained on questions retrieval already solved, which both shrinks
the training set and teaches it that the top-ranked passage is always right.
Evaluation uses the retriever's real output, so the number is honest.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bnqa.config import (CFG, MODELS, QTYPE_MODEL, READER_GBDT_MODEL,  # noqa: E402
                         READER_MODEL, ensure_dirs)
from bnqa.eval.qa_metrics import aggregate, score_all, set_f1  # noqa: E402
from bnqa.eval.retrieval_metrics import load_queries  # noqa: E402
from bnqa.preprocess.normalize import normalize  # noqa: E402
from bnqa.reader.candidates import IdfTable, generate, gold_candidate_index  # noqa: E402
from bnqa.reader.features import FEATURE_NAMES, extract, to_matrix  # noqa: E402
from bnqa.reader.qtype import QuestionTypeClassifier, analyze  # noqa: E402
from bnqa.reader.span_ranker import FeatureReader, SpanRanker  # noqa: E402
from bnqa.retrieval.index import prepare  # noqa: E402
from bnqa.utils import load_jsonl, log_result, save_table, set_seed, timer  # noqa: E402
from bnqa.pipeline import build_retriever  # noqa: E402


# --------------------------------------------------------------------------- #
# Training data                                                                #
# --------------------------------------------------------------------------- #


def gold_spans(row: dict) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for ans in row.get("answers", []):
        if ans.get("answer_type") == "yes/no" or not ans.get("aligned"):
            continue
        for span in ans.get("spans") or []:
            if span.get("start") is not None:
                out.append((span.get("text") or ans["text"], span["start"], span["end"]))
    return out


def build_training_set(rows, by_pid, retriever, *, rng, verbose=True):
    """``(X, y, groups)`` over candidate spans, with negatives subsampled."""
    X_rows: list[dict] = []
    y: list[int] = []
    groups: list[int] = []
    n_pos = n_neg = skipped = 0

    for qi, row in enumerate(rows):
        spans = gold_spans(row)
        if not spans:
            skipped += 1
            continue
        gold_pid = row["gold_passage_id"]
        gold_passage = by_pid.get(gold_pid)
        if gold_passage is None:
            skipped += 1
            continue

        hits = retriever.search(row["question"], CFG.reader_top_k)
        passages: list[dict] = []
        for rank, (pid, score) in enumerate(hits, start=1):
            p = by_pid.get(pid)
            if p is not None:
                passages.append({**p, "rank": rank, "retriever_score": float(score)})
        # teacher forcing
        if all(p["pid"] != gold_pid for p in passages):
            passages.append({**gold_passage, "rank": len(passages) + 1,
                             "retriever_score": 0.0})

        qa = analyze(row["question"], row.get("question_type"))
        ranks = [(p["rank"], p["retriever_score"]) for p in passages]
        cands = generate(row["question"], passages, expected_class=qa.expected_class,
                         ranks=ranks)
        if not cands:
            skipped += 1
            continue

        positives = set()
        for _text, s, e in spans:
            idx = gold_candidate_index(cands, gold_pid, s, e)
            if idx >= 0:
                positives.add(idx)
        if not positives:
            # The gold span is not among the candidates, so this question can
            # teach the ranker nothing.  It is counted, not silently dropped:
            # candidate recall is the reader's ceiling and it belongs in the
            # results table, not in a comment.
            skipped += 1
            continue

        negatives = [i for i in range(len(cands)) if i not in positives]
        rng.shuffle(negatives)
        keep = sorted(positives) + negatives[:CFG.reader_negatives_per_question]

        idf = IdfTable([p["text"] for p in passages])
        for i in keep:
            c = cands[i]
            X_rows.append(extract(row["question"], c, idf=idf,
                                  expected_class=qa.expected_class,
                                  question_type=qa.question_type))
            y.append(1 if i in positives else 0)
            groups.append(qi)
            if i in positives:
                n_pos += 1
            else:
                n_neg += 1

        if verbose and (qi + 1) % 250 == 0:
            print(f"    {qi + 1:5,d} questions -> {len(y):,d} rows "
                  f"({n_pos:,d} positive)")

    usable = len(rows) - skipped
    recall = usable / max(len(rows), 1)
    if verbose:
        print(f"  training rows: {len(y):,d} ({n_pos:,d} positive, {n_neg:,d} negative)")
        print(f"  candidate recall: {usable:,d}/{len(rows):,d} = {recall:.3f} "
              f"— the reader's ceiling.  {skipped:,d} questions have no gold span "
              f"among their candidates and can teach it nothing.")
    return to_matrix(X_rows), np.asarray(y), groups, {
        "skipped": skipped, "positives": n_pos, "negatives": n_neg,
        "questions": len(rows), "usable_questions": usable,
        "candidate_recall": round(recall, 4),
        "cand_sentences": CFG.cand_sentences, "cand_ngram_max": CFG.cand_ngram_max}


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #


def evaluate(reader: FeatureReader, rows, by_pid, retriever, *, verbose=True) -> dict:
    """Metric-ladder tiers 1-3 on the retriever's real output."""
    per_item: list[dict] = []
    for i, row in enumerate(rows):
        hits = retriever.search(row["question"], CFG.reader_top_k)
        passages = []
        for rank, (pid, score) in enumerate(hits, start=1):
            p = by_pid.get(pid)
            if p is not None:
                passages.append({**p, "rank": rank, "retriever_score": float(score)})
        if not passages:
            per_item.append({"qid": row["qid"], "is_answerable": row["is_answerable"],
                             "abstained": True, "tier1_em": 0.0, "tier2_token_f1": 0.0,
                             "tier3_stem_syn_f1": 0.0})
            continue

        ranks = [(p["rank"], p["retriever_score"]) for p in passages]
        ans = reader.read(row["question"], passages, ranks=ranks,
                          question_type=row.get("question_type"))
        golds = [a["text"] for a in row.get("answers", []) if a.get("text")]

        if ans.route == "list" and ans.spans:
            f1 = set_f1([s[0] for s in ans.spans], golds)
            scored = {"tier1_em": 0.0, "tier2_token_f1": f1, "tier3_stem_syn_f1": f1}
        else:
            scored = score_all(ans.text, golds, evidence=ans.sentence)

        per_item.append({
            "qid": row["qid"], "is_answerable": row.get("is_answerable", True),
            "abstained": False, "route": ans.route,
            "expected_class": ans.expected_class, "pid": ans.pid,
            "gold_pid": row["gold_passage_id"],
            "retrieved_gold": any(p["pid"] == row["gold_passage_id"] for p in passages),
            "answer": ans.text, "gold": golds[0] if golds else "",
            "score": ans.score, "margin": ans.margin, **scored,
        })
        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1:5,d}/{len(rows):,d}")
    return {"per_item": per_item, **aggregate(per_item)}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=CFG.headline_index)
    ap.add_argument("--train", type=int, default=CFG.reader_train_questions)
    ap.add_argument("--eval", type=int, default=CFG.reader_eval_questions)
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--no-bag", action="store_true", help="skip the S5 ensemble")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    rng = random.Random(CFG.seed)
    print(f"T10  reader  (index {args.size // 1000}k, retriever {args.retriever})")

    passages = prepare(args.size)
    by_pid = {p["pid"]: {**p, "text": normalize(p["text"])} for p in passages}
    with timer("build retriever"):
        retriever = build_retriever(args.retriever, passages)

    train_rows = [r for r in load_queries("train") if r.get("is_answerable")][:args.train]
    val_rows = load_queries("val", answerable_only=False)[:args.eval]
    print(f"  {len(train_rows):,d} train questions · {len(val_rows):,d} val questions")

    # ---- the 4-way question-type head -----------------------------------
    print("\n  question-type head (4-way, BanglaRQA labels)")
    all_train = load_queries("train", answerable_only=False)
    labelled = [(r["question"], r["question_type"]) for r in all_train if r.get("question_type")]
    qtype_clf = QuestionTypeClassifier().fit([q for q, _ in labelled],
                                             [t for _, t in labelled])
    val_labelled = [(r["question"], r["question_type"]) for r in
                    load_queries("val", answerable_only=False) if r.get("question_type")]
    preds = qtype_clf.predict([q for q, _ in val_labelled])
    acc = float(np.mean([p == t for p, (_, t) in zip(preds, val_labelled)]))
    rule_only = [analyze(q).route for q, _ in val_labelled]
    print(f"    trained on {len(labelled):,d} · val accuracy {acc:.4f}")
    qtype_rows = [{"n_train": len(labelled), "n_val": len(val_labelled),
                   "accuracy": round(acc, 4),
                   "routes_polarity": sum(1 for r in rule_only if r == "polarity"),
                   "routes_list": sum(1 for r in rule_only if r == "list"),
                   "routes_span": sum(1 for r in rule_only if r == "span")}]
    save_table("qtype", qtype_rows)
    with open(QTYPE_MODEL, "wb") as fh:
        pickle.dump(qtype_clf, fh)

    # ---- training data ---------------------------------------------------
    print("\n  building candidate training set")
    with timer("candidates"):
        X, y, groups, stats = build_training_set(train_rows, by_pid, retriever, rng=rng)

    if not len(y) or y.sum() == 0:
        raise SystemExit("no positive training candidates - check the alignment audit")

    # ---- fit both models -------------------------------------------------
    table: list[dict] = []
    feature_rows: list[dict] = []
    fitted: dict[str, SpanRanker] = {}
    for kind, path in (("logreg", READER_MODEL), ("gbdt", READER_GBDT_MODEL)):
        print(f"\n  fitting {kind}")
        with timer(f"fit {kind}"):
            ranker = SpanRanker(kind).fit(X, y, groups)
        ranker.save(path)
        fitted[kind] = ranker
        feature_rows.extend(ranker.feature_importance())

        reader = FeatureReader(ranker, qtype_clf=qtype_clf)
        res = evaluate(reader, val_rows, by_pid, retriever)
        row = {"reader": ranker.name, "split": "val", "index_size": args.size,
               "retriever": retriever.name,
               "candidate_recall_train": stats["candidate_recall"],
               **{k: round(v, 4) for k, v in res.items() if k != "per_item"}}
        table.append(row)
        print(f"    tier1 EM {row['tier1_em']:.4f} · tier2 F1 {row['tier2_token_f1']:.4f} "
              f"(headline) · tier3 F1 {row['tier3_stem_syn_f1']:.4f}")
        if "hasans_f1" in row:
            print(f"    HasAns-F1 {row['hasans_f1']:.4f} · "
                  f"NoAns-F1 {row.get('noans_f1', float('nan')):.4f}")
        _save_predictions(kind, res["per_item"])

    # ---- permutation importance for the GBDT ----------------------------
    print("\n  permutation importance (gbdt, on a val slice)")
    feature_rows.extend(_permutation_importance(fitted["gbdt"], X, y))

    save_table("reader", table)
    save_table("reader_features", feature_rows)

    top = [r for r in feature_rows if r["model"] == "reader-logreg"][:10]
    print("  top features (logreg):")
    for r in top:
        print(f"    {r['rank']:2d}. {r['feature']:26s} {r['weight']:+.4f}")

    # ---- the S5 bag ------------------------------------------------------
    if not args.no_bag:
        print(f"\n  bagging {CFG.reader_ensemble_size} rankers for S5")
        _fit_bag(X, y, groups, rng)

    log_result("t10_reader", "val", {"rows": table, "training": stats,
                                     "qtype": qtype_rows[0],
                                     "train_questions": len(train_rows),
                                     "eval_questions": len(val_rows)})


def _save_predictions(kind: str, per_item: list[dict]) -> None:
    from bnqa.config import TABLES

    import pandas as pd

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_item).to_csv(TABLES / f"reader_predictions_{kind}.csv",
                                  index=False, encoding="utf-8")


def _permutation_importance(ranker: SpanRanker, X, y, n_repeats: int = 3) -> list[dict]:
    """Importance for the GBDT, which has no readable coefficients."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    # A slice keeps this to seconds; the ranking is stable well before the
    # full matrix, and this is an explanatory table, not a headline metric.
    n = min(len(y), 20_000)
    idx = np.random.default_rng(CFG.seed).choice(len(y), size=n, replace=False)
    res = permutation_importance(ranker.clf, X[idx], y[idx], n_repeats=n_repeats,
                                 random_state=CFG.seed,
                                 scoring=lambda est, Xs, ys: roc_auc_score(
                                     ys, est.predict_proba(Xs)[:, 1]))
    rows = [{"feature": name, "weight": float(m), "abs_weight": float(abs(m)),
             "model": "reader-gbdt(permutation)"}
            for name, m in zip(FEATURE_NAMES, res.importances_mean)]
    rows.sort(key=lambda r: -r["abs_weight"])
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def _fit_bag(X, y, groups, rng) -> None:
    """Bagged rankers over subsampled rows and features — S5's ensemble."""
    n = len(y)
    for b in range(CFG.reader_ensemble_size):
        seed = CFG.seed + 101 * (b + 1)
        sub = np.random.default_rng(seed).choice(n, size=int(0.8 * n), replace=True)
        ranker = SpanRanker("logreg", name=f"reader-bag{b}").fit(X[sub], y[sub])
        ranker.save(MODELS / f"reader_bag_{b}.pkl")
    print(f"    -> {CFG.reader_ensemble_size} bagged rankers in {MODELS}")


if __name__ == "__main__":
    main()
