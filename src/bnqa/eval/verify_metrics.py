"""Verifier evaluation (PLAN.md §10.7, task T11's acceptance criterion).

T11 is done when **all** of these are in ``reports/``: a reliability diagram, a
risk-coverage curve, ECE/ACE, the AUROC of confidence-vs-correctness, the
conformal bound with its achieved test risk, and **contrast-set accuracy with
the veto layer on vs off**.

The last one carries the argument.  §10.7: the held-out contrast set is scored
**separately** from the aggregate, because near-chance accuracy there means the
verifier is only a similarity detector — and the report has to say so rather
than hide it inside an average.  The veto-on/veto-off ablation on that set is
the second half of the project's headline sentence.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import CFG
from ..verify.calibration import (adaptive_calibration_error,
                                  expected_calibration_error, reliability_bins)


def correctness(token_f1: float) -> bool:
    """Is this answer 'correct' for the purposes of selective risk?

    Selective risk needs a binary notion of error, and EM is too strict to be
    meaningful in Bangla (§9) while a soft tier is too generous to promise a
    bound on.  Tier-2 token F1 above ``CFG.correctness_threshold`` is the
    compromise, it is stated here rather than buried, and the threshold is a
    config value so the sensitivity can be re-run.
    """
    return float(token_f1) >= CFG.correctness_threshold


def confidence_auroc(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Does confidence rank correct answers above wrong ones?  0.5 is useless."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(correct, dtype=int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(confidence, dtype=float)))


def calibration_row(name: str, confidence: Sequence[float],
                    correct: Sequence[bool]) -> dict:
    p = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    from sklearn.metrics import brier_score_loss

    return {
        "arm": name,
        "n": len(p),
        "accuracy": round(float(y.mean()), 4),
        "mean_confidence": round(float(p.mean()), 4),
        "ece": round(expected_calibration_error(p, y), 4),
        "ace": round(adaptive_calibration_error(p, y), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "auroc_conf_vs_correct": round(confidence_auroc(p, y > 0.5), 4),
    }


# --------------------------------------------------------------------------- #
# The contrast-set ablation — the headline                                     #
# --------------------------------------------------------------------------- #


def contrast_ablation(items: Sequence[dict], support_model) -> list[dict]:
    """Contrast-set accuracy **with the veto layer on and off**.

    Veto-off is the classifier's own three-way prediction.  Veto-on overrides
    it to CONTRADICTED whenever the veto layer fires, which is exactly what the
    pipeline does.  The difference between the two rows is what the veto layer
    bought, per generator — and if it is near zero, the report says the
    contribution did not materialise rather than quietly dropping the table.
    """
    from ..verify.banglaverify import CONTRADICTED, LABELS
    from ..verify.constraints import check_passage

    if not items:
        return []

    gold = [i["label"] for i in items]
    if support_model is not None:
        proba = support_model.predict_proba([(i["statement"], i["evidence"]) for i in items])
        base = [support_model.classes_[int(k)] for k in np.argmax(proba, axis=1)]
    else:
        base = [CONTRADICTED] * len(items)

    fired = [check_passage(i["statement"], i["evidence"]).fired for i in items]
    with_veto = [CONTRADICTED if f else b for b, f in zip(base, fired)]

    rows: list[dict] = []
    for arm, pred in (("veto_off", base), ("veto_on", with_veto)):
        row = {"arm": arm, "n": len(items),
               "accuracy": round(float(np.mean([p == g for p, g in zip(pred, gold)])), 4)}
        for label in LABELS:
            idx = [k for k, g in enumerate(gold) if g == label]
            if idx:
                row[f"acc_{label.lower()}"] = round(
                    float(np.mean([pred[k] == gold[k] for k in idx])), 4)
        rows.append(row)

    # majority-class baseline: "near chance" needs a number to be near.
    counts = {label: gold.count(label) for label in set(gold)}
    rows.append({"arm": "majority_baseline", "n": len(items),
                 "accuracy": round(max(counts.values()) / len(gold), 4)})

    by_gen: dict[str, list[int]] = {}
    for k, item in enumerate(items):
        by_gen.setdefault(item.get("generator", "?"), []).append(k)
    for gen, idx in sorted(by_gen.items()):
        rows.append({
            "arm": f"veto_on/{gen}", "n": len(idx),
            "accuracy": round(float(np.mean([with_veto[k] == gold[k] for k in idx])), 4),
            "accuracy_veto_off": round(
                float(np.mean([base[k] == gold[k] for k in idx])), 4),
            "veto_fire_rate": round(float(np.mean([fired[k] for k in idx])), 4),
        })
    return rows


# --------------------------------------------------------------------------- #
# Reliability table (the numbers behind the diagram)                           #
# --------------------------------------------------------------------------- #


def reliability_table(confidence: Sequence[float], correct: Sequence[bool],
                      bins: int = 10) -> list[dict]:
    return reliability_bins(np.asarray(confidence, dtype=float),
                            np.asarray(correct, dtype=float), bins)
