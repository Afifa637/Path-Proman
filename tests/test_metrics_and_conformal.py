"""The metric ladder's guards (§9) and the conformal guarantee (§10.6).

Two properties carry real weight here:

* **Guard 1** — tier 3 must not grant soft credit to a vetoed answer.  Without
  it, synonym matching makes a *contradiction* score like a paraphrase, and
  evaluation gets worse while looking better.
* **The bound** — a Clopper-Pearson upper bound must actually be an upper
  bound.  If ``select_threshold`` returned a τ whose UCB exceeded α the
  guarantee in the report would be false, so it is asserted rather than
  assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bnqa.eval.qa_metrics import (aggregate, exact_match, score_all, set_f1,
                                  tier3_f1, token_f1)
from bnqa.verify.calibration import (adaptive_calibration_error,
                                     expected_calibration_error, fit_calibrator)
from bnqa.verify.conformal import (clopper_pearson_upper, curve_auc,
                                   risk_coverage_curve, select_threshold, verify)


# --------------------------------------------------------------------------- #
# The ladder                                                                   #
# --------------------------------------------------------------------------- #


def test_exact_match_is_normalisation_insensitive():
    assert exact_match("১৯৭১", "1971") == 1.0
    assert exact_match("১৯৭১", "১৯৫২") == 0.0


def test_token_f1_partial_credit():
    assert token_f1("কার্বন ডাই-অক্সাইড", "কার্বন ডাই-অক্সাইড") == pytest.approx(1.0)
    assert 0 < token_f1("কার্বন", "কার্বন ডাই-অক্সাইড") < 1


def test_tier3_never_falls_below_tier2():
    pred, gold = "উদ্ভিদের পাতা", "উদ্ভিদ পাতা"
    assert tier3_f1(pred, gold) >= token_f1(pred, gold)


def test_guard_one_tier3_is_veto_gated():
    """A contradiction must not be rewarded as a paraphrase."""
    pred = "মুক্তিযুদ্ধ ১৯৫২ সালে সংঘটিত হয়"
    gold = "মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়"
    evidence = gold

    ungated = tier3_f1(pred, gold)                      # no evidence -> no gate
    gated = tier3_f1(pred, gold, evidence=evidence)     # gate applies
    assert gated == pytest.approx(token_f1(pred, gold))
    assert gated <= ungated


def test_guard_two_every_tier_is_returned_together():
    out = score_all("১৯৭১", ["১৯৭১"])
    assert set(out) == {"tier1_em", "tier2_token_f1", "tier3_stem_syn_f1"}


def test_set_f1_ignores_order_and_penalises_extras():
    assert set_f1(["ক", "খ"], ["খ", "ক"]) == pytest.approx(1.0)
    assert set_f1(["ক", "খ", "গ"], ["ক", "খ"]) < 1.0


def test_aggregate_splits_hasans_and_noans():
    rows = [
        {"is_answerable": True, "abstained": False, "tier2_token_f1": 1.0},
        {"is_answerable": True, "abstained": False, "tier2_token_f1": 0.0},
        {"is_answerable": False, "abstained": True, "tier2_token_f1": 0.0},
    ]
    out = aggregate(rows)
    assert out["hasans_f1"] == pytest.approx(0.5)
    assert out["noans_f1"] == pytest.approx(1.0)
    assert out["abstention_rate"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# Conformal                                                                    #
# --------------------------------------------------------------------------- #


def test_clopper_pearson_is_an_upper_bound():
    for errors, n in ((0, 10), (1, 50), (5, 100), (25, 100)):
        ucb = clopper_pearson_upper(errors, n, 0.10)
        assert ucb >= errors / n
        assert 0.0 <= ucb <= 1.0


def test_clopper_pearson_tightens_with_more_data():
    assert clopper_pearson_upper(1, 20, 0.1) > clopper_pearson_upper(50, 1000, 0.1)


def test_selected_threshold_respects_the_bound():
    rng = np.random.default_rng(0)
    n = 600
    conf = rng.uniform(0, 1, n)
    # correctness correlated with confidence, which is what a working verifier
    # produces and what the procedure is supposed to exploit
    correct = rng.uniform(0, 1, n) < conf
    threshold, curve = select_threshold(conf, correct, alpha=0.10, delta=0.10)

    assert threshold.feasible
    assert threshold.risk_upper_bound <= 0.10
    assert 0.0 < threshold.coverage <= 1.0
    # and it is the *most permissive* feasible τ (the curve rounds to 4dp)
    feasible = [r for r in curve if r["feasible"]]
    assert threshold.coverage == pytest.approx(max(r["coverage"] for r in feasible),
                                               abs=1e-4)


def test_infeasible_when_nothing_can_meet_the_bound():
    conf = np.linspace(0.4, 0.6, 200)
    correct = np.zeros(200, dtype=bool)      # everything is wrong
    threshold, _ = select_threshold(conf, correct, alpha=0.10, delta=0.10)
    assert not threshold.feasible, "an impossible promise must be reported, not faked"


def test_verify_reports_achieved_risk():
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    correct = [True, True, False, False, False]
    threshold, _ = select_threshold(conf, correct, alpha=0.5, delta=0.5)
    out = verify(threshold, conf, correct)
    assert out["answered"] <= len(conf)
    assert 0.0 <= out["achieved_selective_risk"] <= 1.0
    assert "confidence" in out["claim"]


def test_risk_coverage_curve_is_monotone_in_the_right_direction():
    conf = np.linspace(1.0, 0.0, 100)
    correct = np.arange(100) < 70          # the confident ones are the right ones
    curve = risk_coverage_curve(conf, correct)
    assert curve[0]["selective_risk"] <= curve[-1]["selective_risk"]
    assert 0.0 <= curve_auc(curve) <= 1.0


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #


def test_calibration_error_is_zero_for_a_perfect_model():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(p, y) == pytest.approx(0.0, abs=1e-9)
    assert adaptive_calibration_error(p, y) == pytest.approx(0.0, abs=1e-9)


def test_isotonic_calibration_reduces_ece_on_a_skewed_model():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=2000) < 0.5).astype(float)
    # a badly over-confident score
    p = np.clip(y * 0.95 + rng.normal(0, 0.25, 2000), 0.01, 0.99)
    before = expected_calibration_error(p, y)
    cal = fit_calibrator("isotonic", p, y)
    after = expected_calibration_error(cal(p), y)
    assert after <= before + 1e-9
