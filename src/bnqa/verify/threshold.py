"""The abstention decision (PLAN.md §10.6, RQ8b).

Four policies, in increasing order of what they know, so the ablation grid can
say what each layer bought:

==== ==================== ======================================================
arm  policy               decision rule
==== ==================== ======================================================
a    ``reader``           S1 alone above a fixed threshold
b    ``fused``            the fusion model's probability above a fixed threshold
c    ``fused_veto``       (b), and abstain outright when the veto layer fires
d    ``conformal``        (c), with τ chosen by the conformal procedure
==== ==================== ======================================================

Arm (d) is the one that ships.  The others exist because "the veto layer
helps" and "the guarantee costs coverage" are claims that need numbers.

An abstention is never silent.  ``Decision`` carries the reason and the closest
passage anyway, because VC-5's demo is *"the system abstains **and shows you
what it looked at**"* — a refusal with no evidence is indistinguishable from a
crash to the person watching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conformal import ConformalThreshold
from .constraints import VetoResult

POLICIES = ("reader", "fused", "fused_veto", "conformal")

DEFAULT_THRESHOLD = 0.5

ABSTAIN_MESSAGE = (
    "এই প্রশ্নের উত্তর কর্পাসে পাওয়া যায়নি — তাই উত্তর দেওয়া হচ্ছে না।"
)
ABSTAIN_MESSAGE_EN = "No supporting evidence in the corpus — abstaining."


@dataclass
class Decision:
    answered: bool
    confidence: float
    policy: str
    tau: float
    reason: str = ""
    veto_reason: str = ""
    message: str = ""
    signals: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def decide(confidence: float, veto: VetoResult, *, policy: str = "conformal",
           threshold: ConformalThreshold | None = None,
           fixed_tau: float = DEFAULT_THRESHOLD, signals: dict | None = None) -> Decision:
    """Answer or abstain, with the reason attached either way."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")

    tau = fixed_tau
    if policy == "conformal":
        if threshold is None or not threshold.feasible:
            # No feasible τ means the guarantee could not be met on the
            # calibration split.  Falling back to the fixed threshold is
            # correct, but the decision has to say so rather than pretend a
            # guarantee is in force.
            tau = fixed_tau
            note = "conformal τ unavailable — fixed threshold in force"
        else:
            tau = threshold.tau
            note = (f"conformal τ={tau:.3f} "
                    f"(≤{100 * threshold.alpha:.0f}% error at "
                    f"{100 * (1 - threshold.delta):.0f}% confidence)")
    else:
        note = f"fixed τ={tau:.3f}"

    if policy in ("fused_veto", "conformal") and veto.fired:
        return Decision(answered=False, confidence=float(confidence), policy=policy,
                        tau=tau, reason="veto", veto_reason=veto.reason,
                        message=f"{ABSTAIN_MESSAGE} ({veto.reason})",
                        signals=signals or {})

    answered = float(confidence) >= tau
    return Decision(
        answered=answered, confidence=float(confidence), policy=policy, tau=tau,
        reason=note if answered else f"confidence {confidence:.3f} < τ {tau:.3f}",
        veto_reason=veto.reason,
        message="" if answered else ABSTAIN_MESSAGE,
        signals=signals or {})
