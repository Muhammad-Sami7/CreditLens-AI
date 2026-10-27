from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskAssessment:
    risk_level: str
    decision: str
    probability: float
    threshold: float


class RiskEngine:
    """
    Converts calibrated model probability into a
    business-friendly credit-risk assessment.

    The production decision threshold is the calibrated
    threshold selected during model development.
    """

    def __init__(self, threshold: float) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Risk threshold must be between 0 and 1."
            )

        self.threshold = float(threshold)

    def assess(
        self,
        probability: float,
    ) -> RiskAssessment:

        # Reject impossible values early so a bad upstream result can
        # never silently turn into a credit decision.
        probability = float(probability)

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        # The decision rule. `probability` must be the CALIBRATED
        # probability: the production threshold was selected on
        # calibrated scores, so raw model or ensemble scores must never
        # be passed in here. The comparison is inclusive (>=): a
        # probability exactly equal to the threshold needs review.
        if probability >= self.threshold:

            risk_level = "High Risk"
            decision = "Review Required"

        else:

            risk_level = "Low Risk"
            decision = "Below Risk Threshold"

        # Immutable result: nothing downstream (explanations, UI,
        # persistence) can alter the decision after this point.
        return RiskAssessment(
            risk_level=risk_level,
            decision=decision,
            probability=probability,
            threshold=self.threshold,
        )