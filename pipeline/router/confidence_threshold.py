"""Confidence-threshold router.

Initial route is always the small model. After the small model answers, if the
extracted confidence (max of p_yes, p_no from first decision-token logprobs) is
below `threshold`, the router requests escalation to the large model.

Entry points (Router contract):
    route(state, point) -> "small"
    should_escalate(state, point, small_pred, small_raw, small_confidence) -> bool
"""

from __future__ import annotations

from typing import Literal

from pipeline.common.types import DataPoint, RunState

from ._base import BaseRouter, EscalationDecision


class ConfidenceThresholdRouter(BaseRouter):
    def __init__(self, threshold: float = 0.8):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def route(
        self, state: RunState, point: DataPoint
    ) -> Literal["small", "large"]:
        return "small"

    def should_escalate(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str,
        small_raw: str,
        small_confidence: float,
        **_kwargs,  # accept (and ignore) richer signals like small_features
    ) -> EscalationDecision:
        # Forced escalation cases: the small model has signaled it does
        # not have a definitive answer (parse failure, or 3-way "Unsure"
        # was the argmax over the True/False/Unsure logprobs).
        if small_pred == "UNKNOWN" or small_pred == "Unsure":
            return EscalationDecision(escalate=True)
        return EscalationDecision(escalate=small_confidence < self.threshold)
