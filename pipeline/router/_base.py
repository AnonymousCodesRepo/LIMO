"""Base class for routers — supplies no-op defaults for the optional
surface (``should_escalate``, ``on_escalation_observed``, ``prefetch``)
so the runner can call them unconditionally.

Subclasses override only the methods they care about. The required
``route`` method is left abstract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pipeline.common.types import DataPoint, RunState


@dataclass
class EscalationDecision:
    """Per-point routing decision + diagnostics returned by
    ``should_escalate``.

    ``escalate`` is the binary action. ``p_z1`` (when populated) is the
    router's calibrated estimate of P(z=1 | x), where z=1 means the small
    model agreed with the large model, so higher p_z1 means escalation is
    less likely needed. ``bootstrap`` is True while the router is still in
    its pre-trained-head phase. Routers without a calibrated head
    (``confidence_threshold``, ``all_small``, ``all_big``) leave both as
    ``None``.

    ``verify_small_*`` carry the cost of any EXTRA small-model calls a router
    issued while deciding (e.g. AutoMix's k self-verification generations);
    they default to 0 and the runner folds them into the cost accounting.
    """

    escalate: bool
    p_z1: float | None = None
    bootstrap: bool | None = None
    verify_small_calls: int = 0
    verify_small_prompt_tokens: int = 0
    verify_small_completion_tokens: int = 0
    # AutoMix self-verification consistency score in [0, 1] (fraction of
    # verifications that said "Correct"); higher = keep the small model.
    # None for routers that do not compute a verification score.
    verify_score: float | None = None


class BaseRouter:
    """Routers inherit from this so the runner never needs to ``hasattr``-
    check optional methods."""

    def route(
        self, state: RunState, point: DataPoint
    ) -> Literal["small", "large"]:
        raise NotImplementedError("subclasses must implement route()")

    def should_escalate(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str,
        small_raw: str,
        small_confidence: float,
        **_kwargs: Any,
    ) -> EscalationDecision:
        """Return an ``EscalationDecision`` for the small-call result.

        Default never escalates — appropriate for routers that always pin
        the route at construction time (``all_small`` / ``all_big``).
        """
        return EscalationDecision(escalate=False)

    def on_escalation_observed(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str,
        large_pred: str,
        **_kwargs: Any,
    ) -> None:
        """Hook fired after the runner has both small + large verdicts.

        Online learners override this to update their decision model.
        Default no-op.
        """
        return None

    def prefetch(self, data: list[DataPoint]) -> None:
        """Bulk pre-compute work the router would otherwise do per-point.

        Routers that need bulk precompute (e.g. query / doc embeddings)
        override this. Default no-op.
        """
        return None
