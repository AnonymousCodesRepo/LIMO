"""Router stage.

Required entry points (all routers inherit from ``BaseRouter`` which
provides no-op defaults so the runner can call them unconditionally):

    route(state, point) -> Literal["small", "large"]
    should_escalate(state, point, small_pred, small_raw, small_confidence,
                    **signals) -> bool
    on_escalation_observed(state, point, small_pred, large_pred,
                           **signals) -> None
    prefetch(data) -> None
"""

from __future__ import annotations

from typing import Literal, Protocol

from pipeline.common.types import DataPoint, RunState

from ._base import BaseRouter


class Router(Protocol):
    def route(self, state: RunState, point: DataPoint) -> Literal["small", "large"]:
        ...

    def should_escalate(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str,
        small_raw: str,
        small_confidence: float,
        **signals,
    ) -> bool:
        ...

    def on_escalation_observed(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str,
        large_pred: str,
        **signals,
    ) -> None:
        ...

    def prefetch(self, data: list[DataPoint]) -> None:
        ...


from .all_big import AllBigRouter  # noqa: E402
from .all_small import AllSmallRouter  # noqa: E402
from .automix import AutoMixRouter  # noqa: E402
from .confidence_threshold import ConfidenceThresholdRouter  # noqa: E402
from .lightgbm_router import LightGBMRouter  # noqa: E402


REGISTRY: dict[str, type[Router]] = {
    "all_small": AllSmallRouter,
    "all_big": AllBigRouter,
    "automix": AutoMixRouter,
    "confidence_threshold": ConfidenceThresholdRouter,
    "lightgbm": LightGBMRouter,
}


def build(name: str, **kwargs) -> Router:
    if name not in REGISTRY:
        raise ValueError(f"unknown router: {name!r} (known: {sorted(REGISTRY)})")
    return REGISTRY[name](**kwargs)


__all__ = [
    "BaseRouter",
    "Router",
    "REGISTRY",
    "build",
    "AllBigRouter",
    "AllSmallRouter",
    "AutoMixRouter",
    "ConfidenceThresholdRouter",
    "LightGBMRouter",
]
