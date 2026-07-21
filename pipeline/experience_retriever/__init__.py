"""Experience retriever stage.

Required entry points (all retrievers inherit from
``BaseExperienceRetriever`` which provides no-op defaults for the
optional surface — ``observe_escalation``, ``router_signals``,
``set_features_for``, ``prefetch``, ``add`` — so the runner and the
joint-feature routers can call them unconditionally).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from pipeline.common.types import DataPoint, Experience, RunState

from ._base import BaseExperienceRetriever


class ExperienceRetriever(Protocol):
    def retrieve(self, state: RunState, point: DataPoint, k: int) -> list[Experience]:
        ...

    def add(self, exp: Experience) -> None: ...

    def observe_escalation(
        self,
        exp_hits: list[Experience],
        small_pred: str | None,
        final_pred: str | None,
        *,
        query_name: str | None = None,
        small_confidence: float | None = None,
        point: DataPoint | None = None,
        **kwargs: Any,
    ) -> None: ...

    def router_signals(
        self, point: DataPoint, exp_hits: list[Experience],
    ) -> dict[str, Any] | None: ...

    def set_features_for(
        self, point: DataPoint, exp_hits: list[Experience],
    ) -> dict[str, np.ndarray]: ...

    def prefetch(self, data: list[DataPoint]) -> None: ...


from .topk_semantic import TopKSemanticExperienceRetriever  # noqa: E402
from .topk_semantic_online import (  # noqa: E402
    TopKSemanticOnlineExperienceRetriever,
)
from .care import CAREExperienceRetriever  # noqa: E402
from .care_pq import CAREPerQueryExperienceRetriever  # noqa: E402


REGISTRY: dict[str, type[ExperienceRetriever]] = {
    "topk_semantic": TopKSemanticExperienceRetriever,
    "topk_semantic_online": TopKSemanticOnlineExperienceRetriever,
    "care": CAREExperienceRetriever,
    "care_pq": CAREPerQueryExperienceRetriever,
}


def build(name: str, **kwargs) -> ExperienceRetriever:
    if name not in REGISTRY:
        raise ValueError(f"unknown experience retriever: {name!r} "
                         f"(known: {sorted(REGISTRY)})")
    return REGISTRY[name](**kwargs)


__all__ = [
    "BaseExperienceRetriever",
    "ExperienceRetriever",
    "REGISTRY",
    "build",
    "TopKSemanticExperienceRetriever",
    "TopKSemanticOnlineExperienceRetriever",
    "CAREExperienceRetriever",
    "CAREPerQueryExperienceRetriever",
]
