"""Base class for experience retrievers — supplies no-op defaults for
the optional surface (``observe_escalation``, ``router_signals``,
``set_features_for``, ``prefetch``, ``add``) so the runner and the
joint-feature routers can call them unconditionally.

Subclasses override only the methods they care about. The required
``retrieve`` method is left abstract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.common.types import DataPoint, Experience, RunState


class BaseExperienceRetriever:
    """Common base — gives every retriever the full method surface."""

    def retrieve(
        self, state: RunState, point: DataPoint, k: int,
    ) -> list[Experience]:
        raise NotImplementedError("subclasses must implement retrieve()")

    def add(self, exp: Experience) -> None:
        """Install a newly generated experience.

        Static-pool retrievers override this to no-op or raise; growable
        retrievers extend their pool.
        """
        raise NotImplementedError(
            "this retriever is static; pass an online experience generator "
            "only with a growable retriever (topk_semantic / "
            "topk_semantic_online / care / care_pq)."
        )

    def observe_escalation(
        self,
        exp_hits: list[Experience],
        small_pred: str | None,
        final_pred: str | None,
        *,
        query_name: str | None = None,
        small_confidence: float | None = None,
        point: DataPoint | None = None,
        **_kwargs: Any,
    ) -> None:
        """Hook fired after the runner observes a small / large verdict
        pair. Online learners (helpfulness counters, BLR / LinUCB heads)
        override this. Default no-op.
        """
        return None

    def router_signals(
        self, point: DataPoint, exp_hits: list[Experience],
    ) -> dict[str, Any] | None:
        """Per-point signals the router can consume (top-k score stats,
        helpfulness aggregates, pool state). Default returns ``None`` so
        the router treats it as missing.
        """
        return None

    def set_features_for(
        self, point: DataPoint, exp_hits: list[Experience],
    ) -> dict[str, np.ndarray]:
        """Per-set CARE-shaped features (φ, p_calib, blr_var) for the
        joint-feature router (lightgbm). Retrievers without a feature
        head return empty arrays."""
        empty = np.empty((0,), dtype=np.float64)
        return {"phi": empty, "p_calib": empty, "blr_var": empty}

    def prefetch(self, data: list[DataPoint]) -> None:
        """Bulk pre-compute embeddings the retriever would otherwise do
        per-point. Default no-op."""
        return None
