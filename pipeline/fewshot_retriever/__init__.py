"""Few-shot retriever stage.

Canonical entry point:

    retrieve(state, point, k) -> list[FewShotDemo]

Draws only from `state.processed` (already-processed points in the streaming
run), so no future leakage.
"""

from __future__ import annotations

from typing import Protocol

from pipeline.common.types import DataPoint, FewShotDemo, RunState


class FewShotRetriever(Protocol):
    def retrieve(self, state: RunState, point: DataPoint, k: int) -> list[FewShotDemo]:
        ...


from typing import Callable  # noqa: E402

from .topk_semantic import TopKSemanticFewShotRetriever  # noqa: E402


def _build_large_only(**kwargs) -> FewShotRetriever:
    kwargs.setdefault("produced_by_filter", ("large",))
    return TopKSemanticFewShotRetriever(**kwargs)


REGISTRY: dict[str, Callable[..., FewShotRetriever]] = {
    "topk_semantic": TopKSemanticFewShotRetriever,
    # Drop-in variant: top-k cosine over processed points, but only those
    # whose final label was produced by the large LLM. Returns [] when the
    # large-labeled pool is empty (e.g. under the all_small router).
    "topk_semantic_large_only": _build_large_only,
}


def build(name: str, **kwargs) -> FewShotRetriever:
    if name not in REGISTRY:
        raise ValueError(f"unknown fewshot retriever: {name!r} "
                         f"(known: {sorted(REGISTRY)})")
    return REGISTRY[name](**kwargs)
