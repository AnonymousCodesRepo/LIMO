"""Experience generator stage.

A generator is called AFTER a data point finishes. If its `should_generate`
predicate is True for that (point, record) pair, `generate(...)` produces a
new `Experience` that is then added to the experience retriever's pool so
LATER points in the stream can benefit from it.

Canonical entry points:

    should_generate(point, record) -> bool
    generate(point, record) -> Experience | None

Generators must be thread-safe: the runner fires them concurrently within a
chunk boundary.
"""

from __future__ import annotations

from typing import Protocol

from pipeline.common.types import DataPoint, Experience, ProcessedRecord


class ExperienceGenerator(Protocol):
    def should_generate(
        self, point: DataPoint, record: ProcessedRecord
    ) -> bool: ...

    def generate(
        self, point: DataPoint, record: ProcessedRecord
    ) -> Experience | None: ...


from .online_discrepancy import OnlineDiscrepancyGenerator  # noqa: E402


REGISTRY: dict[str, type[ExperienceGenerator]] = {
    "online_discrepancy": OnlineDiscrepancyGenerator,
}


def build(name: str, **kwargs) -> ExperienceGenerator:
    if name not in REGISTRY:
        raise ValueError(f"unknown experience generator: {name!r} "
                         f"(known: {sorted(REGISTRY)})")
    return REGISTRY[name](**kwargs)
