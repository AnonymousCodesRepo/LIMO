"""Trivial router: every data point goes to the large model."""

from __future__ import annotations

from typing import Literal

from pipeline.common.types import DataPoint, RunState

from ._base import BaseRouter


class AllBigRouter(BaseRouter):
    def route(self, state: RunState, point: DataPoint) -> Literal["small", "large"]:
        return "large"
