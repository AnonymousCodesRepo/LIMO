"""Shared types for label synthesis stage.

A Rollout is a single labeled (predicate, doc) record — the unit consumed
by the offline trainers. Features are pre-computed in the shape the
existing router heads expect (34-d for the LightGBM router) so the
trainer doesn't need to rebuild them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class Rollout:
    """One labeled rollout consumed by the offline trainers."""

    cand_id: str
    predicate: str
    doc_id: int
    expected_answer: str
    # 34-d float feature vector used by the LightGBM router head.
    features: list[float]
    # Small-LLM cheap-pass artifacts (always populated; the labeler runs
    # the small pass on every candidate before deciding whether to pay
    # for a large call).
    small_prediction: str | None
    small_confidence: float | None
    s_entropy: float                    # H(p_yes_2way, p_no_2way)
    # Large-LLM artifacts (populated only on selected candidates).
    large_prediction: str | None
    # Silver label: 1 iff small_prediction == large_prediction. None when L
    # was not called for this candidate.
    z: int | None
    # Probability that this candidate was selected for large-LLM evaluation in
    # the round that labeled it (recorded for provenance/diagnostics).
    q: float
    round_idx: int                       # which acquisition round labeled this
    acquisition: str                     # "uniform" | "s_entropy" | "router_bald"
    # Free-form metadata (e.g. retrieved-experience IDs, generation provenance).
    meta: dict[str, Any] = field(default_factory=dict)


def dump_rollouts(path: str | Path, rollouts: Iterable[Rollout],
                  append: bool = True) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    n = 0
    with open(p, mode) as f:
        for r in rollouts:
            d = asdict(r)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_rollouts(path: str | Path) -> list[Rollout]:
    p = Path(path)
    out: list[Rollout] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(Rollout(**r))
    return out


def features_matrix(rollouts: list[Rollout]) -> np.ndarray:
    """Stack feature vectors into a (N, D) float64 matrix."""
    if not rollouts:
        return np.zeros((0, 0), dtype=np.float64)
    arr = np.asarray([r.features for r in rollouts], dtype=np.float64)
    return arr
