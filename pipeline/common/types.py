"""Shared lightweight types for the pipeline stages.

Each stage exposes a single entry point whose I/O is expressed with these
types. Keeping them tiny and explicit avoids accidental coupling between
stages (e.g. a retriever leaking its scoring internals into the runner).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DataPoint:
    """One (doc, query) evaluation unit."""

    doc_id: int
    doc_name: str
    doc_text: str
    query_name: str
    query_description: str
    ground_truth: str


@dataclass
class Experience:
    """Pre-generated experience record (subset of fields the retriever needs)."""

    experience_id: str
    source_query: str
    source_doc_id: int
    source_doc_excerpt: str
    experience_text: str
    applicability_signal: str = ""
    score: float = 0.0


@dataclass
class FewShotDemo:
    """One already-processed point usable as a few-shot demonstration."""

    doc_id: int
    query_name: str
    doc_text: str
    prediction: str            # "Yes" | "No"  (label comes from whichever LLM ran)
    produced_by: Literal["small", "large"]
    score: float = 0.0


@dataclass
class ProcessedRecord:
    """Result of running one data point through the pipeline."""

    doc_id: int
    query_name: str
    ground_truth: str
    prediction: str
    raw: str
    routed_to: Literal["small", "large"]
    retrieved_experiences: list[dict[str, Any]] = field(default_factory=list)
    retrieved_fewshot: list[dict[str, Any]] = field(default_factory=list)
    latency: float = 0.0
    # Populated when the small model was called. None if it was not (e.g.
    # router sent directly to large). Values:
    #   * "Yes" | "No"  — definitive verdict (argmax of p_yes/p_no).
    #   * "Unsure"      — 3-way argmax was "Unsure"; routers force escalation.
    #   * "UNKNOWN"     — parse failure / unparseable response.
    small_prediction: str | None = None
    small_confidence: float | None = None
    # Raw class probabilities from the small model's 3-way head. None when the
    # small model was not called or logprobs were unavailable.
    small_p_yes: float | None = None
    small_p_no: float | None = None
    small_p_unsure: float | None = None
    # True when the router's should_escalate fired after a small call, so the
    # final prediction came from the large model.
    escalated: bool = False
    # Router decision diagnostics. Populated only when the router supplies
    # them (lightgbm has a calibrated head).
    #   router_p_z1       — calibrated P(z=1 | x); z=1 means the small model
    #                       agreed with the large model (higher = less
    #                       escalation pressure).
    #   router_bootstrap  — True iff the router was still in its pre-trained-
    #                       head bootstrap phase for this point.
    router_p_z1: float | None = None
    router_bootstrap: bool | None = None
    # Per-call wall times (seconds). None means the call was not made.
    #   t_small         — initial small-LLM prediction call
    #   t_route_decision — total time spent inside router.route() +
    #                      router.should_escalate() (~CPU; tiny for non-LR)
    #   t_large         — large-LLM call (escalation OR direct large route)
    t_small: float | None = None
    t_route_decision: float | None = None
    t_large: float | None = None
    # Per-stage wall times (seconds) for the non-LLM pipeline work. None
    # means the stage did not run for this point.
    #   t_retrieve_exp   — experience_retriever.retrieve() (embedding lookup
    #                      + shortlist scoring + rerank)
    #   t_retrieve_fs    — fewshot_retriever.retrieve()
    #   t_router_signals — experience_retriever.router_signals() (re-scores
    #                      the retrieved set for the router)
    #   t_observe        — router.on_escalation_observed() +
    #                      experience_retriever.observe_escalation() after a
    #                      completed escalation; includes any synchronous
    #                      router refit triggered by this observation
    t_retrieve_exp: float | None = None
    t_retrieve_fs: float | None = None
    t_router_signals: float | None = None
    t_observe: float | None = None
    # Token usage per call, metered live from the serving stack (vLLM usage).
    # The small model is always called live, so its counts reflect the exact
    # prompt seen. The large model's counts are 0 when the escalation was
    # served from the prediction cache (no live call); the offline token table
    # supplies the cached-escalation large cost at aggregation time.
    small_prompt_tokens: int = 0
    small_completion_tokens: int = 0
    large_prompt_tokens: int = 0
    large_completion_tokens: int = 0
    # Extra small-model calls a router made while deciding (AutoMix issues k
    # self-verification generations per point). Kept separate from the
    # answer-call tokens so the two costs stay distinguishable. 0 for other
    # routers.
    verify_small_calls: int = 0
    verify_small_prompt_tokens: int = 0
    verify_small_completion_tokens: int = 0
    # AutoMix self-verification consistency score in [0, 1] (None for other
    # routers).
    verify_score: float | None = None


@dataclass
class RunState:
    """State that grows as the runner streams through data points.

    Stages receive this read-only (by convention) — only the runner appends to
    `processed` after a point finishes.
    """

    processed: list[ProcessedRecord] = field(default_factory=list)
    # Fast lookup: doc_id -> doc_text for demos (avoids re-plumbing text).
    _doc_text_by_id: dict[int, str] = field(default_factory=dict)

    def record(self, point: DataPoint, result: ProcessedRecord) -> None:
        self.processed.append(result)
        self._doc_text_by_id[point.doc_id] = point.doc_text

    def doc_text(self, doc_id: int) -> str:
        return self._doc_text_by_id.get(doc_id, "")
