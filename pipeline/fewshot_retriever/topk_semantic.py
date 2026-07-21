"""Top-k few-shot demonstration retriever over already-processed points.

Scope rules:
  - Only points from `state.processed` whose query_name matches the current
    point's query_name are eligible.
  - Skips UNKNOWN / unparseable predictions.
  - Optional `produced_by_filter`: restrict candidates to those whose final
    label was produced by a specific LLM (e.g. ("large",) to prefer the
    more-accurate 70B labels). Returns [] if no candidates survive.
  - Score = cosine similarity between the current document and the candidate
    demo document; embeddings are cached in-memory by doc_id so each document
    is embedded at most once per run.
"""

from __future__ import annotations

import threading

import numpy as np

from pipeline.common.embeddings import EmbeddingClient
from pipeline.common.types import DataPoint, FewShotDemo, RunState


class TopKSemanticFewShotRetriever:
    def __init__(
        self,
        embed_client: EmbeddingClient | None = None,
        doc_prefix_chars: int = 1500,
        allowed_labels: tuple[str, ...] = ("Yes", "No"),
        produced_by_filter: tuple[str, ...] | None = None,
    ):
        self.client = embed_client or EmbeddingClient()
        self.doc_prefix_chars = doc_prefix_chars
        self.allowed_labels = set(allowed_labels)
        self.produced_by_filter = (
            set(produced_by_filter) if produced_by_filter else None
        )
        self._cache: dict[int, np.ndarray] = {}
        self._cache_lock = threading.Lock()

    def _vec_for_doc(self, doc_id: int, doc_text: str) -> np.ndarray:
        with self._cache_lock:
            v = self._cache.get(doc_id)
        if v is None:
            v = self.client.embed_one(doc_text[:self.doc_prefix_chars])
            with self._cache_lock:
                # Another thread may have populated concurrently; either value is fine.
                self._cache.setdefault(doc_id, v)
                v = self._cache[doc_id]
        return v

    def retrieve(
        self, state: RunState, point: DataPoint, k: int
    ) -> list[FewShotDemo]:
        if k <= 0 or not state.processed:
            return []
        cands = [
            r for r in state.processed
            if r.query_name == point.query_name
            and r.prediction in self.allowed_labels
            and r.doc_id != point.doc_id
            and (
                self.produced_by_filter is None
                or r.routed_to in self.produced_by_filter
            )
        ]
        if not cands:
            return []
        q_vec = self._vec_for_doc(point.doc_id, point.doc_text)
        scored: list[tuple[float, FewShotDemo]] = []
        for r in cands:
            d_text = state.doc_text(r.doc_id)
            if not d_text:
                continue
            v = self._vec_for_doc(r.doc_id, d_text)
            s = float(q_vec @ v)
            scored.append((s, FewShotDemo(
                doc_id=r.doc_id,
                query_name=r.query_name,
                doc_text=d_text,
                prediction=r.prediction,
                produced_by=r.routed_to,
                score=s,
            )))
        scored.sort(key=lambda x: -x[0])
        return [d for _, d in scored[:k]]
