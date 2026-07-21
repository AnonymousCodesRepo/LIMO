"""Top-k experience retriever using the all-mpnet-base-v2 embedding server.

Each experience is embedded as: "{source_query_description or ''} | {experience_text}".
Queries are embedded as:        "{query_description} | {document_text[:1500]}".
Ranking is cosine similarity (embeddings are L2-normalized by the client, so a
simple dot product suffices).

Experiences generated from the point's own document are NOT excluded; the
feature-based rerankers expose a `same_doc` indicator feature instead, so the
utility estimator learns whether such experiences help.
"""

from __future__ import annotations

import threading

import numpy as np

from pathlib import Path

from pipeline.common.embeddings import EmbeddingClient
from pipeline.common.types import DataPoint, Experience, RunState

from ._base import BaseExperienceRetriever


def load_point_cache(path: str | Path) -> dict[tuple[str, int], np.ndarray]:
    """Load a point-embedding cache (query_names, doc_ids, vectors)."""
    data = np.load(Path(path), allow_pickle=False)
    qns = data["query_names"]
    dids = data["doc_ids"]
    vecs = data["vectors"]
    out: dict[tuple[str, int], np.ndarray] = {}
    for i in range(len(qns)):
        out[(str(qns[i]), int(dids[i]))] = vecs[i]
    return out


def _exp_text(e: Experience) -> str:
    return (e.experience_text or "").strip()


class TopKSemanticExperienceRetriever(BaseExperienceRetriever):
    def __init__(
        self,
        experiences: list[Experience],
        embed_client: EmbeddingClient | None = None,
        doc_prefix_chars: int = 1500,
        point_cache: dict[tuple[str, int], np.ndarray] | None = None,
        restrict_to_source_query: bool = False,
    ):
        # `list(...)` so callers can still hold their own reference without us
        # mutating it when `add()` is used.
        self.experiences = list(experiences)
        self.client = embed_client or EmbeddingClient()
        self.doc_prefix_chars = doc_prefix_chars
        # Optional precomputed point-side embeddings: {(query_name, doc_id): vec}.
        # When present, retrieve() skips the embedding round-trip.
        self._point_cache = point_cache
        # When True, retrieve() only returns experiences whose source_query
        # matches the current point's query_name (no cross-query sharing).
        # Default False shares experiences across queries.
        self.restrict_to_source_query = bool(restrict_to_source_query)
        self._cache_hits = 0
        self._cache_misses = 0
        self._exp_vecs: np.ndarray | None = None
        # experience_id -> row index into `_exp_vecs` / `experiences`, so
        # per-candidate vector lookups are O(1) instead of a linear scan.
        self._id2row: dict[str, int] = {}
        # Protects `self.experiences` and `self._exp_vecs` against concurrent
        # reads (from retrieve()) and writes (from add()) in the chunked-parallel
        # runner. Embedding the new experience happens OUTSIDE the lock.
        self._lock = threading.Lock()
        self._fit()

    def _fit(self) -> None:
        self._id2row = {
            e.experience_id: i for i, e in enumerate(self.experiences)
        }
        if not self.experiences:
            # Empty pool: shape (0, 0) is a harmless placeholder (callers check
            # `not self.experiences` first); add() resizes it to the embed dim.
            self._exp_vecs = np.zeros((0, 0), dtype=np.float32)
            return
        texts = [_exp_text(e) for e in self.experiences]
        self._exp_vecs = self.client.embed(texts)

    def _vec_for_exp_id(self, exp_id: str) -> np.ndarray | None:
        """Look up the embedding vector for a given experience_id. O(1)."""
        with self._lock:
            if self._exp_vecs is None or self._exp_vecs.size == 0:
                return None
            i = self._id2row.get(exp_id)
            if i is None:
                return None
            return np.asarray(self._exp_vecs[i])

    def size(self) -> int:
        with self._lock:
            return len(self.experiences)

    def add(self, exp: Experience) -> None:
        """Append a new experience to the pool and update the embedding matrix.

        Thread-safe: embedding is computed outside the lock (avoiding a long
        critical section), then the matrix append happens under the lock.
        """
        text = _exp_text(exp)
        # Guard against empty strings slipping in (they embed to a zero-ish
        # vector and would just add noise).
        if not text:
            return
        vec = self.client.embed_one(text).astype(np.float32).reshape(1, -1)
        with self._lock:
            self.experiences.append(exp)
            self._id2row[exp.experience_id] = len(self.experiences) - 1
            if self._exp_vecs is None or self._exp_vecs.size == 0:
                self._exp_vecs = vec
            else:
                self._exp_vecs = np.vstack([self._exp_vecs, vec])

    def prefetch(self, data: list[DataPoint]) -> None:
        """Batch-embed every point's recall text into the point cache.

        Called once by the driver before the stream starts. Removes the
        per-point ``embed_one`` round trip from ``retrieve()`` — the recall
        query vector is a pure function of the (query_description, doc_text)
        pair, so precomputing it for the whole stream leaks nothing.
        Idempotent; keys already present (e.g. from an on-disk
        ``point_embedding_cache``) are kept as-is."""
        if self._point_cache is None:
            self._point_cache = {}
        missing: list[tuple[str, int]] = []
        texts: list[str] = []
        seen: set[tuple[str, int]] = set()
        for p in data:
            key = (p.query_name, p.doc_id)
            if key in self._point_cache or key in seen:
                continue
            seen.add(key)
            missing.append(key)
            texts.append(
                f"{p.query_description} | {p.doc_text[:self.doc_prefix_chars]}"
            )
        if not texts:
            return
        vecs = self.client.embed(texts)
        for key, v in zip(missing, vecs):
            self._point_cache[key] = np.asarray(v, dtype=np.float32)

    def retrieve(
        self, state: RunState, point: DataPoint, k: int
    ) -> list[Experience]:
        # Take a snapshot under the lock so a concurrent add() can't splice a
        # new row into `_exp_vecs` mid-scoring. We copy the array reference
        # (not data) and the list shallowly.
        with self._lock:
            if not self.experiences or k <= 0:
                return []
            experiences = list(self.experiences)
            exp_vecs = self._exp_vecs
        q_vec = None
        if self._point_cache is not None:
            q_vec = self._point_cache.get((point.query_name, point.doc_id))
        if q_vec is None:
            self._cache_misses += 1
            q_text = (
                f"{point.query_description} | {point.doc_text[:self.doc_prefix_chars]}"
            )
            q_vec = self.client.embed_one(q_text)
        else:
            self._cache_hits += 1
        scores = exp_vecs @ q_vec  # (N,)
        order = np.argsort(-scores)
        out: list[Experience] = []
        for idx in order:
            e = experiences[int(idx)]
            if (self.restrict_to_source_query
                    and e.source_query != point.query_name):
                continue
            hit = Experience(
                experience_id=e.experience_id,
                source_query=e.source_query,
                source_doc_id=e.source_doc_id,
                source_doc_excerpt=e.source_doc_excerpt,
                experience_text=e.experience_text,
                applicability_signal=e.applicability_signal,
                score=float(scores[int(idx)]),
            )
            out.append(hit)
            if len(out) >= k:
                break
        return out
