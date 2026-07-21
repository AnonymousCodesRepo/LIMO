"""Recording layer for experience generation + retrieval.

Strictly additive: it does not change generator or retriever behaviour.

- ``RecordingDiscrepancyGenerator`` — subclass of
  ``OnlineDiscrepancyGenerator``. Inherits all behaviour and overrides only
  the LLM-touching helpers and ``generate`` to side-record into a
  per-trigger buffer. The accept/reject outcome is determined
  deterministically from what was recorded during the call, so concurrent
  generates don't race.

- ``RecordingExperienceRetriever`` — wrapper around any retriever satisfying
  the runner's ``_ExperienceRetrieverP`` protocol. Forwards every attribute
  access via ``__getattr__`` so the runner sees the inner retriever's full
  surface. Intercepts only ``retrieve`` (counts per-experience retrievals)
  and ``add`` (records install order). ``add`` is exposed only when the
  inner retriever implements it.

Both classes expose ``recordings()`` returning serialisable dicts.
"""

from __future__ import annotations

import threading
from typing import Any

from pipeline.common.types import DataPoint, Experience, ProcessedRecord, RunState
from pipeline.experience_generator.online_discrepancy import (
    OnlineDiscrepancyGenerator,
)


# ── Generator ──────────────────────────────────────────────────────────────


class RecordingDiscrepancyGenerator(OnlineDiscrepancyGenerator):
    """Drop-in replacement that captures small reasoning, raw meta output,
    and accept/reject reasons per generation trigger.

    The base class's locking and stats are inherited unchanged. We only
    add a separate recording lock + per-(doc_id, query_name) buffer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rec_lock = threading.Lock()
        # In-flight buffers keyed by (doc_id, query_name); finalised entries
        # are moved to ``_records`` once ``generate`` returns.
        self._rec_buf: dict[tuple[int, str], dict[str, Any]] = {}
        self._records: list[dict[str, Any]] = []
        # Leading chars of the source doc kept on the record (the full text
        # is too large to ship in the sidecar JSON).
        self.doc_excerpt_chars = 2000

    # ── helpers ────────────────────────────────────────────────────────

    def _rec_get(self, doc_id: int, query_name: str) -> dict[str, Any]:
        key = (doc_id, query_name)
        with self._rec_lock:
            d = self._rec_buf.get(key)
            if d is None:
                d = {"doc_id": doc_id, "query_name": query_name}
                self._rec_buf[key] = d
            return d

    # ── overridden LLM seams (record success only — base handles errors) ──

    def _call_small_explain(
        self, point: DataPoint, small_pred: str
    ) -> str:
        out = super()._call_small_explain(point, small_pred)
        d = self._rec_get(point.doc_id, point.query_name)
        with self._rec_lock:
            d["small_reasoning"] = out
            d["small_pred_at_trigger"] = small_pred
            d["query_description"] = point.query_description
            d["doc_name"] = point.doc_name
            d["doc_excerpt"] = (
                point.doc_text[: self.doc_excerpt_chars]
                if point.doc_text else ""
            )
        return out

    def _call_large_synthesize(
        self,
        point: DataPoint,
        small_reasoning: str,
        small_pred: str,
        large_pred: str,
    ) -> str:
        out = super()._call_large_synthesize(
            point, small_reasoning, small_pred, large_pred
        )
        d = self._rec_get(point.doc_id, point.query_name)
        with self._rec_lock:
            d["meta_raw"] = out
            d["large_pred_at_trigger"] = large_pred
        return out

    # ── main entry: stamp accept/reject after the base call ───────────

    def generate(
        self, point: DataPoint, record: ProcessedRecord
    ) -> Experience | None:
        key = (point.doc_id, point.query_name)
        exp = super().generate(point, record)

        with self._rec_lock:
            d = self._rec_buf.get(key)
            if d is None:
                # ``should_generate`` rejected this point before ``generate``; skip.
                return exp
            # Always-on metadata about the trigger.
            d.setdefault("doc_id", point.doc_id)
            d.setdefault("query_name", point.query_name)
            d.setdefault("query_description", point.query_description)
            d.setdefault("doc_name", point.doc_name)
            d.setdefault(
                "doc_excerpt",
                point.doc_text[: self.doc_excerpt_chars] if point.doc_text else "",
            )
            d["small_pred_at_trigger"] = (
                d.get("small_pred_at_trigger") or record.small_prediction
            )
            d["large_pred_at_trigger"] = (
                d.get("large_pred_at_trigger") or record.prediction
            )
            d["accept"] = exp is not None
            if exp is not None:
                d["experience_id"] = exp.experience_id
                d["parsed"] = {
                    "experience_text": exp.experience_text,
                }
                d["reject_reason"] = None
            else:
                d["experience_id"] = None
                d["parsed"] = None
                # Reason is fully determined by what we recorded.
                if "meta_raw" not in d:
                    if "small_reasoning" not in d:
                        d["reject_reason"] = "llm_error_explain"
                    else:
                        d["reject_reason"] = "llm_error_synthesize"
                else:
                    # meta_raw recorded but no exp → parse_fail.
                    d["reject_reason"] = "parse_fail"
            self._records.append(dict(d))
            self._rec_buf.pop(key, None)
        return exp

    # ── public surface ────────────────────────────────────────────────

    def recordings(self) -> list[dict[str, Any]]:
        """Snapshot of all finalised trigger records (accepted + rejected)."""
        with self._rec_lock:
            return [dict(r) for r in self._records]


# ── Retriever wrapper ──────────────────────────────────────────────────────


class RecordingExperienceRetriever:
    """Forwarding wrapper that records per-experience usage and install order.

    Whatever the inner retriever exposes is forwarded via ``__getattr__`` so
    the runner sees no behavioural difference. Two methods are intercepted:

    - ``retrieve(state, point, k)``: forwards, then increments per-experience
      retrieval counters keyed by ``experience_id``.
    - ``add(exp)``: present iff the inner retriever has ``add``. Records
      install order (and source query / doc) before delegating.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._rec_lock = threading.Lock()
        self._installs: dict[str, dict[str, Any]] = {}
        self._usage: dict[str, dict[str, Any]] = {}
        self._install_counter = 0

    # ── intercepted ───────────────────────────────────────────────────

    def retrieve(
        self, state: RunState, point: DataPoint, k: int
    ) -> list[Experience]:
        out = self._inner.retrieve(state, point, k)
        if not out:
            return out
        with self._rec_lock:
            for rank, e in enumerate(out):
                u = self._usage.get(e.experience_id)
                if u is None:
                    u = {
                        "times_retrieved": 0,
                        "rank_sum": 0,
                        "retrieved_for_queries": {},
                    }
                    self._usage[e.experience_id] = u
                u["times_retrieved"] += 1
                u["rank_sum"] += rank
                rfq = u["retrieved_for_queries"]
                rfq[point.query_name] = rfq.get(point.query_name, 0) + 1
        return out

    # ``add`` is exposed via property so ``getattr(retriever, "add", None)``
    # in the runner returns ``None`` exactly when the inner retriever has no
    # ``add`` (the property raises AttributeError in that case).
    @property
    def add(self):
        inner_add = getattr(self._inner, "add", None)
        if inner_add is None:
            raise AttributeError("inner retriever does not implement add()")

        rec_lock = self._rec_lock
        installs = self._installs
        cnt_holder = self  # mutable order counter via attribute access

        def _wrapped_add(exp: Experience) -> None:
            with rec_lock:
                cnt_holder._install_counter += 1
                installs[exp.experience_id] = {
                    "install_order": cnt_holder._install_counter,
                    "source_query": exp.source_query,
                    "source_doc_id": exp.source_doc_id,
                }
            inner_add(exp)

        return _wrapped_add

    # ── forward everything else ──────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        # Only called when ``name`` isn't found on self; falls through to inner.
        return getattr(self._inner, name)

    # ── public surface ────────────────────────────────────────────────

    def recordings(self) -> dict[str, Any]:
        """Snapshot of install metadata + usage counters keyed by experience_id."""
        with self._rec_lock:
            installs = {k: dict(v) for k, v in self._installs.items()}
            usage = {
                k: {
                    "times_retrieved": v["times_retrieved"],
                    "rank_sum": v["rank_sum"],
                    "avg_rank": (
                        v["rank_sum"] / v["times_retrieved"]
                        if v["times_retrieved"] else None
                    ),
                    "retrieved_for_queries": dict(v["retrieved_for_queries"]),
                }
                for k, v in self._usage.items()
            }
        return {"installs": installs, "usage": usage}
