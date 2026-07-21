"""Online experience generator (discrepancy-driven).

Flow:

  1. Trigger: the just-completed data point has
       escalated=True  AND  small_prediction != final_prediction
     i.e. the small model and the (large-model) final answer disagreed.
     Ground truth is no longer consulted — the large model's answer is
     treated as the reference. Note: under the 3-way small prompt
     ``small_prediction`` may be "Unsure"; that is treated as a valid
     discrepancy ("Unsure" ≠ "Yes"/"No"), so the generator fires and the
     resulting experience teaches the small model to commit to the right
     verdict on similar future inputs.
  2. Small-model explain: ask the 0.8B to rationalize its own prior
     (disagreed-with) answer. We don't re-query; we only need its reasoning.
  3. Large-model synthesize: give the 70B the (doc, query, small reasoning,
     small prediction, large prediction) and ask for a structured experience
     JSON ({experience_text}). The 70B internally diagnoses the misstep
     first, but only the actionable lesson is returned.
  4. Produce an `Experience` and return it for the runner to install into
     the retriever's pool.

Per-query cap: once a query has reached `max_per_query` generated experiences
for the current run, further triggers are dropped, bounding generation cost.

Thread-safety:
  - `_count_by_query` and `_generated_ids` are guarded by a single lock.
  - `should_generate` takes the lock only to read/reserve the slot so two
    concurrent triggers for the same query can't both slip under the cap.
    The actual LLM calls happen in `generate(...)` OUTSIDE the lock.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from openai import OpenAI

from pipeline.common.types import DataPoint, Experience, ProcessedRecord

from .prompts import (
    meta_prompt_messages,
    parse_llm_experience_json,
    small_model_explain_messages,
)


class OnlineDiscrepancyGenerator:
    def __init__(
        self,
        *,
        small_client: OpenAI,
        small_model: str,
        large_client: OpenAI,
        large_model: str,
        max_per_query: int = 5,
        explain_max_tokens: int = 200,
        meta_max_tokens: int = 300,
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        domain: str = "legal",
        position_gate_rho: float | None = None,
        large_extra_body: dict | None = None,
    ):
        self.small_client = small_client
        self.small_model = small_model
        self.large_client = large_client
        self.large_model = large_model
        # Provider-specific request body forwarded to the live large-model
        # synthesis call (e.g. Qwen3.5's enable_thinking=False). Empty for Llama.
        self.large_extra_body = large_extra_body or {}
        # Classification domain name injected into the explain / synthesis
        # prompts so experiences are framed to the actual task (privacy-policy,
        # cancer-biology, fact-verification) rather than hardcoded "legal".
        self.domain = domain
        self.max_per_query = max(0, int(max_per_query))
        self.explain_max_tokens = explain_max_tokens
        self.meta_max_tokens = meta_max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout

        # Cost-aware position gate (the t* rule). When set, a trigger at
        # per-query position t (1-based, in stream order) qualifies only if
        #     t <= t* = n_q / (1 + position_gate_rho)
        # where rho = c_gen / (beta * c_L) is the measured cost ratio between
        # generating one experience and the (efficacy-discounted) saving of
        # one large-model call. Derivation: a failure first seen at position
        # t recurs ~ n_q/t - 1 more times, so generation pays off only while
        # beta * c_L * (n_q/t - 1) >= c_gen. Requires ``prefetch(data)`` so
        # n_q is known; queries with unknown n_q are not gated. The
        # max_per_query cap stays as an independent safeguard.
        if position_gate_rho is not None and position_gate_rho < 0:
            raise ValueError(
                f"position_gate_rho must be >= 0 or None, got {position_gate_rho}"
            )
        self.position_gate_rho = (
            float(position_gate_rho) if position_gate_rho is not None else None
        )
        # Per-query totals (from prefetch) and running per-query positions.
        self._n_by_query: dict[str, int] = {}
        self._pos_by_query: dict[str, int] = {}
        self._pos_seen: set[tuple[int, str]] = set()

        self._lock = threading.Lock()
        self._count_by_query: dict[str, int] = {}
        # Track the (doc_id, query_name) we've already claimed a slot for,
        # so a retry can't double-count if someone calls should_generate twice.
        self._claimed: set[tuple[int, str]] = set()
        # Simple generation stats for the run-level JSON output.
        self._stats: dict[str, Any] = {
            "attempted": 0,
            "succeeded": 0,
            "parse_failures": 0,
            "llm_errors": 0,
            "skipped_cap": 0,
            "skipped_position_gate": 0,
            "skipped_stale_stream": 0,
            # Cumulative wall time + call counts for each LLM phase.
            # Averages = total / count; counts include failing calls.
            "t_small_explain_total": 0.0,
            "n_small_explain_calls": 0,
            "t_large_synthesize_total": 0.0,
            "n_large_synthesize_calls": 0,
            # Prompt/completion tokens metered live for each generation LLM
            # phase (small explain + large synthesize), so the cost accounting
            # can charge experience generation exactly.
            "gen_small_prompt_tokens": 0,
            "gen_small_completion_tokens": 0,
            "gen_large_prompt_tokens": 0,
            "gen_large_completion_tokens": 0,
        }

    # ── Prefetch (position-gate totals) ──────────────────────────────────

    def prefetch(self, data: list[DataPoint]) -> None:
        """Record per-query point totals ``n_q`` for the position gate.

        Called once by the driver before the stream starts (same pattern as
        the router/retriever prefetch hooks). Idempotent."""
        counts: dict[str, int] = {}
        for p in data:
            counts[p.query_name] = counts.get(p.query_name, 0) + 1
        with self._lock:
            self._n_by_query.update(counts)

    # ── Trigger check ────────────────────────────────────────────────────

    def should_generate(
        self, point: DataPoint, record: ProcessedRecord
    ) -> bool:
        # Per-query position bookkeeping happens for EVERY distinct point,
        # before any early return, so t is the point's stream position
        # within its query regardless of what this trigger decides.
        with self._lock:
            pkey = (point.doc_id, point.query_name)
            if pkey not in self._pos_seen:
                self._pos_seen.add(pkey)
                self._pos_by_query[point.query_name] = (
                    self._pos_by_query.get(point.query_name, 0) + 1
                )
            t = self._pos_by_query[point.query_name]

        if self.max_per_query <= 0:
            return False
        if not record.escalated:
            return False
        small_pred = record.small_prediction
        if small_pred is None or small_pred == "UNKNOWN":
            return False
        large_pred = record.prediction
        if large_pred == "UNKNOWN":
            # No reliable large-model answer to treat as the reference.
            return False
        if small_pred == large_pred:
            # Small and large agreed; no discrepancy to distill.
            return False

        # Position gate: past t* the expected future saving of a new
        # experience no longer covers its generation cost — skip.
        if self.position_gate_rho is not None:
            n_q = self._n_by_query.get(point.query_name)
            if n_q:
                t_star = n_q / (1.0 + self.position_gate_rho)
                if t > t_star:
                    with self._lock:
                        self._stats["skipped_position_gate"] += 1
                    return False

        key = (point.doc_id, point.query_name)
        with self._lock:
            if key in self._claimed:
                return False
            cur = self._count_by_query.get(point.query_name, 0)
            if cur >= self.max_per_query:
                self._stats["skipped_cap"] += 1
                return False
            # Reserve the slot so a concurrent trigger for the same query
            # won't also claim it.
            self._count_by_query[point.query_name] = cur + 1
            self._claimed.add(key)
        return True

    def _release_slot(self, point: DataPoint) -> None:
        """Undo the reservation made in `should_generate`. Caller holds the lock."""
        self._count_by_query[point.query_name] = max(
            0, self._count_by_query.get(point.query_name, 0) - 1
        )
        self._claimed.discard((point.doc_id, point.query_name))

    # ── LLM calls ────────────────────────────────────────────────────────

    def _call_small_explain(
        self, point: DataPoint, small_pred: str
    ) -> tuple[str, Any]:
        messages = small_model_explain_messages(
            document_text=point.doc_text,
            query_description=point.query_description,
            small_prediction=small_pred,
            domain=self.domain,
        )
        resp = self.small_client.with_options(
            timeout=self.request_timeout
        ).chat.completions.create(
            model=self.small_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.explain_max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, getattr(resp, "usage", None)

    def _call_large_synthesize(
        self,
        point: DataPoint,
        small_reasoning: str,
        small_pred: str,
        large_pred: str,
    ) -> tuple[str, Any]:
        messages = meta_prompt_messages(
            query_name=point.query_name,
            query_description=point.query_description,
            document_text=point.doc_text,
            small_reasoning=small_reasoning,
            small_prediction=small_pred,
            large_prediction=large_pred,
            domain=self.domain,
        )
        resp = self.large_client.with_options(
            timeout=self.request_timeout
        ).chat.completions.create(
            model=self.large_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.meta_max_tokens,
            **({"extra_body": self.large_extra_body} if self.large_extra_body else {}),
        )
        text = (resp.choices[0].message.content or "").strip()
        return text, getattr(resp, "usage", None)

    # ── Main entry ───────────────────────────────────────────────────────

    def generate(
        self, point: DataPoint, record: ProcessedRecord
    ) -> Experience | None:
        small_pred = record.small_prediction or ""
        large_pred = record.prediction  # final came from large (escalated)

        # Stale-stream skip: generation runs on a background executor, so by
        # the time this call starts the stream may already be fully processed —
        # an experience produced now could never be retrieved by any point.
        # Skip the two LLM calls entirely. The audience is the remaining STREAM
        # (reuse is mostly cross-query), so only "no points left at all" is a
        # safe zero-audience signal. Requires prefetch(); without it inert.
        with self._lock:
            n_total = sum(self._n_by_query.values())
            if n_total and len(self._pos_seen) >= n_total:
                self._stats["skipped_stale_stream"] += 1
                self._release_slot(point)
                return None

        with self._lock:
            self._stats["attempted"] += 1

        # Step A: small model rationalizes its prior (wrong) answer.
        _t = time.perf_counter()
        try:
            small_reasoning, se_usage = self._call_small_explain(point, small_pred)
            dt = time.perf_counter() - _t
            with self._lock:
                self._stats["t_small_explain_total"] += dt
                self._stats["n_small_explain_calls"] += 1
                if se_usage is not None:
                    self._stats["gen_small_prompt_tokens"] += se_usage.prompt_tokens or 0
                    self._stats["gen_small_completion_tokens"] += se_usage.completion_tokens or 0
        except Exception:
            dt = time.perf_counter() - _t
            with self._lock:
                self._stats["t_small_explain_total"] += dt
                self._stats["n_small_explain_calls"] += 1
                self._stats["llm_errors"] += 1
                self._release_slot(point)
            return None

        # Step B: large model synthesizes experience.
        _t = time.perf_counter()
        try:
            meta_raw, ls_usage = self._call_large_synthesize(
                point, small_reasoning, small_pred, large_pred
            )
            dt = time.perf_counter() - _t
            with self._lock:
                self._stats["t_large_synthesize_total"] += dt
                self._stats["n_large_synthesize_calls"] += 1
                if ls_usage is not None:
                    self._stats["gen_large_prompt_tokens"] += ls_usage.prompt_tokens or 0
                    self._stats["gen_large_completion_tokens"] += ls_usage.completion_tokens or 0
        except Exception:
            dt = time.perf_counter() - _t
            with self._lock:
                self._stats["t_large_synthesize_total"] += dt
                self._stats["n_large_synthesize_calls"] += 1
                self._stats["llm_errors"] += 1
                self._release_slot(point)
            return None

        try:
            parsed = parse_llm_experience_json(meta_raw)
        except Exception:
            with self._lock:
                self._stats["parse_failures"] += 1
                self._release_slot(point)
            return None

        exp_text = parsed.get("experience_text", "")

        # Build the Experience (subset of fields the retriever needs).
        exp = Experience(
            experience_id=f"online__{point.query_name}__doc_{point.doc_id}",
            source_query=point.query_name,
            source_doc_id=point.doc_id,
            source_doc_excerpt=point.doc_text[:20000],
            experience_text=exp_text,
        )
        with self._lock:
            self._stats["succeeded"] += 1
        return exp

    def stats(self) -> dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
            n_se = s["n_small_explain_calls"]
            n_ls = s["n_large_synthesize_calls"]
            s["avg_t_small_explain"] = (
                round(s["t_small_explain_total"] / n_se, 4)
                if n_se else None
            )
            s["avg_t_large_synthesize"] = (
                round(s["t_large_synthesize_total"] / n_ls, 4)
                if n_ls else None
            )
            s["t_small_explain_total"] = round(s["t_small_explain_total"], 3)
            s["t_large_synthesize_total"] = round(
                s["t_large_synthesize_total"], 3
            )
            s["per_query_counts"] = dict(self._count_by_query)
            s["max_per_query"] = self.max_per_query
            s["position_gate_rho"] = self.position_gate_rho
            if self.position_gate_rho is not None:
                s["t_star_by_query"] = {
                    q: round(n / (1.0 + self.position_gate_rho), 1)
                    for q, n in self._n_by_query.items()
                }
            return s
