"""Streaming pipeline runner.

Composes pre-built stage modules into an in-order run over a list of
DataPoints. The runner contains no method-specific logic — all variant choices
live in the stage modules it was handed.

Protocol:
  - Data points are processed in the order provided.
  - For the point currently being processed, only information from earlier
    points is visible (few-shot pool grows monotonically).
  - Experience pool may grow online: experiences generated from escalations
    are installed in data order (see Concurrency below).

Concurrency:
  - With `workers=1` the loop is strictly sequential (each point sees all
    prior completions).
  - With `workers>1` the runner keeps a bounded sliding window of at most
    `workers` points in flight. It blocks only on the OLDEST in-flight
    point, installs it into state (strictly in original data order), and
    immediately admits the next point. "No future leakage" still holds —
    installs are in data order, so anything visible in `state` is a
    data-order predecessor. The relaxation vs. strict sequential is that
    a point does not see its up-to-`workers-1` immediate predecessors
    that are still in flight when it is admitted (same worst case as the
    old fixed-chunk mode, without the per-chunk barrier stall).
  - Experience generation runs on its own small executor (`gen_workers`)
    and OVERLAPS point processing instead of blocking between chunks.
    Produced experiences install serially in trigger (data) order as
    soon as they are ready, so the retriever's pool still advances
    deterministically in order; a new experience becomes visible to any
    point admitted after its install.

Router escalation protocol:
  - `router.route(state, point)` picks the initial model ("small" | "large").
  - When the initial choice is "small" AND the router defines
    `should_escalate(state, point, small_pred, small_raw, small_confidence)`,
    the runner consults it AFTER the small call. If it returns True and a
    large client is configured, the runner calls the large model and uses
    its prediction as the final answer (the small answer is still recorded
    under `small_prediction` / `small_confidence`).
"""

from __future__ import annotations

import csv
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Literal, Protocol

from openai import OpenAI

from pipeline.common.confidence import (
    extract_yes_no_features,
    extract_yes_no_features_3way,
)
from pipeline.common.prompts import (
    DEFAULT_LARGE_SYSTEM,
    SMALL_2WAY_SYSTEMS,
    build_replay_messages,
    build_replay_messages_3way,
    parse_yes_no,
)
from pipeline.common.types import (
    DataPoint,
    Experience,
    FewShotDemo,
    ProcessedRecord,
    RunState,
)


class _ExperienceRetrieverP(Protocol):
    def retrieve(self, state: RunState, point: DataPoint, k: int) -> list[Experience]: ...
    # Online generator support: optional. Runner only calls add() when it has
    # an experience_generator configured AND the retriever advertises add().
    # def add(self, exp: Experience) -> None: ...


class _FewShotRetrieverP(Protocol):
    def retrieve(self, state: RunState, point: DataPoint, k: int) -> list[FewShotDemo]: ...


class _RouterP(Protocol):
    def route(self, state: RunState, point: DataPoint) -> Literal["small", "large"]: ...


class _ExperienceGeneratorP(Protocol):
    def should_generate(
        self, point: DataPoint, record: ProcessedRecord
    ) -> bool: ...

    def generate(
        self, point: DataPoint, record: ProcessedRecord
    ) -> Experience | None: ...


class StreamingRunner:
    """Two-model cascade runner.

    Prompt contract (enforced — not configurable):
      * Small LLM uses the 3-way prompt (True / False / Unsure).
      * Large LLM uses the clean 2-way prompt (Yes / No) with NO
        retrieved experiences and NO few-shot demonstrations. The large
        model is the cascade's reference oracle and must stay
        uncontaminated by small-model-specific guidance. Live large calls
        therefore match the cached zero-shot 70B predictions bit-exactly.

    "Unsure" output handling:
      * The 3-way feature extractor surfaces ``prediction == "Unsure"``
        when p_unsure is the argmax. ``small_prediction`` propagates that
        through to ``ProcessedRecord``.
      * Routers escalate explicitly on ``prediction == "Unsure"``
        (treated like ``"UNKNOWN"``). The conf-threshold path still works
        too because ``confidence = max(p_yes, p_no)`` drops when p_unsure
        is high.
      * Few-shot retriever filters demos to {"Yes","No"}, so Unsure
        points are not used as demos.
      * Experience generator treats Unsure ≠ Yes/No as a discrepancy and
        will fire (subject to its per-query cap).
    """

    def __init__(
        self,
        *,
        experience_retriever: _ExperienceRetrieverP | None,
        fewshot_retriever: _FewShotRetrieverP | None,
        router: _RouterP,
        small_client: OpenAI,
        small_model: str,
        large_client: OpenAI | None,
        large_model: str | None,
        k_experiences: int,
        k_fewshot: int,
        max_tokens: int = 10,
        temperature: float = 0.0,
        request_timeout: float = 60.0,
        small_logprobs: bool = True,
        top_logprobs: int = 20,
        workers: int = 1,
        gen_workers: int = 4,
        experience_generator: _ExperienceGeneratorP | None = None,
        large_prediction_cache=None,
        small_prompt_mode: Literal["2way", "3way"] = "3way",
        small_prompt_style: str = "default",
        large_system_prompt: str = DEFAULT_LARGE_SYSTEM,
    ):
        self.exp_r = experience_retriever
        self.fs_r = fewshot_retriever
        self.router = router
        self.small_client = small_client
        self.small_model = small_model
        self.large_client = large_client
        self.large_model = large_model
        self.k_exp = k_experiences
        self.k_fs = k_fewshot
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.small_logprobs = small_logprobs
        self.top_logprobs = top_logprobs
        self.workers = max(1, int(workers))
        # Concurrency of the background experience-generation executor
        # (each generation is a serial small-explain + large-synthesize
        # LLM chain). Deliberately smaller than `workers` so overlapped
        # generation can't starve the small-model server that the main
        # point stream depends on.
        self.gen_workers = max(1, int(gen_workers))
        self.exp_gen = experience_generator
        # Optional precomputed-prediction cache (LargePredictionCache).
        # When set, _call_large() consults it first and skips the live LLM
        # call on hits. Misses fall through to the live call (and store the
        # result back if the cache was constructed with persist_to). See
        # pipeline/common/large_prediction_cache.py for semantic notes.
        self.large_cache = large_prediction_cache
        if small_prompt_mode not in ("2way", "3way"):
            raise ValueError(
                f"small_prompt_mode must be '2way' or '3way', got "
                f"{small_prompt_mode!r}"
            )
        self.small_prompt_mode = small_prompt_mode
        if small_prompt_style not in SMALL_2WAY_SYSTEMS:
            raise ValueError(
                f"small_prompt_style must be one of "
                f"{sorted(SMALL_2WAY_SYSTEMS)}, got {small_prompt_style!r}"
            )
        self.small_prompt_style = small_prompt_style
        # Base system sentence for the large-LLM 2-way prompt. Dataset-specific
        # so the domain framing matches the corpus; defaults to the LegalBench
        # wording for backward bit-exactness.
        self.large_system_prompt = large_system_prompt
        # Log of every experience added online during the run (for output JSON).
        self.online_experiences: list[dict] = []
        # When True, the runner is in frozen-inference mode: it still routes,
        # calls the small model, and records the router's score, but performs
        # NO online updates — no router refit, no retriever counter update, no
        # experience generation.
        self.frozen: bool = False
        # Optional streaming cost-abort (opt-in via env vars). When a run's
        # accumulated USD cost (small on every processed point + per-doc 70B
        # large_in on escalated points) exceeds MOP_COST_ABORT_USD, the run
        # stops early and marks itself aborted.
        self.cost_aborted = False
        self._abort_info: dict | None = None
        self._cost_cap, self._cost_map = self._load_cost_abort()

    # Cost model constants (USD per token).
    _S_IN, _S_OUT = 1e-8, 5e-8
    _L_IN, _L_OUT, _L_OUT_TOK = 5.2e-7, 7.5e-7, 1

    @staticmethod
    def _load_cost_abort() -> tuple[float | None, dict]:
        cap = os.environ.get("MOP_COST_ABORT_USD")
        tbl = os.environ.get("MOP_COST_TOKEN_TABLE")
        if not cap or not tbl:
            return None, {}
        cost_map: dict[tuple[str, int], int] = {}
        with open(tbl) as f:
            for r in csv.DictReader(f):
                cost_map[(r["query_name"], int(r["doc_id"]))] = int(r["large_in"])
        return float(cap), cost_map

    def _record_cost_usd(self, rec) -> float:
        """Incremental USD cost this record adds (small always; large_in from the
        token table on escalation, since cached large calls meter 0 tokens)."""
        c = rec.small_prompt_tokens * self._S_IN + rec.small_completion_tokens * self._S_OUT
        if rec.escalated:
            li = self._cost_map.get((rec.query_name, int(rec.doc_id)), 0)
            c += li * self._L_IN + self._L_OUT_TOK * self._L_OUT
        return c

    def _trip_cost_abort(self, acc_cost: float, processed: int, n_total: int) -> None:
        self.cost_aborted = True
        self._abort_info = {
            "processed": processed, "n_total": n_total,
            "acc_cost_usd": round(acc_cost, 4), "cap_usd": self._cost_cap,
        }
        print(
            f"  [COST-ABORT] ${acc_cost:.3f} > cap ${self._cost_cap:.2f} "
            f"at {processed}/{n_total} — abandoning cell",
            flush=True,
        )

    def _call_small(self, messages: list[dict]) -> dict:
        """Call the small model and return the rich feature dict.

        Keys: prediction, raw, confidence, p_yes, p_no, margin, entropy_2way,
        logprob_yes, logprob_no, logprobs_available.
        """
        kwargs: dict = dict(
            model=self.small_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if self.small_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.top_logprobs
        resp = self.small_client.with_options(
            timeout=self.request_timeout
        ).chat.completions.create(**kwargs)
        if self.small_prompt_mode == "3way":
            feats = extract_yes_no_features_3way(resp)
        else:
            feats = extract_yes_no_features(resp)
        # Live token usage for cost accounting. The small prompt is the only
        # one that varies (few-shot demos), so this is the faithful source.
        usage = getattr(resp, "usage", None)
        feats["prompt_tokens"] = usage.prompt_tokens if usage else 0
        feats["completion_tokens"] = usage.completion_tokens if usage else 0
        return feats

    def _call_large(
        self, messages: list[dict], point: DataPoint | None = None
    ) -> tuple[str, str, int, int]:
        """Return (prediction, raw, prompt_tokens, completion_tokens).

        On a prediction-cache hit there is no live call, so the token counts
        are 0; the offline clean-2-way token table charges the large cost for
        cached escalations at aggregation time.
        """
        # Cached-prediction shortcut. When a LargePredictionCache is wired
        # in AND we know which point we're answering, look up the saved
        # (prediction, raw) and skip the live LLM call. The cache's
        # semantic note (zero-shot vs experience-augmented prompt) is
        # documented in pipeline/common/large_prediction_cache.py.
        if self.large_cache is not None and point is not None:
            hit = self.large_cache.lookup(point)
            if hit is not None:
                return hit[0], hit[1], 0, 0
        if self.large_client is None or self.large_model is None:
            raise RuntimeError("routed to 'large' but no large client configured")
        resp = self.large_client.with_options(
            timeout=self.request_timeout
        ).chat.completions.create(
            model=self.large_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        pred = parse_yes_no(raw)
        usage = getattr(resp, "usage", None)
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        # Store back to the cache so subsequent points (or runs) hit on this key.
        if self.large_cache is not None and point is not None:
            try:
                self.large_cache.store(point, pred, raw)
            except Exception:
                pass
        return pred, raw, pt, ct

    def _process_point(
        self, state: RunState, point: DataPoint
    ) -> ProcessedRecord:
        """Per-point body. Reads `state` only; never mutates it."""
        t_retrieve_exp: float | None = None
        t_retrieve_fs: float | None = None
        t_router_signals: float | None = None
        t_observe: float | None = None
        exp_hits: list[Experience] = []
        if self.exp_r is not None:
            _t = time.perf_counter()
            exp_hits = self.exp_r.retrieve(state, point, self.k_exp)
            t_retrieve_exp = time.perf_counter() - _t
        fs_hits: list[FewShotDemo] = []
        if self.fs_r is not None:
            _t = time.perf_counter()
            fs_hits = self.fs_r.retrieve(state, point, self.k_fs)
            t_retrieve_fs = time.perf_counter() - _t
        # Time the router decision (route + should_escalate). Both are
        # CPU-only for non-LR routers (microseconds); the LR routers spend
        # a few hundred microseconds on scaler.transform + predict_proba.
        _t = time.perf_counter()
        initial_route = self.router.route(state, point)
        t_route_decision = time.perf_counter() - _t
        if self.small_prompt_mode == "3way":
            messages = build_replay_messages_3way(
                document_text=point.doc_text,
                query_description=point.query_description,
                retrieved_experiences=exp_hits,
                fewshot_demos=fs_hits,
                system_style=self.small_prompt_style,
            )
        else:
            messages = build_replay_messages(
                document_text=point.doc_text,
                query_description=point.query_description,
                retrieved_experiences=exp_hits,
                fewshot_demos=fs_hits,
                system_base=SMALL_2WAY_SYSTEMS[self.small_prompt_style],
            )
        # Large-LLM prompt is ALWAYS the clean 2-way prompt with NO retrieved
        # experiences and NO few-shot demos: the large model is the cascade's
        # reference oracle and must stay clean (uncontaminated by small-model-
        # specific guidance).
        large_messages = build_replay_messages(
            document_text=point.doc_text,
            query_description=point.query_description,
            retrieved_experiences=None,
            fewshot_demos=None,
            system_base=self.large_system_prompt,
        )

        ts = time.perf_counter()
        small_pred: str | None = None
        small_raw = ""
        small_conf: float | None = None
        small_p_yes: float | None = None
        small_p_no: float | None = None
        small_p_unsure: float | None = None
        final_pred = "UNKNOWN"
        final_raw = ""
        final_from: Literal["small", "large"] = initial_route
        escalated = False
        err = None
        t_small_call: float | None = None
        t_large_call: float | None = None
        router_p_z1: float | None = None
        router_bootstrap: bool | None = None
        small_pt = small_ct = large_pt = large_ct = 0
        verify_calls = verify_pt = verify_ct = 0
        verify_score: float | None = None

        try:
            if initial_route == "small":
                _t = time.perf_counter()
                small_feats = self._call_small(messages)
                t_small_call = time.perf_counter() - _t
                small_pt = small_feats.get("prompt_tokens", 0)
                small_ct = small_feats.get("completion_tokens", 0)
                small_pred = small_feats["prediction"]
                small_raw = small_feats["raw"]
                small_conf = small_feats["confidence"]
                small_p_yes = small_feats.get("p_yes")
                small_p_no = small_feats.get("p_no")
                small_p_unsure = small_feats.get("p_unsure")
                final_pred, final_raw = small_pred, small_raw
                if self.large_client is not None:
                    # Retriever-side signals (top-k score stats,
                    # helpfulness aggregates, pool state). The base
                    # retriever returns None — only retrievers that
                    # care about the routing signal compute one.
                    router_sig = None
                    if self.exp_r is not None:
                        _t = time.perf_counter()
                        router_sig = self.exp_r.router_signals(point, exp_hits)
                        t_router_signals = time.perf_counter() - _t
                    _t = time.perf_counter()
                    decision = self.router.should_escalate(
                        state, point, small_pred, small_raw, small_conf,
                        small_features=small_feats,
                        retrieved_experiences=exp_hits,
                        retriever_signals=router_sig,
                    )
                    t_route_decision += time.perf_counter() - _t
                    router_p_z1 = decision.p_z1
                    router_bootstrap = decision.bootstrap
                    # Cost of any extra small calls the router made while
                    # deciding (AutoMix self-verification). 0 for other routers.
                    verify_calls = decision.verify_small_calls
                    verify_pt = decision.verify_small_prompt_tokens
                    verify_ct = decision.verify_small_completion_tokens
                    verify_score = decision.verify_score
                    if decision.escalate:
                        try:
                            _t = time.perf_counter()
                            final_pred, final_raw, large_pt, large_ct = self._call_large(large_messages, point=point)
                            t_large_call = time.perf_counter() - _t
                            final_from = "large"
                            escalated = True
                        except Exception as e:
                            err = f"escalate_err: {type(e).__name__}: {e}"
                # Online routing feedback: after a completed escalation,
                # give the router AND retriever a chance to update on
                # the observed (small_pred, large_pred) pair. Both
                # methods are part of the Protocol with no-op defaults.
                if escalated and not self.frozen:
                    _t = time.perf_counter()
                    self.router.on_escalation_observed(
                        state, point, small_pred, final_pred,
                        small_features=small_feats,
                        small_confidence=small_conf,
                        n_retrieved_experiences=len(exp_hits),
                        retrieved_experiences=exp_hits,
                        retriever_signals=router_sig,
                    )
                    if self.exp_r is not None:
                        self.exp_r.observe_escalation(
                            exp_hits, small_pred, final_pred,
                            query_name=point.query_name,
                            small_confidence=small_conf,
                            point=point,
                        )
                    t_observe = time.perf_counter() - _t
            else:
                _t = time.perf_counter()
                final_pred, final_raw, large_pt, large_ct = self._call_large(large_messages, point=point)
                t_large_call = time.perf_counter() - _t
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            final_pred, final_raw = "UNKNOWN", ""
        latency = time.perf_counter() - ts

        return ProcessedRecord(
            doc_id=point.doc_id,
            query_name=point.query_name,
            ground_truth=point.ground_truth,
            prediction=final_pred,
            raw=final_raw if err is None else (err if not final_raw else final_raw),
            routed_to=final_from,
            retrieved_experiences=[
                {
                    "experience_id": e.experience_id,
                    "source_query": e.source_query,
                    "source_doc_id": e.source_doc_id,
                    "score": round(e.score, 4),
                }
                for e in exp_hits
            ],
            retrieved_fewshot=[
                {
                    "doc_id": d.doc_id,
                    "prediction": d.prediction,
                    "produced_by": d.produced_by,
                    "score": round(d.score, 4),
                }
                for d in fs_hits
            ],
            latency=round(latency, 4),
            small_prediction=small_pred,
            small_confidence=(
                round(small_conf, 4) if small_conf is not None else None
            ),
            small_p_yes=(round(small_p_yes, 6) if small_p_yes is not None else None),
            small_p_no=(round(small_p_no, 6) if small_p_no is not None else None),
            small_p_unsure=(
                round(small_p_unsure, 6) if small_p_unsure is not None else None
            ),
            escalated=escalated,
            router_p_z1=(
                round(router_p_z1, 6) if router_p_z1 is not None else None
            ),
            router_bootstrap=router_bootstrap,
            t_small=(
                round(t_small_call, 4) if t_small_call is not None else None
            ),
            t_route_decision=round(t_route_decision, 6),
            t_large=(
                round(t_large_call, 4) if t_large_call is not None else None
            ),
            t_retrieve_exp=(
                round(t_retrieve_exp, 6) if t_retrieve_exp is not None else None
            ),
            t_retrieve_fs=(
                round(t_retrieve_fs, 6) if t_retrieve_fs is not None else None
            ),
            t_router_signals=(
                round(t_router_signals, 6)
                if t_router_signals is not None else None
            ),
            t_observe=(
                round(t_observe, 6) if t_observe is not None else None
            ),
            small_prompt_tokens=small_pt,
            small_completion_tokens=small_ct,
            large_prompt_tokens=large_pt,
            large_completion_tokens=large_ct,
            verify_small_calls=verify_calls,
            verify_small_prompt_tokens=verify_pt,
            verify_small_completion_tokens=verify_ct,
            verify_score=verify_score,
        )

    def _maybe_generate_and_install(
        self, point: DataPoint, rec: ProcessedRecord
    ) -> None:
        """Sequential-path (workers=1) generation: fire the generator for a
        qualifying point inline, then install the produced experience so the
        NEXT point can retrieve it. The pipelined path (workers>1) handles
        generation itself, overlapped on a background executor."""
        # Frozen-inference mode grows nothing online: skip generation entirely
        # so the experience pool stays at its loaded state (matches the
        # documented frozen contract — no experience generation).
        if self.frozen:
            return
        if self.exp_gen is None or self.exp_r is None:
            return
        if not self.exp_gen.should_generate(point, rec):
            return
        try:
            exp = self.exp_gen.generate(point, rec)
        except Exception:
            exp = None
        if exp is None:
            return
        try:
            self.exp_r.add(exp)
        except Exception:
            return
        self.online_experiences.append({
            "experience_id": exp.experience_id,
            "source_query": exp.source_query,
            "source_doc_id": exp.source_doc_id,
            "experience_text": exp.experience_text,
        })

    def run(
        self, data: list[DataPoint], progress_every: int = 25
    ) -> tuple[list[ProcessedRecord], RunState]:
        state = RunState()
        records: list[ProcessedRecord] = []
        t0 = time.perf_counter()
        self.cost_aborted = False
        self._abort_info = None
        acc_cost = 0.0

        if self.workers <= 1:
            # Strict sequential path.
            for i, point in enumerate(data, 1):
                rec = self._process_point(state, point)
                records.append(rec)
                state.record(point, rec)
                # Online experience generation, inline.
                self._maybe_generate_and_install(point, rec)
                if self._cost_cap is not None:
                    acc_cost += self._record_cost_usd(rec)
                    if acc_cost > self._cost_cap:
                        self._trip_cost_abort(acc_cost, i, len(data))
                        break
                if i % progress_every == 0 or i == len(data):
                    el = time.perf_counter() - t0
                    n_esc = sum(1 for r in records if r.escalated)
                    n_online = len(self.online_experiences)
                    print(
                        f"  [{i}/{len(data)}] {el:.1f}s  "
                        f"({i/el:.2f} pts/s)  escalated={n_esc}  "
                        f"online_exp={n_online}",
                        flush=True,
                    )
            return records, state

        # Pipelined sliding-window path (workers > 1).
        #
        # Invariants:
        #   * At most W points in flight (admitted − installed ≤ W).
        #   * Installs happen strictly in data order — the loop blocks only
        #     on the OLDEST in-flight point, never on the whole window, so
        #     one slow call no longer stalls W−1 finished siblings.
        #   * Experience generation is submitted at install time to its own
        #     `gen_workers`-sized executor and overlaps point processing;
        #     produced experiences install serially in trigger (data) order
        #     as soon as the head of the generation queue resolves.
        W = self.workers
        n = len(data)
        n_esc = 0

        def _progress(done: int) -> None:
            el = time.perf_counter() - t0
            print(
                f"  [{done}/{n}] {el:.1f}s  "
                f"({done/el:.2f} pts/s)  escalated={n_esc}  "
                f"online_exp={len(self.online_experiences)}",
                flush=True,
            )

        def _install_experience(fut: Any) -> None:
            try:
                exp = fut.result()
            except Exception:
                exp = None
            if exp is None:
                return
            try:
                self.exp_r.add(exp)
            except Exception:
                return
            self.online_experiences.append({
                "experience_id": exp.experience_id,
                "source_query": exp.source_query,
                "source_doc_id": exp.source_doc_id,
                "experience_text": exp.experience_text,
            })

        generation_on = (
            self.exp_gen is not None
            and self.exp_r is not None
            and not self.frozen
        )
        gen_pool = (
            ThreadPoolExecutor(max_workers=self.gen_workers)
            if generation_on else None
        )
        # Generation futures in trigger order. Install only from the head
        # so the retriever sees additions in data order even when a later
        # generation finishes first.
        gen_pending: deque[Any] = deque()

        pool = ThreadPoolExecutor(max_workers=W)
        futures: dict[int, Any] = {}
        next_submit = 0

        def _fill_window(installed: int) -> None:
            nonlocal next_submit
            while next_submit < n and next_submit - installed < W:
                futures[next_submit] = pool.submit(
                    self._process_point, state, data[next_submit]
                )
                next_submit += 1

        try:
            _fill_window(0)
            for i, point in enumerate(data):
                rec = futures.pop(i).result()
                records.append(rec)
                state.record(point, rec)
                if rec.escalated:
                    n_esc += 1
                if generation_on and self.exp_gen.should_generate(point, rec):
                    gen_pending.append(
                        gen_pool.submit(self.exp_gen.generate, point, rec)
                    )
                while gen_pending and gen_pending[0].done():
                    _install_experience(gen_pending.popleft())
                _fill_window(i + 1)
                if self._cost_cap is not None:
                    acc_cost += self._record_cost_usd(rec)
                    if acc_cost > self._cost_cap:
                        self._trip_cost_abort(acc_cost, i + 1, n)
                        break
                if (i + 1) % progress_every == 0 or i + 1 == n:
                    _progress(i + 1)
        finally:
            # On a cost abort, queued-but-unstarted work is cancelled;
            # already-running calls finish (bounded by request_timeout) and
            # are simply not installed. On a normal finish both queues are
            # already empty/consumed.
            pool.shutdown(wait=True, cancel_futures=True)
            if gen_pool is not None:
                if not self.cost_aborted:
                    # Wait out in-flight generations so the run record (and
                    # the retriever pool handed back to the driver) is
                    # complete.
                    while gen_pending:
                        _install_experience(gen_pending.popleft())
                gen_pool.shutdown(wait=True, cancel_futures=True)
        return records, state
