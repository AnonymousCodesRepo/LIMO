"""Top-k semantic retriever with online escalation-based reweighting.

Extends ``TopKSemanticExperienceRetriever`` with per-experience positive /
negative counters updated on each observed escalation, and a ``retrieve()``
that rescores the cosine shortlist by multiplying similarity with a
Laplace-smoothed helpfulness weight per experience.

Signal (maintained by ``observe_escalation``, called by the runner whenever
the router escalates after a small-model call):

  - positive: ``small_pred == final_pred``
              (small matched large — the retrieved experiences were "good
              enough" that the small model agreed with the large one).
  - negative: ``small_pred != final_pred`` AND ``final_pred`` is parseable
              (the experiences did not prevent a disagreement).
  - skip:    ``final_pred == "UNKNOWN"`` or ``None`` (no reliable signal).

Per-experience weight at retrieval time:

    w(E) = (pos + alpha) / (pos + neg + alpha + beta)

With default ``alpha = beta = 1.0`` this is plain Laplace smoothing, so an
unseen experience has ``w = 0.5`` and ``w`` stays strictly in ``(0, 1)``.
When ``pos + neg < min_observations`` we fall back to the neutral prior
``0.5`` so a handful of early, noisy signals don't dominate the ranking.

Final score:

    adjusted = ((cos + 1) / 2) * w(E)

The ``(cos + 1) / 2`` shift maps cosine into ``[0, 1]`` so ordering stays
monotone even in the (rare) case ``cos < 0``. When weights are equal across
items the adjusted order matches the raw cosine order, so with no
observations the retriever behaves like the plain top-k.
"""

from __future__ import annotations

import math
import threading
from typing import Literal

import numpy as np

from pipeline.common.types import DataPoint, Experience, RunState

from .topk_semantic import TopKSemanticExperienceRetriever


ScoringMode = Literal["rerank", "sample"]


class TopKSemanticOnlineExperienceRetriever(TopKSemanticExperienceRetriever):
    """Semantic retriever that learns from escalation observations.

    Inherits ``add()``, ``size()``, and the embedding machinery from the
    parent, so online experience generation still works unchanged.
    """

    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        beta: float = 1.0,
        shortlist_mult: int = 2,
        min_observations: int = 4,
        scoring_mode: ScoringMode = "rerank",
        rerank_seed: int = 0,
        confidence_weighted: bool = False,
        mmr_lambda: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0 (Laplace smoothing).")
        if shortlist_mult < 1:
            raise ValueError("shortlist_mult must be >= 1.")
        if scoring_mode not in ("rerank", "sample"):
            raise ValueError(f"unknown scoring_mode: {scoring_mode!r}")
        if not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(
                f"mmr_lambda must be in [0, 1], got {mmr_lambda}"
            )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.shortlist_mult = int(shortlist_mult)
        self.min_observations = int(min_observations)
        self.scoring_mode: ScoringMode = scoring_mode
        self.confidence_weighted = bool(confidence_weighted)
        self.mmr_lambda = float(mmr_lambda)
        # Global counters (pooled across queries). Populated lazily.
        self.stats: dict[str, dict[str, float]] = {}
        # Per-(experience, query) counters. Same structure, keyed by tuple.
        # Always populated alongside the global counters so a run can be
        # analysed per-query later, but retrieval reads only the global
        # weight (no live per-query reweighting in production).
        self.stats_pq: dict[tuple[str, str], dict[str, float]] = {}
        self._stats_lock = threading.Lock()
        self._rng = np.random.default_rng(int(rerank_seed))
        self._rng_lock = threading.Lock()

    # ----- weighting --------------------------------------------------------

    def _weight(self, exp_id: str, query_name: str | None = None) -> float:
        """Helpfulness weight for an experience: Laplace-smoothed pos /
        (pos + neg) ratio over global pooled counters. Returns the neutral
        0.5 prior until ``min_observations`` of evidence accumulate."""
        with self._stats_lock:
            s = self.stats.get(exp_id)
            if s is None:
                return 0.5
            pos, neg = float(s["pos"]), float(s["neg"])
        if pos + neg < self.min_observations:
            return 0.5
        return (pos + self.alpha) / (pos + neg + self.alpha + self.beta)

    # ----- retrieval --------------------------------------------------------

    def retrieve(
        self, state: RunState, point: DataPoint, k: int
    ) -> list[Experience]:
        if k <= 0:
            return []
        shortlist_k = k * self.shortlist_mult
        shortlist = super().retrieve(state, point, shortlist_k)
        if not shortlist:
            return shortlist

        adjusted: list[Experience] = []
        for e in shortlist:
            w = self._weight(e.experience_id, query_name=point.query_name)
            sim_pos = 0.5 * (float(e.score) + 1.0)
            if sim_pos < 0.0:
                sim_pos = 0.0
            adj = sim_pos * w
            adjusted.append(Experience(
                experience_id=e.experience_id,
                source_query=e.source_query,
                source_doc_id=e.source_doc_id,
                source_doc_excerpt=e.source_doc_excerpt,
                experience_text=e.experience_text,
                applicability_signal=e.applicability_signal,
                score=float(adj),
            ))

        if self.scoring_mode == "rerank":
            adjusted.sort(key=lambda x: -x.score)
            if self.mmr_lambda < 1.0 and len(adjusted) > k:
                return self._mmr_select(adjusted, k)
            return adjusted[:k]

        # "sample": weighted sampling without replacement over the shortlist.
        if len(adjusted) <= k:
            adjusted.sort(key=lambda x: -x.score)
            return adjusted
        probs = np.array([x.score for x in adjusted], dtype=np.float64)
        probs = np.maximum(probs, 1e-12)
        probs /= probs.sum()
        with self._rng_lock:
            idx = self._rng.choice(
                len(adjusted), size=k, replace=False, p=probs,
            )
        sampled = [adjusted[int(i)] for i in idx]
        sampled.sort(key=lambda x: -x.score)
        return sampled

    # ----- MMR diversity ---------------------------------------------------

    def _mmr_select(
        self, candidates: list[Experience], k: int
    ) -> list[Experience]:
        """Iterative greedy MMR selection over the already-scored shortlist.

        For each pick, score is:

            mmr(e) = mmr_lambda * adj_score(e)
                   - (1 - mmr_lambda) * max_cosine(e, already_selected)

        ``adj_score`` is the input ``e.score`` (already weight × cosine).
        Cosine similarity for the diversity penalty is computed from the
        embeddings stored in ``self._exp_vecs``. Experiences whose
        embedding can't be found fall back to a 0 similarity penalty
        (no penalty), so the MMR pass degrades gracefully to plain top-k.
        """
        if k <= 0 or not candidates:
            return []
        # Pre-fetch embeddings for the shortlist (k * shortlist_mult entries
        # in practice, so the per-shortlist O(N²) MMR is cheap).
        vecs: list[np.ndarray | None] = [
            self._vec_for_exp_id(e.experience_id) for e in candidates
        ]
        selected: list[Experience] = []
        selected_idx: list[int] = []
        remaining: set[int] = set(range(len(candidates)))
        lam = self.mmr_lambda

        while remaining and len(selected) < k:
            best_idx = -1
            best_score = -float("inf")
            for i in remaining:
                e = candidates[i]
                relevance = float(e.score)
                # Diversity penalty: max cosine to anything already picked.
                penalty = 0.0
                if selected_idx and vecs[i] is not None:
                    sims = []
                    vi = vecs[i]
                    for j in selected_idx:
                        vj = vecs[j]
                        if vj is None:
                            continue
                        sims.append(float(vi @ vj))
                    if sims:
                        penalty = max(sims)
                mmr_score = lam * relevance - (1.0 - lam) * penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(candidates[best_idx])
            selected_idx.append(best_idx)
            remaining.discard(best_idx)
        return selected

    # ----- online feedback --------------------------------------------------

    def observe_escalation(
        self,
        exp_hits: list[Experience],
        small_pred: str | None,
        final_pred: str | None,
        query_name: str | None = None,
        small_confidence: float | None = None,
        point: "DataPoint | None" = None,
        **_unused,
    ) -> None:
        """Called by the runner on every completed escalation.

        Treats ``final_pred in (None, "UNKNOWN")`` as "no signal" and skips
        the update — we have no reliable large-model answer to compare
        against.

        ``query_name``: the current point's query — populates per-(exp,
        query) counters in ``stats_pq`` for offline analysis. Retrieval
        does not consult these in production; only the global pool drives
        the helpfulness weight at retrieval time.

        ``small_confidence``: max(p_yes, p_no) from the small-call
        logprobs. When ``confidence_weighted=True``, increments are scaled
        by this value (so a confident agreement moves the weight more than
        a borderline one). When False, ignored — increments are 1.0 each.
        """
        if not exp_hits:
            return
        if final_pred is None or final_pred == "UNKNOWN":
            return
        positive = (small_pred is not None) and (small_pred == final_pred)
        # Increment magnitude. With confidence_weighted, scale by the
        # small-call confidence (clamped to [0, 1]).
        if self.confidence_weighted and small_confidence is not None:
            try:
                inc = float(small_confidence)
            except (TypeError, ValueError):
                inc = 1.0
            inc = max(0.0, min(1.0, inc))
        else:
            inc = 1.0
        with self._stats_lock:
            for e in exp_hits:
                # Global counter (always updated).
                s = self.stats.setdefault(
                    e.experience_id, {"pos": 0.0, "neg": 0.0}
                )
                if positive:
                    s["pos"] = float(s["pos"]) + inc
                else:
                    s["neg"] = float(s["neg"]) + inc
                # Per-(exp, query) counter (populated regardless of flag,
                # so analyses can inspect the data even when retrieval
                # didn't consult it).
                if query_name is not None:
                    s_pq = self.stats_pq.setdefault(
                        (e.experience_id, query_name),
                        {"pos": 0.0, "neg": 0.0},
                    )
                    if positive:
                        s_pq["pos"] = float(s_pq["pos"]) + inc
                    else:
                        s_pq["neg"] = float(s_pq["neg"]) + inc

    # ----- introspection ----------------------------------------------------

    def stats_snapshot(self) -> dict[str, dict[str, float]]:
        """Snapshot of the current counters with derived weights.

        Useful for logging / saving to the run's output JSON. The returned
        dict is a copy, safe to serialize.

        Counters are floats (cross-query decay and confidence-weighting can
        both produce non-integer counts); they're rounded to 4 decimal
        places in the snapshot for readability.
        """
        with self._stats_lock:
            items = [
                (eid, float(s["pos"]), float(s["neg"]))
                for eid, s in self.stats.items()
            ]
        out: dict[str, dict[str, float]] = {}
        for eid, pos, neg in items:
            out[eid] = {
                "pos": round(pos, 4),
                "neg": round(neg, 4),
                "weight": round(
                    (pos + self.alpha)
                    / (pos + neg + self.alpha + self.beta),
                    4,
                ),
            }
        return out

    def stats_snapshot_per_query(self) -> dict:
        """Snapshot of the per-(experience, query) counters.

        Returns a dict keyed by experience_id, with each value a dict
        mapping query_name → {pos, neg, weight}. Populated whenever
        observe_escalation knew the query_name; provided for offline
        analysis even though retrieval reads only the global pool.
        """
        with self._stats_lock:
            items = [
                (eid, qn, float(s["pos"]), float(s["neg"]))
                for (eid, qn), s in self.stats_pq.items()
            ]
        out: dict[str, dict[str, dict[str, float]]] = {}
        for eid, qn, pos, neg in items:
            out.setdefault(eid, {})[qn] = {
                "pos": round(pos, 4),
                "neg": round(neg, 4),
                "weight": round(
                    (pos + self.alpha)
                    / (pos + neg + self.alpha + self.beta),
                    4,
                ),
            }
        return out

    # ----- router signals --------------------------------------------------

    ROUTER_SIGNAL_NAMES: tuple[str, ...] = (
        "max_score",                 # max retriever score in top-k
        "mean_score",                # mean of top-k scores
        "score_std",                 # std of top-k scores (peakiness)
        "top1_minus_meantail",       # top1 − mean of top-{2..k} (gap)
        "frac_same_query",           # fraction of top-k with source_query==point.query_name
        "mean_global_helpfulness",   # mean (pos+α)/(pos+neg+α+β) across top-k
        "mean_per_query_helpfulness",# same but per-(exp, point.query_name)
        "mean_log_n_obs_norm",       # mean log(1+n_obs)/log(101) across top-k
        "n_exp_for_query_norm",      # |{exp in pool : source_query==point.query_name}|/16
        "mean_log_n_obs_q_norm",     # mean log(1+n_obs_q)/log(101) across top-k
    )

    def router_signals(
        self,
        point: DataPoint,
        exp_hits: list[Experience],
    ) -> np.ndarray:
        """Aggregate the top-k retrieval result into a fixed-length vector
        the router can consume as additional features.

        Returns a 10-d numpy array with the components named in
        ``ROUTER_SIGNAL_NAMES`` (most entries in [0, 1]; ``mean_score`` /
        ``score_std`` inherit the underlying score scale).
        """
        n = len(exp_hits) if exp_hits else 0
        out = np.zeros(len(self.ROUTER_SIGNAL_NAMES), dtype=np.float64)
        if n == 0:
            return out

        scores = np.array(
            [float(getattr(e, "score", 0.0)) for e in exp_hits],
            dtype=np.float64,
        )
        out[0] = float(scores.max())
        out[1] = float(scores.mean())
        out[2] = float(scores.std()) if n > 1 else 0.0
        if n >= 2:
            top1 = float(scores[0]) if n > 0 else 0.0
            tail_mean = float(scores[1:].mean())
            out[3] = top1 - tail_mean
        else:
            out[3] = 0.0

        out[4] = float(
            sum(1 for e in exp_hits if e.source_query == point.query_name) / n
        )

        # Counter-based aggregates: pull from inherited stats / stats_pq.
        gh, pqh, lno, lnoq = [], [], [], []
        with self._stats_lock:
            for e in exp_hits:
                s = self.stats.get(e.experience_id, {"pos": 0.0, "neg": 0.0})
                pos, neg = float(s["pos"]), float(s["neg"])
                gh.append(
                    (pos + self.alpha) / (pos + neg + self.alpha + self.beta)
                )
                lno.append(
                    math.log(1.0 + pos + neg) / math.log(101.0)
                )
                spq = self.stats_pq.get((e.experience_id, point.query_name))
                if spq is not None:
                    pq, nq = float(spq["pos"]), float(spq["neg"])
                else:
                    pq, nq = 0.0, 0.0
                pqh.append(
                    (pq + self.alpha) / (pq + nq + self.alpha + self.beta)
                )
                lnoq.append(
                    math.log(1.0 + pq + nq) / math.log(101.0)
                )
        out[5] = float(np.mean(gh)) if gh else 0.5
        out[6] = float(np.mean(pqh)) if pqh else 0.5
        out[7] = min(1.0, float(np.mean(lno))) if lno else 0.0
        out[9] = min(1.0, float(np.mean(lnoq))) if lnoq else 0.0

        # Pool-state count for current query (normalized by max_per_query=16).
        with self._lock:
            n_q = sum(
                1 for ex in self.experiences
                if ex.source_query == point.query_name
            )
        out[8] = min(1.0, n_q / 16.0)

        return out
