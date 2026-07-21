"""CARE — Calibrated Adaptive Retrieval of Experiences.

Online demonstration selection for in-context learning, in three stages:

  Stage 1 — Recall:    cosine over the dual-encoder shortlist (cheap, frozen).
  Stage 2 — Rerank:    a small Bayesian logistic-regression head trained
                       online from binary escalation outcomes, outputting a
                       calibrated P(small matches large | x, e).
  Stage 3 — Set sel.:  greedy MMR on top of the per-experience scores so the
                       returned set is relevance-and-diversity aware.

Supervision (this is the core algorithmic move):

  After each escalation we observe a single binary outcome
      z = 1[small_pred == large_pred]
  for the *whole set* S = {e_1, ..., e_k}. We decompose set-level credit
  to per-experience labels via the model's own marginal contribution:

      w_i = k * softmax_i([s_1, ..., s_k]) -- normalized so weights sum to k
      (s_i is the BLR-predicted helpfulness of e_i pre-update)

  An experience whose score drove the prediction gets the bulk of the credit
  (positive or negative); a near-uniform contribution gets ~uniform weight.

Inheritance reuses the engineering from FeatureLinUCBV3Retriever:

  * 19-dim hand-designed feature builder (cos, BM25, helpfulness, freshness,
    per-query observation counters, ...).
  * Embedding + BM25 caches and the prefetch() entry point.
  * The (pos, neg) and per-(exp, query) counters that supply features 3, 4, 5,
    10 in v3 — these stay populated by the inherited observe_escalation, so
    the *features* still move even though the *learner* is replaced.

The inherited LinUCB (A_inv, b) state is unused in CARE.
"""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from pipeline.common.types import DataPoint, Experience, RunState

from .feature_linucb_v3 import FeatureLinUCBV3Retriever, _FEATURE_DIM_V3
from .topk_semantic import TopKSemanticExperienceRetriever


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _sigmoid_clipped(z: float, eps: float = 1e-6) -> float:
    p = _sigmoid(z)
    if p < eps:
        return eps
    if p > 1.0 - eps:
        return 1.0 - eps
    return p


class CAREExperienceRetriever(FeatureLinUCBV3Retriever):
    """Bayesian-logistic-regression reranker over a cosine shortlist.

    Drop-in replacement for the FeatureLinUCB family. Same constructor
    plumbing (so existing point_cache / embed_client / prefetch wiring
    keeps working), same observe_escalation signature.

    Extra knobs (all keyword-only):

      prior_precision : float
          Prior precision τ on θ. A_inv starts at I/τ; smaller τ = looser
          prior = faster fit but noisier early. Default 1.0.

      mmr_lambda : float
          Already exists on the parent (range [0,1], 1.0 = no MMR). CARE's
          default is 0.7 (light diversification).
    """

    # ------------- construction -------------------------------------------

    def __init__(
        self,
        *args,
        prior_precision: float = 1.0,
        mmr_lambda: float = 0.7,
        **kwargs,
    ):
        # Default MMR unless the caller overrode it (parent accepts mmr_lambda).
        kwargs.setdefault("mmr_lambda", mmr_lambda)
        super().__init__(*args, **kwargs)
        if prior_precision <= 0:
            raise ValueError("prior_precision must be > 0")

        self.prior_precision = float(prior_precision)

        # BLR head: μ ∈ ℝ^d, A_inv ∈ ℝ^{d×d} (Laplace posterior covariance),
        # independent of the inherited (unused) LinUCB (A_inv, b) state.
        self._d_care = _FEATURE_DIM_V3
        self._blr_mu = np.zeros(self._d_care, dtype=np.float64)
        self._blr_A_inv = (
            np.eye(self._d_care, dtype=np.float64) / self.prior_precision
        )
        self._blr_n_updates = 0
        self._blr_lock = threading.Lock()

        # Diagnostic counters surfaced via care_snapshot().
        self._care_stats = {
            "n_observations": 0,
            "n_positives": 0,
            "n_negatives": 0,
            # rolling average credit weight across updates
            "avg_credit_weight": 0.0,
        }
        # Keep a rolling sum of credit weights for the moving average.
        self._credit_sum = 0.0
        self._credit_n = 0

    # ------------- BLR scoring ---------------------------------------------

    def _score_one(self, phi: np.ndarray, query_name: str) -> tuple[float, float]:
        """Return (predicted P(z=1|x,e), variance of logit) under the
        current BLR posterior.

        The variance is φ A_inv φ — surfaced through router_signals /
        set_features_for as an uncertainty summary (cheap; d=19).
        """
        with self._blr_lock:
            mu = self._blr_mu
            A_inv = self._blr_A_inv
        eta = float(mu @ phi)
        var = float(phi @ (A_inv @ phi))
        if var < 0.0:
            var = 0.0
        p = _sigmoid(eta)
        return p, var

    def _score_many(self, Phi: np.ndarray, query_name: str) -> np.ndarray:
        """Vectorized ``_score_one`` over the rows of ``Phi`` (n, d).

        One lock acquisition and one matrix product for the whole
        candidate set instead of two lock round trips per candidate."""
        with self._blr_lock:
            mu = self._blr_mu.copy()
        eta = Phi @ mu
        # Stable sigmoid: |eta| is clipped where exp() would overflow —
        # sigmoid saturates to 0/1 long before ±60 anyway.
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))

    # ------------- retrieve ------------------------------------------------

    def retrieve(  # type: ignore[override]
        self, state: RunState, point: DataPoint, k: int
    ) -> list[Experience]:
        if k <= 0:
            return []

        # Stream-position bookkeeping (inherited).
        with self._stream_lock:
            self._stream_position += 1

        # Stage 1 — Recall: cosine top-(m·k) shortlist via grandparent.
        shortlist_k = max(k * self.shortlist_mult, k)
        shortlist = TopKSemanticExperienceRetriever.retrieve(
            self, state, point, shortlist_k,
        )
        if not shortlist:
            return shortlist

        # Stage 2 — Rerank: BLR P(z=1|x,e). Cache φ for use at observe time.
        cached_phi: dict[str, np.ndarray] = {}
        Phi = np.empty((len(shortlist), self._d_care), dtype=np.float64)
        for i, e in enumerate(shortlist):
            phi = self._build_features(point, e)  # 19-d via v3
            cached_phi[e.experience_id] = phi
            Phi[i] = phi
        ps = self._score_many(Phi, point.query_name)
        scored: list[tuple[float, Experience]] = [
            (float(p), e) for p, e in zip(ps, shortlist)
        ]

        # Park φ under the point key — observe_escalation reads it back so
        # we update with exactly the vector that produced the score.
        with self._feature_cache_lock:
            self._feature_cache[(point.query_name, point.doc_id)] = cached_phi

        # Pack scores back into Experience copies so MMR can run on them.
        scored.sort(key=lambda t: -t[0])
        rescored: list[Experience] = []
        for s, e in scored:
            rescored.append(Experience(
                experience_id=e.experience_id,
                source_query=e.source_query,
                source_doc_id=e.source_doc_id,
                source_doc_excerpt=e.source_doc_excerpt,
                experience_text=e.experience_text,
                applicability_signal=e.applicability_signal,
                score=float(s),
            ))

        # Stage 3 — Set selection (MMR via parent if mmr_lambda < 1).
        if self.mmr_lambda < 1.0 and len(rescored) > k:
            return self._mmr_select(rescored, k)
        return rescored[:k]

    # ------------- BLR online update --------------------------------------

    def _blr_update_one(
        self, phi: np.ndarray, y: float, w: float
    ) -> None:
        """One online Bayesian-logistic-regression (Laplace) step on (phi, y, w).

        Precision update A ← A + w·p(1-p)·φφᵀ via Sherman–Morrison on A_inv,
        then a Newton step μ ← μ - A_inv·w·(p-y)·φ. Guarded by self._blr_lock.
        """
        if w <= 0.0:
            return
        with self._blr_lock:
            eta = float(self._blr_mu @ phi)
            p = _sigmoid_clipped(eta)
            r = w * p * (1.0 - p)  # IRLS effective weight, > 0

            # Sherman–Morrison: A_inv ← A_inv − r·u uᵀ / (1 + r·φ·u)
            if r > 0.0:
                u = self._blr_A_inv @ phi
                denom = 1.0 + r * float(phi @ u)
                if denom > 1e-12:
                    self._blr_A_inv = (
                        self._blr_A_inv - (r / denom) * np.outer(u, u)
                    )
            # Newton step on μ using the *new* A_inv.
            grad = w * (p - y) * phi
            self._blr_mu = self._blr_mu - self._blr_A_inv @ grad
            self._blr_n_updates += 1

    # ------------- observe ------------------------------------------------

    def observe_escalation(  # type: ignore[override]
        self,
        exp_hits: list[Experience],
        small_pred: str | None,
        final_pred: str | None,
        query_name: str | None = None,
        small_confidence: float | None = None,
        point: DataPoint | None = None,
        sample_weight: float = 1.0,
        **_unused,
    ) -> None:
        # Update the inherited pos/neg counters first (they drive several v3
        # features), but skip FeatureLinUCB's LinUCB regression update by
        # calling the TopKSemanticOnline hook directly instead of super().
        from .topk_semantic_online import TopKSemanticOnlineExperienceRetriever
        TopKSemanticOnlineExperienceRetriever.observe_escalation(
            self, exp_hits, small_pred, final_pred,
            query_name=query_name, small_confidence=small_confidence,
        )
        # Doc/query history counters (features 15-18) update on every
        # escalation observation, even when no experiences were retrieved.
        self._observe_history(point, small_pred, final_pred)

        if not exp_hits:
            return
        if final_pred is None or final_pred == "UNKNOWN":
            return
        if point is None:
            return

        z = 1.0 if (small_pred is not None and small_pred == final_pred) else 0.0
        k = len(exp_hits)

        # Pull the cached φ from retrieve time so credit decomposition uses
        # the same vectors that produced the ranking (a fresh build would
        # drift — helpfulness features were just bumped above).
        key = (point.query_name, point.doc_id)
        with self._feature_cache_lock:
            phi_map = self._feature_cache.pop(key, None) or {}

        # Pre-update predicted helpfulness per experience (current posterior,
        # before this observation lands).
        scored: list[tuple[Experience, np.ndarray, float]] = []
        for e in exp_hits:
            phi = phi_map.get(e.experience_id)
            if phi is None:
                phi = self._build_features(point, e)
            p, _ = self._score_one(phi, point.query_name)
            scored.append((e, phi, p))

        # Marginal credit via softmax over pre-update scores:
        # w_i = k * softmax_i, so the weights sum to k.
        if k == 1:
            weights = [1.0]
        else:
            ps = np.array([t[2] for t in scored], dtype=np.float64)
            # Softmax in log-odds space — equivalent to softmax(eta) up to
            # a constant (and avoids numerical issues when all p ≈ 0.5).
            etas = np.log(np.clip(ps, 1e-6, 1.0 - 1e-6))
            etas = etas - np.log(np.clip(1.0 - ps, 1e-6, 1.0 - 1e-6))
            etas = etas - etas.max()
            ws_soft = np.exp(etas)
            ws_soft = ws_soft / ws_soft.sum()
            weights = (ws_soft * k).tolist()

        # Apply the BLR update per experience. `sample_weight` scales the
        # credit weights (inverse-propensity correction during offline
        # pretraining; 1.0 online).
        is_pos = (z == 1.0)
        for (_e, phi, _p_pre), w in zip(scored, weights):
            self._blr_update_one(phi, z, float(w) * float(sample_weight))
            with self._blr_lock:
                self._credit_sum += float(w)
                self._credit_n += 1

        with self._blr_lock:
            self._care_stats["n_observations"] += 1
            if is_pos:
                self._care_stats["n_positives"] += 1
            else:
                self._care_stats["n_negatives"] += 1
            if self._credit_n > 0:
                self._care_stats["avg_credit_weight"] = round(
                    self._credit_sum / self._credit_n, 4,
                )

    # ------------- router signals -----------------------------------------

    # Override the inherited ROUTER_SIGNAL_NAMES so introspection / config
    # logging shows the right thing for CARE.
    ROUTER_SIGNAL_NAMES: tuple[str, ...] = (
        "mean_p_calib",          # mean σ(μ·φ_e) across selected k
        "max_p_calib",           # best single-experience helpfulness
        "min_p_calib",           # worst selected experience
        "std_p_calib",           # disagreement across selected experiences
        "top1_minus_topk_p",     # rank-1 vs rank-k margin in calibrated p
        "mean_blr_var",          # mean BLR posterior variance φ^T Σ φ
        "max_blr_var",           # worst-case posterior variance
        "n_blr_updates_norm",    # log(1+n_updates)/log(1001), clamped to [0,1]
        "frac_same_query",       # share of selected from same source query
    )

    def router_signals(  # type: ignore[override]
        self,
        point: DataPoint,
        exp_hits: list[Experience],
    ) -> np.ndarray:
        """CARE-specific routing signals: a 9-d vector exposing the BLR
        head's calibrated outputs. First five slots are aggregates over
        {p_e : e ∈ S}; next two are posterior-variance summaries; last two
        surface model state (maturity, pool composition).
        """
        n = len(exp_hits) if exp_hits else 0
        out = np.zeros(len(self.ROUTER_SIGNAL_NAMES), dtype=np.float64)

        # Re-score each experience under the current posterior, using cached
        # φ from retrieve time when present (consistent with the ranking).
        ps: list[float] = []
        vars_: list[float] = []
        if n > 0:
            with self._feature_cache_lock:
                phi_map = (
                    self._feature_cache.get((point.query_name, point.doc_id))
                    or {}
                )
            for e in exp_hits:
                phi = phi_map.get(e.experience_id)
                if phi is None:
                    phi = self._build_features(point, e)
                p, var = self._score_one(phi, point.query_name)
                ps.append(p)
                vars_.append(var)

        if ps:
            arr_p = np.asarray(ps, dtype=np.float64)
            arr_v = np.asarray(vars_, dtype=np.float64)
            out[0] = float(arr_p.mean())
            out[1] = float(arr_p.max())
            out[2] = float(arr_p.min())
            out[3] = float(arr_p.std()) if n > 1 else 0.0
            out[4] = float(arr_p[0] - arr_p[-1]) if n > 1 else 0.0
            out[5] = float(arr_v.mean())
            out[6] = float(arr_v.max())
        # else: leave all five aggregates and both variance summaries at 0.

        # Model maturity, clamped to [0, 1].
        with self._blr_lock:
            n_upd = int(self._blr_n_updates)
        # log(1+n)/log(1001) — saturates around n≈1000 escalations.
        out[7] = min(1.0, math.log(1.0 + n_upd) / math.log(1001.0))

        # Same-query share of the selected set.
        if n > 0:
            out[8] = float(
                sum(1 for e in exp_hits if e.source_query == point.query_name) / n
            )
        return out

    # ------------- set-level feature export (for joint routers) ----------

    def set_features_for(
        self,
        point: DataPoint,
        exp_hits: list[Experience],
    ) -> dict:
        """Expose per-experience φ + calibrated probability for the
        selected set. Consumed by routers that want to build set-level
        features (mean/max/min/std of φ, plus calibrated p) for a
        joint head.

        Returns a dict with keys:
            phi          — np.ndarray (k, d)    per-experience features
            p_calib      — np.ndarray (k,)      σ(μ·φ) per experience
            blr_var      — np.ndarray (k,)      φ Σ φ posterior variance
            mu           — np.ndarray (d,)      current BLR posterior mean
            n_blr_updates — int

        Pulls cached φ from retrieve time when available so routing-time
        features are consistent with what drove the ranking.
        """
        n = len(exp_hits) if exp_hits else 0
        if n == 0:
            return {
                "phi": np.zeros((0, self._d_care), dtype=np.float64),
                "p_calib": np.zeros((0,), dtype=np.float64),
                "blr_var": np.zeros((0,), dtype=np.float64),
                "mu": self._blr_mu.copy(),
                "n_blr_updates": int(self._blr_n_updates),
            }
        with self._feature_cache_lock:
            phi_map = (
                self._feature_cache.get((point.query_name, point.doc_id))
                or {}
            )
        phis: list[np.ndarray] = []
        ps: list[float] = []
        vars_: list[float] = []
        for e in exp_hits:
            phi = phi_map.get(e.experience_id)
            if phi is None:
                phi = self._build_features(point, e)
            p, var = self._score_one(phi, point.query_name)
            phis.append(phi)
            ps.append(p)
            vars_.append(var)
        with self._blr_lock:
            mu = self._blr_mu.copy()
            n_upd = int(self._blr_n_updates)
        return {
            "phi": np.asarray(phis, dtype=np.float64),
            "p_calib": np.asarray(ps, dtype=np.float64),
            "blr_var": np.asarray(vars_, dtype=np.float64),
            "mu": mu,
            "n_blr_updates": n_upd,
        }

    # ------------- diagnostics --------------------------------------------

    def load_offline_snapshot(self, path: str) -> None:
        """Warm-start the BLR head from an offline-pretrained snapshot.

        The snapshot is the JSON file written by
        ``pipeline.trainer.retriever.offline_care.train_care_offline``. We
        copy the global (μ, A_inv) and the observation count so the
        runtime retriever takes the trained scoring path immediately.
        """
        # Local import to avoid circular dep at module load.
        from pipeline.trainer.retriever.offline_care import load_snapshot

        snap = load_snapshot(path)
        mu = np.asarray(snap.mu, dtype=np.float64)
        A_inv = np.asarray(snap.A_inv, dtype=np.float64)
        if mu.shape[0] != self._d_care:
            raise ValueError(
                f"snapshot μ dim {mu.shape[0]} != retriever {self._d_care}"
            )
        if A_inv.shape != (self._d_care, self._d_care):
            raise ValueError(
                f"snapshot A_inv shape {A_inv.shape} != "
                f"({self._d_care},{self._d_care})"
            )
        with self._blr_lock:
            self._blr_mu = mu
            self._blr_A_inv = A_inv
            self._blr_n_updates = int(snap.n_blr_updates)

    def care_snapshot(self) -> dict[str, Any]:
        feature_names = [
            "cos(q,source_q)",     # 0
            "cos(d,source_d)",     # 1
            "same_query",          # 2
            "global_helpfulness",  # 3
            "per_query_helpfulness",  # 4
            "log(1+n_obs)/log(101)",  # 5
            "cos(q,exp_text)",     # 6
            "cos(d,exp_text)",     # 7
            "bm25_src/scale",      # 8
            "stream_position",     # 9
            "log(1+n_obs_q)/log(101)",  # 10
            "is_per_query_cold",   # 11
            "global_helpfulness*cos(d,sd)",  # 12
            "bm25_exp_text/scale",  # 13
            "same_doc",            # 14
            "log(1+n_esc_d)/log(101)",  # 15
            "doc_agree_rate",      # 16
            "log(1+n_esc_q)/log(101)",  # 17
            "query_agree_rate",    # 18
        ]
        with self._blr_lock:
            mu = self._blr_mu.tolist()
            n = int(self._blr_n_updates)
            stats = dict(self._care_stats)
        return {
            "method": "CARE",
            "feature_dim": self._d_care,
            "feature_names": feature_names,
            "prior_precision": self.prior_precision,
            "mmr_lambda": self.mmr_lambda,
            "n_blr_updates": n,
            "mu": mu,
            "mu_named": dict(zip(feature_names, mu)),
            "care_stats": stats,
        }
