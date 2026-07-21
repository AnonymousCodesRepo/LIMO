"""care_pq — CARE with per-query hierarchical shrinkage.

Identical machinery to ``CAREExperienceRetriever`` (Bayesian-logistic-
regression reranker over the cosine shortlist + softmax credit
decomposition + Laplace-IRLS online update), but with two heads:

  * Global   (μ_g, A_inv_g)             — pooled across all queries
  * Per-q    (μ_q, A_inv_q) for q = query_name in the stream

At score time, for the current point's query q:

    n_q  = n_obs_q                          (BLR updates seen for q so far)
    w_q  = n_q / (n_q + blend_lambda)       (smooth shrinkage in [0, 1])

    η_used      = w_q · (μ_q · φ) + (1 − w_q) · (μ_g · φ)
    A_inv_used  = w_q ·  A_inv_q   + (1 − w_q) ·  A_inv_g
    p(z=1|x,e)  = σ(η_used)          (var = φᵀ A_inv_used φ is surfaced
                                      as an uncertainty summary)

A per-query θ_q learns a full direction in feature space, not just a global
offset. When n_q is small (cold-start for q), w_q ≈ 0 and scoring falls back
to the pooled global head; as q accumulates evidence, w_q → 1 and the
per-query head takes over. ``blend_lambda`` controls the transition speed
(default 20: at n_q=20, w_q=0.5).

Updates on ``observe_escalation`` apply the IRLS step (Sherman–Morrison on
the precision matrix + Newton step on the mean) to BOTH the global state and
the current query's state, with the same φ cached at retrieve time and the
same credit weight from the softmax decomposition.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from pipeline.common.types import DataPoint, Experience, RunState

from .care import CAREExperienceRetriever, _sigmoid, _sigmoid_clipped
from .feature_linucb_v3 import _FEATURE_DIM_V3
from .topk_semantic import TopKSemanticExperienceRetriever


class CAREPerQueryExperienceRetriever(CAREExperienceRetriever):
    """CARE + per-query θ with hierarchical shrinkage to a global θ_g.

    Drop-in replacement for ``CAREExperienceRetriever``. Same constructor
    plumbing plus one extra knob:

      blend_lambda : float
          Shrinkage strength. w_q = n_obs_q / (n_obs_q + blend_lambda).
          Default 20.0 — at 20 per-q observations the blend is 50/50.
          Smaller = trust the per-query head sooner. Larger = lean on
          the pooled global head longer.
    """

    # ------------- construction -------------------------------------------

    def __init__(
        self,
        *args,
        blend_lambda: float = 20.0,
        **kwargs,
    ):
        if blend_lambda <= 0:
            raise ValueError("blend_lambda must be > 0")
        super().__init__(*args, **kwargs)
        self.blend_lambda = float(blend_lambda)

        # Per-query BLR state. Lazy: a query gets a (μ_q, A_inv_q) pair
        # the first time it appears in the stream. Initialised to match
        # the global prior so a cold query starts from the same place
        # as the global head did at t=0.
        self._pq_mu: dict[str, np.ndarray] = {}
        self._pq_A_inv: dict[str, np.ndarray] = {}
        self._pq_n: dict[str, int] = {}
        self._pq_lock = threading.Lock()

    def _ensure_pq(self, q: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Lazily allocate per-query state. Caller must hold ``_pq_lock``."""
        if q not in self._pq_mu:
            self._pq_mu[q] = np.zeros(self._d_care, dtype=np.float64)
            self._pq_A_inv[q] = (
                np.eye(self._d_care, dtype=np.float64) / self.prior_precision
            )
            self._pq_n[q] = 0
        return self._pq_mu[q], self._pq_A_inv[q], self._pq_n[q]

    # ------------- BLR scoring (override) ----------------------------------

    def _score_one(  # type: ignore[override]
        self, phi: np.ndarray, query_name: str
    ) -> tuple[float, float]:
        """Return (P(z=1|x,e), variance of logit) under the *blended*
        global+per-query posterior."""
        with self._blr_lock:
            mu_g = self._blr_mu
            A_inv_g = self._blr_A_inv
        with self._pq_lock:
            mu_q, A_inv_q, n_q = self._ensure_pq(query_name)
            # Take copies so we can release the lock without races on
            # later in-place updates from another thread.
            mu_q = mu_q.copy()
            A_inv_q = A_inv_q.copy()

        w_q = n_q / (n_q + self.blend_lambda)
        eta_g = float(mu_g @ phi)
        eta_q = float(mu_q @ phi)
        eta = (1.0 - w_q) * eta_g + w_q * eta_q

        # Blended posterior covariance for the variance term. Linear blend
        # is conservative (overestimates variance vs the proper hierarchical
        # posterior), fine for an uncertainty summary.
        A_inv_blend = (1.0 - w_q) * A_inv_g + w_q * A_inv_q
        var = float(phi @ (A_inv_blend @ phi))
        if var < 0.0:
            var = 0.0

        p = _sigmoid(eta)
        return p, var

    def _score_many(  # type: ignore[override]
        self, Phi: np.ndarray, query_name: str
    ) -> np.ndarray:
        """Vectorized ``_score_one`` under the blended global+per-query
        posterior: one lock round trip and one matrix product per head
        for the whole candidate set."""
        with self._blr_lock:
            mu_g = self._blr_mu.copy()
        with self._pq_lock:
            mu_q, _A_inv_q, n_q = self._ensure_pq(query_name)
            mu_q = mu_q.copy()

        w_q = n_q / (n_q + self.blend_lambda)
        eta = (1.0 - w_q) * (Phi @ mu_g) + w_q * (Phi @ mu_q)
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -60.0, 60.0)))

    # ------------- BLR online update (override) ----------------------------

    def _blr_update_one(  # type: ignore[override]
        self, phi: np.ndarray, y: float, w: float
    ) -> None:
        """One IRLS step on the *global* head only. The per-query head is
        updated by ``_blr_update_one_pq`` (which also takes the query name).
        """
        super()._blr_update_one(phi, y, w)

    def _blr_update_one_pq(
        self, phi: np.ndarray, y: float, w: float, query_name: str
    ) -> None:
        """IRLS step on the *per-query* head ``(μ_q, A_inv_q)``."""
        if w <= 0.0:
            return
        with self._pq_lock:
            mu_q, A_inv_q, n_q = self._ensure_pq(query_name)
            eta = float(mu_q @ phi)
            p = _sigmoid_clipped(eta)
            r = w * p * (1.0 - p)

            if r > 0.0:
                u = A_inv_q @ phi
                denom = 1.0 + r * float(phi @ u)
                if denom > 1e-12:
                    A_inv_q = A_inv_q - (r / denom) * np.outer(u, u)
                    self._pq_A_inv[query_name] = A_inv_q

            grad = w * (p - y) * phi
            mu_q = mu_q - A_inv_q @ grad
            self._pq_mu[query_name] = mu_q
            self._pq_n[query_name] = n_q + 1

    # ------------- observe (override) -------------------------------------

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
        # Re-derive the per-experience credit weights and apply each (φ, z, w)
        # to both heads in one pass — mirrors the parent's observe_escalation
        # but can't call super() because it updates only the global head.
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
        q = point.query_name

        key = (q, point.doc_id)
        with self._feature_cache_lock:
            phi_map = self._feature_cache.pop(key, None) or {}

        scored: list[tuple[Experience, np.ndarray, float]] = []
        for e in exp_hits:
            phi = phi_map.get(e.experience_id)
            if phi is None:
                phi = self._build_features(point, e)
            p, _ = self._score_one(phi, q)
            scored.append((e, phi, p))

        # Softmax credit decomposition (same recipe as parent):
        # w_i = k * softmax_i, so the weights sum to k.
        if k == 1:
            weights = [1.0]
        else:
            ps = np.array([t[2] for t in scored], dtype=np.float64)
            etas = np.log(np.clip(ps, 1e-6, 1.0 - 1e-6))
            etas = etas - np.log(np.clip(1.0 - ps, 1e-6, 1.0 - 1e-6))
            etas = etas - etas.max()
            ws_soft = np.exp(etas)
            ws_soft = ws_soft / ws_soft.sum()
            weights = (ws_soft * k).tolist()

        # Apply IRLS to both heads.
        is_pos = (z == 1.0)
        for (_e, phi, _p_pre), w in zip(scored, weights):
            w_eff = float(w) * float(sample_weight)
            self._blr_update_one(phi, z, w_eff)              # global
            self._blr_update_one_pq(phi, z, w_eff, q)        # per-query
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

    # ------------- diagnostics --------------------------------------------

    def care_snapshot(self) -> dict[str, Any]:  # type: ignore[override]
        snap = super().care_snapshot()
        snap["method"] = "CARE-PQ"
        snap["blend_lambda"] = self.blend_lambda
        with self._pq_lock:
            per_query = {}
            for qn in self._pq_mu:
                n_q = self._pq_n[qn]
                w_q = n_q / (n_q + self.blend_lambda) if n_q else 0.0
                per_query[qn] = {
                    "n_obs_q": n_q,
                    "w_q": round(w_q, 4),
                    "mu_q": self._pq_mu[qn].tolist(),
                }
        snap["per_query"] = per_query
        return snap
