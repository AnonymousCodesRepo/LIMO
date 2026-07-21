"""feature_linucb v3 — 19-dim shared-θ contextual bandit.

Inherits cosine retrieval + per-experience (pos, neg) helpfulness counters
from ``TopKSemanticOnlineExperienceRetriever``; on top of that, runs a
single shared θ over a 19-d hand-designed feature vector φ(p, e):

    score(p, e) = θ · φ(p, e) + α · sqrt(φ · A^{-1} · φ)

Update on each escalation observation, with reward
``r = 1 if small_pred == final_pred else 0`` and the φ cached from
retrieve time:

    A^{-1} ← A^{-1} − (A^{-1} φ)(A^{-1} φ)^T / (1 + φ A^{-1} φ)
    b      ← b + r · φ

Feature vector φ(p, e) (length 19):

   0.  cos(q, source_q)
   1.  cos(d, source_d)
   2.  same_query
   3.  global_helpfulness        (pos+α) / (pos+neg+α+β)
   4.  per_query_helpfulness     (pos_q+α) / (pos_q+neg_q+α+β)
   5.  log(1 + n_obs) / log(101)
   6.  cos(q, exp_text)
   7.  cos(d, exp_text)
   8.  BM25(point.doc, exp.source_doc) / bm25_scale       (clipped to [0, 1])
   9.  stream_position_normalized
  10.  log(1 + n_obs_q) / log(101)
  11.  is_per_query_cold        (1 if n_obs_q < 3 else 0)
  12.  global_helpfulness × cos(d, source_d)
  13.  BM25(point.doc, exp.experience_text) / bm25_scale  (clipped to [0, 1])
  14.  same_doc                 (1 if exp.source_doc_id == point.doc_id)
  15.  log(1 + n_esc_d) / log(101)   (escalations observed for this data point)
  16.  doc_agree_rate           (esc_agree_d+α) / (n_esc_d+α+β)
  17.  log(1 + n_esc_q) / log(101)   (escalations observed under this query)
  18.  query_agree_rate         (esc_agree_q+α) / (n_esc_q+α+β)

Features 15-18 are the paper's data-point / query history signals: how often
this data point (resp. this query) has been escalated so far, and how often
the small model matched the large model on those escalations. The counters
are maintained by ``_observe_history`` on every escalation observation.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter

import numpy as np

from pipeline.common.types import DataPoint, Experience, RunState

from .topk_semantic import TopKSemanticExperienceRetriever
from .topk_semantic_online import TopKSemanticOnlineExperienceRetriever


_FEATURE_DIM_V3 = 19
_PQ_COLD_THRESHOLD = 3.0
_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower()) if text else []


class FeatureLinUCBV3Retriever(TopKSemanticOnlineExperienceRetriever):
    """Shared-θ contextual bandit on 19 hand-designed features."""

    def __init__(
        self,
        *args,
        feature_alpha: float = 1.0,
        feature_ridge: float = 1.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bm25_scale: float = 20.0,
        n_obs_log_scale: float = 101.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if feature_alpha < 0:
            raise ValueError("feature_alpha must be >= 0")
        if feature_ridge <= 0:
            raise ValueError("feature_ridge must be > 0")

        self.feature_alpha = float(feature_alpha)
        self.feature_ridge = float(feature_ridge)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.bm25_scale = float(bm25_scale)
        self.n_obs_log_scale = float(n_obs_log_scale)
        self._d = _FEATURE_DIM_V3

        # Shared-θ state.
        self._A_inv = np.eye(self._d, dtype=np.float64) / self.feature_ridge
        self._b = np.zeros(self._d, dtype=np.float64)
        self._n_updates = 0
        self._theta_lock = threading.Lock()

        # Embedding caches (mpnet, 768-d, L2-normalized).
        self._q_emb_cache: dict[str, np.ndarray] = {}
        self._d_emb_cache: dict[int, np.ndarray] = {}
        self._exp_src_q_emb: dict[str, np.ndarray] = {}
        self._exp_src_d_emb: dict[str, np.ndarray] = {}
        self._query_descriptions: dict[str, str] = {}
        self._emb_cache_lock = threading.Lock()

        # BM25 corpus state.
        self._bm25_idf: dict[str, float] = {}
        self._bm25_avgdl: float = 1.0
        self._bm25_doc_tokens: dict[int, list[str]] = {}
        self._bm25_exp_tokens: dict[str, list[str]] = {}
        self._bm25_exp_text_tokens: dict[str, list[str]] = {}
        # Derived BM25 caches: per-doc distinct-token sets and per-experience
        # (Counter, doc_len) pairs, so scoring iterates the short experience
        # text rather than the full document term set.
        self._bm25_doc_token_sets: dict[int, set[str]] = {}
        self._bm25_exp_counts: dict[str, tuple[Counter, int]] = {}
        self._bm25_exp_text_counts: dict[str, tuple[Counter, int]] = {}
        self._bm25_lock = threading.Lock()

        # Static half of φ(p, e): the five embedding-cosine/same-query values
        # plus the two BM25 norms are pure functions of (point, experience) —
        # computed once per pair and reused on every later featurize (retrieve
        # / observe / router signals). Keyed (query_name, doc_id, exp_id).
        # Plain dict: CPython dict ops are atomic and a racing duplicate
        # compute writes the identical tuple, so no lock is needed.
        self._static_feat_cache: dict[tuple[str, int, str], tuple] = {}

        # Stream-position tracking.
        self._stream_position = 0
        self._total_points = 1
        self._stream_lock = threading.Lock()

        # Data-point / query history signals (features 15-18): per-doc and
        # per-query escalation counts and small==large agreement counts,
        # updated by ``_observe_history`` on every escalation observation.
        # {doc_id: [n_escalations, n_agreements]} / {query_name: [...]}.
        self._doc_hist: dict[int, list[float]] = {}
        self._query_hist: dict[str, list[float]] = {}
        self._hist_lock = threading.Lock()

        # Cache φ at retrieve time so the same φ is used at observe time
        # (avoids drift in helpfulness/stream-position features that other
        # workers may have updated between retrieve and observe). Keyed by
        # (query_name, doc_id) → {experience_id: φ}.
        self._feature_cache: dict[tuple[str, int],
                                  dict[str, np.ndarray]] = {}
        self._feature_cache_lock = threading.Lock()

    # ------------- prefetch ------------------------------------------------

    def prefetch(self, data: list[DataPoint]) -> None:
        """Batch-embed (q, doc), build BM25 IDF over the doc corpus, build
        query_name → query_description map. Idempotent."""
        # Recall-stage point vectors (grandparent's point cache) — one batch
        # call instead of an embed_one round trip per streamed point.
        TopKSemanticExperienceRetriever.prefetch(self, data)
        qs: dict[str, str] = {}
        ds: dict[int, str] = {}
        for p in data:
            qs.setdefault(p.query_name, p.query_description or "")
            ds.setdefault(p.doc_id, p.doc_text or "")
        with self._emb_cache_lock:
            self._query_descriptions.update(qs)
            new_qs = [q for q in qs if q not in self._q_emb_cache]
            new_ds = [i for i in ds if i not in self._d_emb_cache]
        if new_qs:
            vecs = self.client.embed([qs[q] for q in new_qs])
            with self._emb_cache_lock:
                for q, v in zip(new_qs, vecs):
                    self._q_emb_cache[q] = np.asarray(v, dtype=np.float32)
        if new_ds:
            vecs = self.client.embed([ds[i] for i in new_ds])
            with self._emb_cache_lock:
                for i, v in zip(new_ds, vecs):
                    self._d_emb_cache[i] = np.asarray(v, dtype=np.float32)

        # BM25 IDF over the doc corpus.
        with self._bm25_lock:
            if not self._bm25_idf:
                df: Counter[str] = Counter()
                total_len = 0
                for did, doc_text in ds.items():
                    toks = _tokenize(doc_text)
                    self._bm25_doc_tokens[did] = toks
                    total_len += len(toks)
                    for t in set(toks):
                        df[t] += 1
                N = max(1, len(ds))
                self._bm25_idf = {
                    t: math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)
                    for t, dft in df.items()
                }
                self._bm25_avgdl = total_len / N if N else 1.0

        # Stream-position denominator.
        with self._stream_lock:
            self._total_points = max(1, len(data))

        # Pre-supplied pool experiences: batch-embed source-doc excerpts and
        # any source-query texts not covered by the eval queries above, so
        # the first retrievals don't pay one embed round trip per experience.
        with self._lock:
            exps = list(self.experiences)
        if exps:
            with self._emb_cache_lock:
                new_d = [
                    e for e in exps
                    if e.experience_id not in self._exp_src_d_emb
                ]
                new_q = sorted(
                    {e.source_query for e in exps} - set(self._q_emb_cache)
                )
            if new_d:
                vecs = self.client.embed(
                    [e.source_doc_excerpt or "" for e in new_d]
                )
                with self._emb_cache_lock:
                    for e, v in zip(new_d, vecs):
                        self._exp_src_d_emb[e.experience_id] = np.asarray(
                            v, dtype=np.float32
                        )
            if new_q:
                texts = [
                    self._query_descriptions.get(q, q) or q for q in new_q
                ]
                vecs = self.client.embed(texts)
                with self._emb_cache_lock:
                    for q, v in zip(new_q, vecs):
                        self._q_emb_cache.setdefault(
                            q, np.asarray(v, dtype=np.float32)
                        )

    def add(self, exp: Experience) -> None:
        """Install a new experience, then precompute its static quantities.

        ``add()`` runs at chunk boundaries (serial install), so paying the
        embedding round trips and tokenization here keeps them off the
        per-point ``retrieve()`` path. Failures fall back to the lazy
        per-use path — the experience itself is already installed."""
        super().add(exp)
        try:
            self._exp_src_q(exp)
            self._exp_src_d(exp)
            eid = exp.experience_id
            src_toks = _tokenize(exp.source_doc_excerpt or "")
            et_toks = _tokenize(exp.experience_text or "")
            with self._bm25_lock:
                self._bm25_exp_tokens.setdefault(eid, src_toks)
                self._bm25_exp_counts.setdefault(
                    eid, (Counter(src_toks), len(src_toks))
                )
                self._bm25_exp_text_tokens.setdefault(eid, et_toks)
                self._bm25_exp_text_counts.setdefault(
                    eid, (Counter(et_toks), len(et_toks))
                )
        except Exception:
            pass

    # ------------- per-experience embedding lookups -----------------------

    def _q_emb(self, query_name: str, fallback_text: str = "") -> np.ndarray:
        with self._emb_cache_lock:
            v = self._q_emb_cache.get(query_name)
        if v is not None:
            return v
        text = self._query_descriptions.get(query_name, fallback_text) or ""
        v = np.asarray(self.client.embed_one(text), dtype=np.float32)
        with self._emb_cache_lock:
            self._q_emb_cache.setdefault(query_name, v)
            return self._q_emb_cache[query_name]

    def _d_emb(self, point: DataPoint) -> np.ndarray:
        with self._emb_cache_lock:
            v = self._d_emb_cache.get(point.doc_id)
        if v is not None:
            return v
        v = np.asarray(self.client.embed_one(point.doc_text or ""),
                       dtype=np.float32)
        with self._emb_cache_lock:
            self._d_emb_cache.setdefault(point.doc_id, v)
            return self._d_emb_cache[point.doc_id]

    def _exp_src_q(self, exp: Experience) -> np.ndarray:
        with self._emb_cache_lock:
            v = self._exp_src_q_emb.get(exp.experience_id)
        if v is not None:
            return v
        v = self._q_emb(exp.source_query, fallback_text=exp.source_query)
        with self._emb_cache_lock:
            self._exp_src_q_emb[exp.experience_id] = v
            return v

    def _exp_src_d(self, exp: Experience) -> np.ndarray:
        with self._emb_cache_lock:
            v = self._exp_src_d_emb.get(exp.experience_id)
        if v is not None:
            return v
        v = np.asarray(self.client.embed_one(exp.source_doc_excerpt or ""),
                       dtype=np.float32)
        with self._emb_cache_lock:
            self._exp_src_d_emb[exp.experience_id] = v
            return v

    def _exp_text_emb(self, exp: Experience) -> np.ndarray | None:
        # Already cached by the parent's _exp_vecs (built in _fit()).
        return self._vec_for_exp_id(exp.experience_id)

    # ------------- BM25 ---------------------------------------------------

    def _doc_token_set(self, point_doc_id: int) -> set[str]:
        """Distinct-token set of the point's document, cached per doc_id."""
        with self._bm25_lock:
            s = self._bm25_doc_token_sets.get(point_doc_id)
            if s is None:
                s = set(self._bm25_doc_tokens.get(point_doc_id) or ())
                self._bm25_doc_token_sets[point_doc_id] = s
        return s

    def _bm25_score(
        self, qt_set: set[str], counts: Counter, dl: int, avgdl: float,
        idf: dict[str, float],
    ) -> float:
        """Shared BM25 kernel. The point doc acts as the query (a term SET,
        no query-side tf), the experience text as the document. Iterates the
        short experience text's distinct terms, membership-testing against
        the doc term set.
        """
        if not qt_set or not dl:
            return 0.0
        k1, b = self.bm25_k1, self.bm25_b
        denom_add = k1 * (1.0 - b + b * dl / avgdl)
        score = 0.0
        for t, f in counts.items():
            if t not in qt_set:
                continue
            i = idf.get(t)
            if i is None:
                continue
            score += i * (f * (k1 + 1.0) / (f + denom_add))
        return score

    def _bm25(
        self, point_doc_id: int, exp_id: str, exp_source_doc_excerpt: str,
    ) -> float:
        """BM25 with point.doc as the query and exp.source_doc as the document."""
        qt_set = self._doc_token_set(point_doc_id)
        with self._bm25_lock:
            entry = self._bm25_exp_counts.get(exp_id)
            avgdl = self._bm25_avgdl
            idf = self._bm25_idf
        if entry is None:
            dt = _tokenize(exp_source_doc_excerpt)
            entry = (Counter(dt), len(dt))
            with self._bm25_lock:
                self._bm25_exp_tokens.setdefault(exp_id, dt)
                self._bm25_exp_counts.setdefault(exp_id, entry)
        return self._bm25_score(qt_set, entry[0], entry[1], avgdl, idf)

    def _bm25_exp_text(
        self, point_doc_id: int, exp_id: str, exp_text: str,
    ) -> float:
        """BM25 with point.doc as the query and exp.experience_text as the document."""
        qt_set = self._doc_token_set(point_doc_id)
        with self._bm25_lock:
            entry = self._bm25_exp_text_counts.get(exp_id)
            avgdl = self._bm25_avgdl
            idf = self._bm25_idf
        if entry is None:
            dt = _tokenize(exp_text)
            entry = (Counter(dt), len(dt))
            with self._bm25_lock:
                self._bm25_exp_text_tokens.setdefault(exp_id, dt)
                self._bm25_exp_text_counts.setdefault(exp_id, entry)
        return self._bm25_score(qt_set, entry[0], entry[1], avgdl, idf)

    # ------------- feature builder ----------------------------------------

    @staticmethod
    def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        # mpnet outputs are L2-normalized → cos = dot. Clip to [0, 1] so
        # negative cosines (rare on normalized 768-d vectors but possible)
        # don't pull the linear score around the wrong way.
        c = float(np.dot(a, b))
        if c < 0.0:
            return 0.0
        return c if c <= 1.0 else 1.0

    def _static_features(
        self, point: DataPoint, exp: Experience
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """The 8 pure-(point, experience) feature values, cached per pair:
        (cos_q_sq, cos_d_sd, same_query, cos_q_et, cos_d_et,
         bm25_src_norm, bm25_et_norm, same_doc). Embedding lookups and BM25
        run at most once per pair over the whole run."""
        skey = (point.query_name, point.doc_id, exp.experience_id)
        static = self._static_feat_cache.get(skey)
        if static is not None:
            return static

        q_emb = self._q_emb(point.query_name, point.query_description)
        d_emb = self._d_emb(point)
        sq_emb = self._exp_src_q(exp)
        sd_emb = self._exp_src_d(exp)
        et_emb = self._exp_text_emb(exp)

        same_query = 1.0 if point.query_name == exp.source_query else 0.0
        same_doc = 1.0 if exp.source_doc_id == point.doc_id else 0.0
        cos_q_sq = self._cos(q_emb, sq_emb)
        cos_d_sd = self._cos(d_emb, sd_emb)
        cos_q_et = self._cos(q_emb, et_emb)
        cos_d_et = self._cos(d_emb, et_emb)

        bm25_src_norm = self._bm25(
            point.doc_id, exp.experience_id, exp.source_doc_excerpt or "",
        ) / self.bm25_scale
        if bm25_src_norm > 1.0:
            bm25_src_norm = 1.0
        bm25_et_norm = self._bm25_exp_text(
            point.doc_id, exp.experience_id, exp.experience_text or "",
        ) / self.bm25_scale
        if bm25_et_norm > 1.0:
            bm25_et_norm = 1.0

        static = (
            cos_q_sq, cos_d_sd, same_query, cos_q_et, cos_d_et,
            bm25_src_norm, bm25_et_norm, same_doc,
        )
        self._static_feat_cache[skey] = static
        return static

    def _build_features(self, point: DataPoint, exp: Experience) -> np.ndarray:
        (
            cos_q_sq, cos_d_sd, same_query, cos_q_et, cos_d_et,
            bm25_src_norm, bm25_et_norm, same_doc,
        ) = self._static_features(point, exp)

        # Counters (dynamic — move with every escalation observation).
        with self._stats_lock:
            s = self.stats.get(exp.experience_id, {"pos": 0.0, "neg": 0.0})
            pos = float(s["pos"])
            neg = float(s["neg"])
            spq = self.stats_pq.get((exp.experience_id, point.query_name))
            if spq is not None:
                pos_q = float(spq["pos"])
                neg_q = float(spq["neg"])
            else:
                pos_q, neg_q = 0.0, 0.0

        helpfulness = (pos + self.alpha) / (pos + neg + self.alpha + self.beta)
        helpfulness_q = (pos_q + self.alpha) / (
            pos_q + neg_q + self.alpha + self.beta
        )
        n_obs = pos + neg
        n_obs_q = pos_q + neg_q

        log_obs_norm = math.log(1.0 + n_obs) / math.log(self.n_obs_log_scale)
        if log_obs_norm > 1.0:
            log_obs_norm = 1.0
        log_obs_q_norm = math.log(1.0 + n_obs_q) / math.log(self.n_obs_log_scale)
        if log_obs_q_norm > 1.0:
            log_obs_q_norm = 1.0

        with self._stream_lock:
            sp = self._stream_position / self._total_points
            if sp > 1.0:
                sp = 1.0

        is_pq_cold = 1.0 if n_obs_q < _PQ_COLD_THRESHOLD else 0.0
        help_x_dsd = helpfulness * cos_d_sd

        # Data-point / query history signals (escalation counts + agreement
        # rates, Laplace-smoothed like the helpfulness features).
        doc_esc_norm, doc_agree_rate, q_esc_norm, q_agree_rate = (
            self.history_stats(point)
        )

        return np.array([
            cos_q_sq,         # 0
            cos_d_sd,         # 1
            same_query,       # 2
            helpfulness,      # 3
            helpfulness_q,    # 4
            log_obs_norm,     # 5
            cos_q_et,         # 6
            cos_d_et,         # 7
            bm25_src_norm,    # 8
            sp,               # 9
            log_obs_q_norm,   # 10
            is_pq_cold,       # 11
            help_x_dsd,       # 12
            bm25_et_norm,     # 13
            same_doc,         # 14
            doc_esc_norm,     # 15
            doc_agree_rate,   # 16
            q_esc_norm,       # 17
            q_agree_rate,     # 18
        ], dtype=np.float64)

    # ------------- data-point / query history counters ---------------------

    def history_stats(
        self, point: DataPoint
    ) -> tuple[float, float, float, float]:
        """Data-point / query history signals for ``point``:
        (doc_esc_norm, doc_agree_rate, query_esc_norm, query_agree_rate).

        Escalation counts are log-normalized like the other count features;
        agreement rates are Laplace-smoothed (neutral 0.5 with no
        observations). Also consumed by the trained router as part of its
        point-feature block.
        """
        with self._hist_lock:
            dh = self._doc_hist.get(point.doc_id)
            n_esc_d, agree_d = (dh[0], dh[1]) if dh else (0.0, 0.0)
            qh = self._query_hist.get(point.query_name)
            n_esc_q, agree_q = (qh[0], qh[1]) if qh else (0.0, 0.0)
        doc_esc_norm = math.log(1.0 + n_esc_d) / math.log(self.n_obs_log_scale)
        if doc_esc_norm > 1.0:
            doc_esc_norm = 1.0
        doc_agree_rate = (agree_d + self.alpha) / (
            n_esc_d + self.alpha + self.beta
        )
        q_esc_norm = math.log(1.0 + n_esc_q) / math.log(self.n_obs_log_scale)
        if q_esc_norm > 1.0:
            q_esc_norm = 1.0
        q_agree_rate = (agree_q + self.alpha) / (
            n_esc_q + self.alpha + self.beta
        )
        return doc_esc_norm, doc_agree_rate, q_esc_norm, q_agree_rate

    def _observe_history(
        self,
        point: DataPoint | None,
        small_pred: str | None,
        final_pred: str | None,
    ) -> None:
        """Update the per-doc / per-query escalation history counters
        (features 15-18) from one escalation observation. Skipped when the
        large answer is unparseable (no reliable signal) or the point is
        unknown."""
        if point is None:
            return
        if final_pred is None or final_pred == "UNKNOWN":
            return
        agree = 1.0 if (small_pred is not None and small_pred == final_pred) else 0.0
        with self._hist_lock:
            dh = self._doc_hist.setdefault(point.doc_id, [0.0, 0.0])
            dh[0] += 1.0
            dh[1] += agree
            qh = self._query_hist.setdefault(point.query_name, [0.0, 0.0])
            qh[0] += 1.0
            qh[1] += agree

    # ------------- retrieve ------------------------------------------------

    def retrieve(
        self, state: RunState, point: DataPoint, k: int,
    ) -> list[Experience]:
        if k <= 0:
            return []

        # Increment stream position (this point arrived in the stream).
        with self._stream_lock:
            self._stream_position += 1

        # Cosine shortlist (grandparent's top-N by cosine).
        shortlist_k = max(k * self.shortlist_mult, k)
        shortlist = TopKSemanticExperienceRetriever.retrieve(
            self, state, point, shortlist_k,
        )
        if not shortlist:
            return shortlist

        # Score each candidate with shared-θ LinUCB.
        scored: list[tuple[float, Experience, np.ndarray]] = []
        with self._theta_lock:
            A_inv = self._A_inv
            b = self._b
            theta = A_inv @ b

        cached_phi: dict[str, np.ndarray] = {}
        for e in shortlist:
            phi = self._build_features(point, e)
            cached_phi[e.experience_id] = phi
            mean = float(theta @ phi)
            quad = float(phi @ (A_inv @ phi))
            if quad < 0.0:
                quad = 0.0
            uncertainty = float(np.sqrt(quad))
            score = mean + self.feature_alpha * uncertainty
            scored.append((score, e, phi))

        # Cache φ under the point key for use in observe_escalation.
        with self._feature_cache_lock:
            self._feature_cache[(point.query_name, point.doc_id)] = cached_phi

        scored.sort(key=lambda t: -t[0])
        out: list[Experience] = []
        for s, e, _ in scored[:k]:
            out.append(Experience(
                experience_id=e.experience_id,
                source_query=e.source_query,
                source_doc_id=e.source_doc_id,
                source_doc_excerpt=e.source_doc_excerpt,
                experience_text=e.experience_text,
                applicability_signal=e.applicability_signal,
                score=float(s),
            ))
        return out

    # ------------- observe ------------------------------------------------

    def observe_escalation(  # type: ignore[override]
        self,
        exp_hits: list[Experience],
        small_pred: str | None,
        final_pred: str | None,
        query_name: str | None = None,
        small_confidence: float | None = None,
        point: DataPoint | None = None,
        **_unused,
    ) -> None:
        # Update inherited pos/neg counters first (used for diagnostics + as
        # features themselves on subsequent retrievals).
        super().observe_escalation(
            exp_hits, small_pred, final_pred,
            query_name=query_name, small_confidence=small_confidence,
        )
        # Doc/query history counters update on EVERY escalation observation,
        # even when no experiences were retrieved.
        self._observe_history(point, small_pred, final_pred)
        if not exp_hits:
            return
        if final_pred is None or final_pred == "UNKNOWN":
            return
        if point is None:
            return

        r = 1.0 if (small_pred is not None and small_pred == final_pred) else 0.0

        # Look up cached φ from retrieve time so we update with the same
        # vector that produced the score (avoids drift in counter-based
        # features between retrieve and observe).
        key = (point.query_name, point.doc_id)
        with self._feature_cache_lock:
            phi_map = self._feature_cache.get(key, {})

        with self._theta_lock:
            for e in exp_hits:
                phi = phi_map.get(e.experience_id)
                if phi is None:
                    # Fallback: rebuild if not cached (shouldn't normally happen).
                    phi = self._build_features(point, e)
                # Sherman–Morrison update of A_inv after A += φ φ^T.
                u = self._A_inv @ phi
                denom = 1.0 + float(phi @ u)
                if denom <= 0.0:
                    continue
                self._A_inv = self._A_inv - np.outer(u, u) / denom
                self._b = self._b + r * phi
                self._n_updates += 1

        # Drop the cached φ for this point (no longer needed).
        with self._feature_cache_lock:
            self._feature_cache.pop(key, None)

    # ------------- diagnostics --------------------------------------------

    def feature_linucb_snapshot(self) -> dict:
        with self._theta_lock:
            theta = (self._A_inv @ self._b).tolist()
            n = self._n_updates
        feature_names = [
            "cos(q,source_q)",
            "cos(d,source_d)",
            "same_query",
            "global_helpfulness",
            "per_query_helpfulness",
            "log(1+n_obs)/log(101)",
            "cos(q,exp_text)",
            "cos(d,exp_text)",
            "bm25_src/scale",
            "stream_position",
            "log(1+n_obs_q)/log(101)",
            "is_per_query_cold",
            "global_helpfulness*cos(d,sd)",
            "bm25_exp_text/scale",
            "same_doc",
            "log(1+n_esc_d)/log(101)",
            "doc_agree_rate",
            "log(1+n_esc_q)/log(101)",
            "query_agree_rate",
        ]
        return {
            "alpha": self.feature_alpha,
            "ridge": self.feature_ridge,
            "feature_dim": self._d,
            "version": "v3",
            "n_updates": n,
            "theta": theta,
            "theta_named": dict(zip(feature_names, theta)),
            "feature_names": feature_names,
        }
