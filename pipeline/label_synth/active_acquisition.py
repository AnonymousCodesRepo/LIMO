"""Method C — cost-aware active acquisition with IPS reweighting.

Two-phase budget-aware label collection:

  Phase 0 (cheap, full pool): run the small LLM on every candidate. Compute
  the 3-way prediction features and the binary entropy of (p_yes, p_no);
  build the router's feature vector with CARE blocks zeroed (the
  pretraining run does not retrieve experiences). Phase 0 is paid even
  for candidates that are never selected for large-LLM evaluation.

  Phase 1+ (expensive, selective): a fixed number of acquisition rounds.
  Round 0 acquires by ``s_entropy`` (no router exists yet). Subsequent
  rounds acquire by the in-training router's uncertainty score
  ``p(1-p)`` where ``p = P_router(z=1 | features)`` (paper Section 6.2).
  Both are mixed with an ε-fraction of uniform sampling to keep
  coverage of the easy regime.

The selection probability ``q_i`` is recorded with each rollout so the
offline trainer can apply ``sample_weight = 1 / q_i`` and recover an
unbiased estimator of the loss over the original candidate pool.

Refit cadence: after each acquisition round, the router head is refit on
all labeled rollouts so far (with IPS weights). This becomes the
acquisition function for the next round.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.common.types import DataPoint, RunState
from pipeline.query_synth.types import Candidate
from pipeline.router._point_features import _featurize as _point_featurize

from .types import Rollout, dump_rollouts


# Matches LightGBMRouter.use_care_features=True: 14 point features
# (small-model output + length + doc/query history signals) +
# mean-pooled CARE phi + mean estimated utility. CARE blocks are
# zero during offline pretraining (no retriever wired in this path);
# trees trained with all-zero CARE columns ignore them.
from pipeline.experience_retriever.feature_linucb_v3 import _FEATURE_DIM_V3 as _PHI_DIM
from pipeline.router._point_features import POINT_FEATURE_DIM

_FEATURE_DIM = POINT_FEATURE_DIM + _PHI_DIM + 1   # 34 with the 19-d CARE phi


def _history_from(retriever: Any | None, point: DataPoint):
    """Doc/query history signals from the retriever's counters, or None
    (→ neutral defaults) when unavailable."""
    if retriever is None or not hasattr(retriever, "history_stats"):
        return None
    try:
        return retriever.history_stats(point)
    except Exception:
        return None


def _build_features(point: DataPoint, small_feats: dict | None) -> np.ndarray:
    pf = _point_featurize(point, small_feats)
    return np.concatenate([
        pf,
        np.zeros(_PHI_DIM + 1, dtype=np.float64),
    ])


def _build_features_with_retriever(
    point: DataPoint, small_feats: dict | None,
    exp_hits: list | None, retriever: Any | None,
) -> np.ndarray:
    """Full feature vector with the CARE block populated from the
    retriever's current state. Used in joint-pretraining mode.

    Same shape as ``_build_features``; only the CARE block (everything
    after the point features) is non-zero, and only when a retriever +
    experience hits are present.
    """
    pf = _point_featurize(
        point, small_feats, history=_history_from(retriever, point)
    )
    zero_phi_block = np.zeros(_PHI_DIM, dtype=np.float64)
    zero_pcal_block = np.zeros(1, dtype=np.float64)
    if retriever is None or not exp_hits or not hasattr(retriever, "set_features_for"):
        return np.concatenate([pf, zero_phi_block, zero_pcal_block])
    try:
        sf = retriever.set_features_for(point, exp_hits)
    except Exception:
        return np.concatenate([pf, zero_phi_block, zero_pcal_block])
    if not isinstance(sf, dict):
        return np.concatenate([pf, zero_phi_block, zero_pcal_block])
    phi = sf.get("phi")
    p_cal = sf.get("p_calib")
    if phi is None or getattr(phi, "size", 0) == 0:
        return np.concatenate([pf, zero_phi_block, zero_pcal_block])
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    if phi.shape[1] != _PHI_DIM:
        return np.concatenate([pf, zero_phi_block, zero_pcal_block])
    phi_block = phi.mean(axis=0)
    if p_cal is not None and getattr(p_cal, "size", 0) > 0:
        p_cal = np.asarray(p_cal, dtype=np.float64).ravel()
        pcal_block = np.array([float(p_cal.mean())], dtype=np.float64)
    else:
        pcal_block = zero_pcal_block
    return np.concatenate([pf, phi_block, pcal_block])




def _s_entropy_from_features(small_feats: dict | None) -> float:
    """Binary entropy over (p_yes, p_no) from the small-LLM 2-way features.

    Falls back to log(2) (max entropy) when the small call failed or the
    extractor couldn't produce well-formed marginals.
    """
    if not isinstance(small_feats, dict):
        return math.log(2.0)
    p_yes = float(small_feats.get("p_yes", 0.5))
    p_no = float(small_feats.get("p_no", 0.5))
    s = p_yes + p_no
    if s <= 0:
        return math.log(2.0)
    p_yes /= s
    p_no /= s
    eps = 1e-9
    p_yes = min(max(p_yes, eps), 1.0 - eps)
    p_no = min(max(p_no, eps), 1.0 - eps)
    return float(-(p_yes * math.log(p_yes) + p_no * math.log(p_no)))


@dataclass
class ActiveAcquisitionConfig:
    n_rounds: int = 4
    budget_per_round: tuple[int, ...] = (100, 100, 100, 100)
    epsilon: float = 0.15                  # uniform-mix fraction
    seed: int = 0
    # Minimum labeled rollouts before a refit; below this we keep the
    # round-0 acquisition (s_entropy).
    min_for_refit: int = 100
    # Router-fit hyperparameters (match lightgbm_router.py defaults so the
    # pretrained head is ingestible by the eval router).
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 5
    reg_lambda: float = 1.0
    calibration_method: str = "auto"        # "auto" | "sigmoid" | "isotonic" | "none"
    calibration_cv: int = 3
    class_weight: str = "none"              # "none" | "balanced"


@dataclass
class _PoolEntry:
    cand: Candidate
    small_feats: dict | None
    s_entropy: float
    features: np.ndarray
    labeled: bool = False
    q: float | None = None
    round_idx: int | None = None
    acquisition: str | None = None
    L_pred: str | None = None
    z: int | None = None


def _z_label(small_pred: str | None, large_pred: str | None) -> int | None:
    """Silver label: 1 iff the two models gave the same definitive answer.

    UNKNOWN / Unsure / None → z = 0 (escalate when the small model is
    uncertain), matching lightgbm_router.py's convention.
    """
    if small_pred is None or large_pred is None:
        return None
    if small_pred in ("UNKNOWN", "Unsure") or large_pred in ("UNKNOWN", "Unsure"):
        return 0
    return 1 if small_pred == large_pred else 0


def _acquisition_distribution(
    scores: np.ndarray, epsilon: float
) -> np.ndarray:
    """Convert raw acquisition scores to a sampling probability vector.

    Pipeline: clip → ε-mix with uniform → normalize. The uniform mix uses
    each candidate's own slot, so it's a true "with prob ε, sample
    uniformly" mixture. Returns a probability vector of length len(scores).
    """
    s = np.asarray(scores, dtype=np.float64)
    s = np.clip(s, 0.0, None)
    if not np.any(s > 0):
        s = np.ones_like(s)
    s = s / s.sum()
    if epsilon > 0:
        uni = np.full_like(s, 1.0 / len(s))
        s = (1.0 - epsilon) * s + epsilon * uni
    s = s / s.sum()
    return s


def _fit_router_head(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, *,
    cfg: ActiveAcquisitionConfig,
) -> Any | None:
    """Fit a calibrated GBDT classifier on labeled rollouts so far.

    Mirrors lightgbm_router.py's `_fit` (sklearn HGB backend) so the
    pretrained checkpoint is plug-compatible with the online router.
    Returns None when the labeled buffer can't be split into both classes.
    """
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier

    cw = "balanced" if cfg.class_weight == "balanced" else None
    base = HistGradientBoostingClassifier(
        max_iter=int(cfg.n_estimators),
        learning_rate=float(cfg.learning_rate),
        max_leaf_nodes=int(cfg.num_leaves),
        min_samples_leaf=int(cfg.min_child_samples),
        l2_regularization=float(cfg.reg_lambda),
        class_weight=cw,
    )
    if cfg.calibration_method == "auto":
        method = "sigmoid" if len(y) < 200 else "isotonic"
    elif cfg.calibration_method == "none":
        method = None
    else:
        method = cfg.calibration_method

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    cv = max(2, min(int(cfg.calibration_cv), n_pos, n_neg))
    if method is None:
        try:
            base.fit(X, y, sample_weight=w)
        except Exception:
            base.fit(X, y)
        return base
    try:
        m = CalibratedClassifierCV(base, method=method, cv=cv)
        # CalibratedClassifierCV.fit accepts sample_weight via the underlying
        # estimator if it supports it; HGB does.
        m.fit(X, y, sample_weight=w)
        return m
    except Exception:
        try:
            base.fit(X, y, sample_weight=w)
            return base
        except Exception:
            return None


def run_active_acquisition(
    *,
    candidates: list[Candidate],
    docs: dict[int, str],
    query_descriptions: dict[str, str] | None,
    small_call_fn: Callable[[str, str, str], dict],
    large_call_fn: Callable[[str, str, str], tuple[str, str]],
    output_rollouts_path: str | Path,
    config: ActiveAcquisitionConfig | None = None,
    progress_cb: Callable[[str], None] | None = None,
    # Joint-pretraining mode: when provided, the retriever (a) populates
    # the CARE block of each candidate's feature vector and (b) receives
    # observe_escalation() per labeled rollout so its BLR head co-trains
    # with the router. The caller snapshots the retriever afterward.
    retriever: Any | None = None,
    k_experiences: int = 8,
) -> dict[str, Any]:
    """Run the iterative active-acquisition labeling loop.

    Parameters
    ----------
    candidates              : output of a query_synth strategy.
    docs                    : {doc_id: doc_text} for all referenced docs.
    query_descriptions      : optional {predicate: description}; falls back
                              to the predicate text when a predicate is unkeyed.
    small_call_fn           : callable(predicate, doc_text, query_description)
                              -> small_feats dict (must include 'prediction',
                              'confidence', 'p_yes', 'p_no', ...).
    large_call_fn           : callable(predicate, doc_text, query_description)
                              -> (prediction, raw_text).
    output_rollouts_path    : JSONL; rollouts are appended per round (crash-safe).
    config                  : ActiveAcquisitionConfig.
    progress_cb             : optional per-phase/round progress callback.

    Returns
    -------
    dict — diagnostics: per-round budgets, fit summaries, total wallclock.
    """
    cfg = config or ActiveAcquisitionConfig()
    rng = np.random.default_rng(cfg.seed)
    out_path = Path(output_rollouts_path)
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # --- Phase 0: cheap S-pass on the full candidate pool. -----------------
    t_phase0 = time.time()
    state = RunState()
    pool: list[_PoolEntry] = []
    cand_exp_hits: dict[str, list] = {}  # cand_id -> retrieved experiences (joint mode)
    cand_datapoint: dict[str, DataPoint] = {}  # cand_id -> DataPoint
    joint = retriever is not None
    for i, cand in enumerate(candidates):
        doc_text = docs.get(cand.doc_id)
        if doc_text is None:
            continue
        qdesc = (query_descriptions or {}).get(cand.predicate, cand.predicate)
        try:
            sf = small_call_fn(cand.predicate, doc_text, qdesc)
        except Exception:
            sf = None
        s_ent = _s_entropy_from_features(sf)
        dp = DataPoint(
            doc_id=int(cand.doc_id),
            doc_name="",
            doc_text=doc_text,
            query_name=f"synthetic_{cand.cand_id}",
            query_description=qdesc,
            ground_truth="",
        )
        cand_datapoint[cand.cand_id] = dp
        # Joint mode: retrieve experiences AND build the CARE block.
        if joint:
            try:
                exp_hits = retriever.retrieve(state, dp, k_experiences)
            except Exception:
                exp_hits = []
            cand_exp_hits[cand.cand_id] = exp_hits
            feats = _build_features_with_retriever(dp, sf, exp_hits, retriever)
            # Update RunState so subsequent retrievals see this doc.
            try:
                state._doc_text_by_id[int(cand.doc_id)] = doc_text  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            feats = _build_features(dp, sf)
        pool.append(_PoolEntry(
            cand=cand, small_feats=sf, s_entropy=s_ent, features=feats,
        ))
        if (i + 1) % 200 == 0:
            _log(f"phase0: small-pass {i + 1} / {len(candidates)}"
                 f"{' [joint]' if joint else ''}")
    _log(f"phase0 done: pool={len(pool)} t={time.time() - t_phase0:.1f}s"
         f"{' joint' if joint else ''}")

    diagnostics: dict[str, Any] = {
        "n_candidates_in_pool": len(pool),
        "phase0_seconds": round(time.time() - t_phase0, 1),
        "rounds": [],
        "total_seconds": 0.0,
    }
    t_total = time.time()

    # --- Acquisition rounds. ----------------------------------------------
    router_head: Any | None = None
    for r in range(cfg.n_rounds):
        unlabeled_idx = [i for i, e in enumerate(pool) if not e.labeled]
        if not unlabeled_idx:
            break
        budget = (
            cfg.budget_per_round[r]
            if r < len(cfg.budget_per_round) else cfg.budget_per_round[-1]
        )
        budget = min(budget, len(unlabeled_idx))

        # Compute acquisition scores over the unlabeled subset.
        if router_head is None:
            scores = np.array([pool[i].s_entropy for i in unlabeled_idx],
                              dtype=np.float64)
            acq_name = "s_entropy"
        else:
            X_un = np.stack([pool[i].features for i in unlabeled_idx])
            try:
                p_z1 = router_head.predict_proba(X_un)[:, 1]
            except Exception:
                p_z1 = np.full(len(X_un), 0.5)
            # Paper §6.2: uncertainty = p(1-p), maximized at p = 0.5.
            scores = p_z1 * (1.0 - p_z1)
            acq_name = "router_p1p"

        q_pool = _acquisition_distribution(scores, cfg.epsilon)
        # Without-replacement sampling under p ∝ q_pool. The first-stage
        # selection probability q_pool[j] is the IPS weight source
        # (approximate-IPS for iterative active learning).
        chosen = rng.choice(
            len(unlabeled_idx), size=budget, replace=False, p=q_pool,
        )

        # Label the chosen candidates with the large LLM.
        round_rollouts: list[Rollout] = []
        n_l_calls_ok = 0
        n_l_calls_fail = 0
        for j in chosen:
            global_idx = unlabeled_idx[int(j)]
            entry = pool[global_idx]
            cand = entry.cand
            doc_text = docs[cand.doc_id]
            qdesc = (query_descriptions or {}).get(cand.predicate, cand.predicate)
            try:
                L_pred, _L_raw = large_call_fn(cand.predicate, doc_text, qdesc)
                n_l_calls_ok += 1
            except Exception:
                L_pred = None
                n_l_calls_fail += 1
            S_pred = (entry.small_feats or {}).get("prediction") if entry.small_feats else None
            z = _z_label(S_pred, L_pred)
            entry.labeled = True
            entry.q = float(q_pool[j])
            entry.round_idx = r
            entry.acquisition = acq_name
            entry.L_pred = L_pred
            entry.z = z

            # Joint mode: feed the retriever this rollout's outcome so its
            # BLR head co-trains. It credits individual experiences using
            # the cached φ from the Phase-0 retrieve() call.
            if joint and L_pred is not None and S_pred is not None:
                exp_hits = cand_exp_hits.get(cand.cand_id, [])
                if exp_hits:
                    try:
                        retriever.observe_escalation(
                            exp_hits, S_pred, L_pred,
                            query_name=cand_datapoint[cand.cand_id].query_name,
                            small_confidence=(
                                float((entry.small_feats or {}).get("confidence", 0.0))
                                if entry.small_feats else None
                            ),
                            point=cand_datapoint[cand.cand_id],
                        )
                    except Exception:
                        pass
            round_rollouts.append(Rollout(
                cand_id=cand.cand_id,
                predicate=cand.predicate,
                doc_id=int(cand.doc_id),
                expected_answer=cand.expected_answer,
                features=[float(x) for x in entry.features],
                small_prediction=S_pred,
                small_confidence=(
                    float((entry.small_feats or {}).get("confidence", 0.0))
                    if entry.small_feats else None
                ),
                s_entropy=float(entry.s_entropy),
                large_prediction=L_pred,
                z=z,
                q=float(q_pool[j]),
                round_idx=r,
                acquisition=acq_name,
                meta={"role": cand.meta.get("role")},
            ))
        dump_rollouts(out_path, round_rollouts, append=True)

        # Refit the router on the cumulative labeled buffer (IPS weighted).
        labeled = [e for e in pool if e.labeled and e.z is not None]
        fit_summary: dict[str, Any] = {
            "n_labeled_total": len(labeled),
            "round": r,
            "acquisition": acq_name,
            "budget": budget,
            "n_l_calls_ok": n_l_calls_ok,
            "n_l_calls_fail": n_l_calls_fail,
        }
        if len(labeled) >= cfg.min_for_refit:
            X = np.stack([e.features for e in labeled])
            y = np.array([int(e.z) for e in labeled])
            # IPS weight = (1 / N) / q ; centered to mean 1 so the absolute
            # weight scale doesn't shift HGB's effective learning rate, then
            # clipped to keep any single low-q point from dominating.
            n_pool = len(pool)
            raw_w = np.array(
                [1.0 / max((e.q or 1e-6) * n_pool, 1e-6) for e in labeled],
                dtype=np.float64,
            )
            raw_w = raw_w / max(float(raw_w.mean()), 1e-12)
            w = np.clip(raw_w, 0.1, 10.0)
            new_head = _fit_router_head(X, y, w, cfg=cfg)
            if new_head is not None:
                router_head = new_head
                fit_summary["fit"] = "ok"
                fit_summary["pos_frac"] = float((y == 1).mean())
            else:
                fit_summary["fit"] = "skipped (single class or fit error)"
        else:
            fit_summary["fit"] = "below_min_for_refit"
        diagnostics["rounds"].append(fit_summary)
        _log(
            f"round {r}: acq={acq_name} budget={budget} "
            f"l_ok={n_l_calls_ok} l_fail={n_l_calls_fail} "
            f"labeled_total={len(labeled)} "
            f"fit={fit_summary.get('fit')}"
        )

    diagnostics["total_seconds"] = round(time.time() - t_total, 1)
    diagnostics["router_head_trained"] = router_head is not None
    return diagnostics
