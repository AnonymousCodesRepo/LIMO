"""Paper-faithful Unsupervised Self-Pretraining (LIMO Section 6).

Implements the integrated cost-aware active-labeling loop of Section
6.2-6.3, jointly warm-starting all three online modules from a fresh
corpus with NO historical queries:

  1. the router ``g``               (LightGBM agreement-score head),
  2. the experience utility estimator ``g_theta``  (CARE-PQ BLR head), and
  3. a non-empty experience pool ``E_0``.

Three mechanisms make this the paper's method:

  * **The experience pool grows during the loop.** The pool starts EMPTY.
    Every disagreement (z = 0, small answer differs from large) is
    distilled into an experience via the SAME online discrepancy generator
    used at serving time, then added to the pool for later retrieval. This
    bootstraps E_0.

  * **Data augmentation by leaving out experiences (Section 6.2).** For each
    labeled pair whose retrieved experience set changed the small model's
    answer, the loop reruns the small model with each leave-one-out subset
    (and, gate-only, with no experiences). Every variant is compared against
    the SAME large answer, yielding extra (feature, agreement) router rows
    and extra per-experience credit for the utility estimator — with no
    extra large-model call.

  * **Joint refit each round.** After every round the router head is refit
    on the cumulative labeled + augmented buffer with inverse-propensity
    example weights (paper Section 6.2 Step 4), and that head becomes the
    next round's acquisition function. The utility estimator is updated
    continuously through the same ``observe_escalation`` credit-assignment
    update used online (Section 6.3).

Per-round flow (Section 6.2):
  Cheap scoring  -> retrieve from current pool, run M_S WITH experiences,
                    build the router feature vector, score uncertainty
                    (M_S answer entropy in round 0, router p(1-p) afterwards).
  Active select  -> sample ``b_r`` pairs without replacement, prob. prop. to
                    uncertainty (with an epsilon-uniform exploration mix).
  Label          -> one large-model call per selected pair -> agreement z.
  Augment        -> leave-out-experience variants (no extra large call).
  Seed + refit   -> distill z=0 disagreements into the pool; refit router.

Outputs (written by the driver after the loop returns):
  rollouts.jsonl     -- every labeled/augmented Rollout (34-d features).
  care_snapshot.json -- the warm-started utility estimator (mu, A_inv, ...).
  experiences.jsonl  -- the bootstrapped experience pool E_0.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pipeline.common.types import DataPoint, Experience, ProcessedRecord, RunState
from pipeline.query_synth.types import Candidate

from .active_acquisition import (
    _acquisition_distribution,
    _build_features_with_retriever,
    _fit_router_head,
    _s_entropy_from_features,
    _z_label,
)
from .types import Rollout, dump_rollouts


# Small-model caller: (predicate, doc_text, query_description, experiences)
#   -> small-feature dict (must include 'prediction', 'confidence', 'p_yes',
#   'p_no', ...). ``experiences`` is a list[Experience] (possibly empty).
SmallCallFn = Callable[[str, str, str, "list[Experience]"], dict]
# Large-model caller: (predicate, doc_text, query_description)
#   -> (prediction, raw_text).
LargeCallFn = Callable[[str, str, str], "tuple[str | None, str]"]


@dataclass
class PaperPretrainConfig:
    """Configuration for the paper-faithful self-pretraining loop."""

    n_rounds: int = 4
    # Paper §7.1: a fixed budget of M_L calls distributed EVENLY across
    # rounds — 4 × 100 labeled document–query pairs.
    budget_per_round: tuple[int, ...] = (100, 100, 100, 100)
    epsilon: float = 0.0                   # paper: pure proportional sampling (no exploration mix)
    seed: int = 0
    k_experiences: int = 8
    # Per-round cheap-scoring working set drawn UNIFORMLY from the remaining
    # unlabeled pool (paper §6.2 Step 1). None/0 = score every remaining
    # candidate each round.
    scoring_pool_size: int | None = 400
    # Leave-out-experience augmentation (Section 6.2). When False the loop
    # still seeds the pool and refits jointly, but skips the leave-one-out
    # rollouts -- a clean ablation of the augmentation.
    augment_leave_out: bool = True
    # Augmentation variants to emit. Default "loo" = each leave-one-out
    # subset (the paper's set, Section 6.2). "noexp" (drop all) is opt-in.
    # The no-experience gate run always happens (it decides whether to
    # augment at all).
    aug_variants: tuple[str, ...] = ("loo",)
    # Cap on small-model calls spent on augmentation per labeled candidate
    # (leave-one-out costs 1 gate + k calls for k experiences).
    max_aug_small_calls: int = 24
    # Concurrency for the read-only cheap-scoring pass (small-model + retrieve).
    workers: int = 4
    # Minimum labeled rows before the router head is first fit (below this the
    # acquisition stays on M_S answer entropy).
    min_for_refit: int = 100
    acq_conf_cap: float | None = None
    # ---- router head fit hyperparameters (read by _fit_router_head) --------
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 5
    reg_lambda: float = 1.0
    calibration_method: str = "auto"        # "auto"|"sigmoid"|"isotonic"|"none"
    calibration_cv: int = 3
    class_weight: str = "balanced"          # cascade z is skewed -> balance


@dataclass
class _Cand:
    """Per-candidate working state."""

    cand: Candidate
    dp: DataPoint
    labeled: bool = False
    # Filled by the per-round scoring pass (refreshed each round because the
    # pool / retriever change as the loop proceeds).
    exp_hits: list[Experience] = field(default_factory=list)
    small_feats: dict | None = None
    small_pred: str | None = None
    features: np.ndarray | None = None
    s_entropy: float = 0.0
    uncertainty: float = 0.0


def _query_name_for(predicate: str) -> str:
    """Shared query name per predicate.

    All documents carrying the same synthetic predicate form ONE query, so an
    experience distilled on one document is retrievable on the others under
    same-source-query retrieval (``restrict_to_source_query=True``);
    otherwise each (doc, predicate) is a singleton query and no experience
    transfers during pretraining.
    """
    h = hashlib.sha1(predicate.strip().encode("utf-8")).hexdigest()[:12]
    return f"synthq_{h}"


def _make_record(point: DataPoint, small_pred: str | None,
                 large_pred: str | None) -> ProcessedRecord:
    """A minimal ProcessedRecord the discrepancy generator can consume."""
    return ProcessedRecord(
        doc_id=int(point.doc_id),
        query_name=point.query_name,
        ground_truth=point.ground_truth,
        prediction=large_pred or "UNKNOWN",   # final == large (escalated)
        raw="",
        routed_to="large",
        small_prediction=small_pred,
        escalated=True,
    )


def run_paper_pretrain(
    *,
    candidates: list[Candidate],
    docs: dict[int, str],
    small_call_fn: SmallCallFn,
    large_call_fn: LargeCallFn,
    retriever: Any,
    generator: Any | None,
    output_rollouts_path: str | Path,
    config: PaperPretrainConfig | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the paper-faithful self-pretraining loop.

    ``retriever`` is a CARE / CARE-PQ instance constructed with an (ideally
    empty) experience pool; it is mutated in place -- its BLR head is the
    warm-started utility estimator and its pool is E_0 on return. The caller
    snapshots the retriever and dumps its pool after this returns.

    ``generator`` is an ``OnlineDiscrepancyGenerator`` (or None to disable
    experience seeding -- an ablation that keeps the pool empty).
    """
    cfg = config or PaperPretrainConfig()
    rng = np.random.default_rng(cfg.seed)
    out_path = Path(output_rollouts_path)
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    # Build the per-candidate working set + a shared RunState the retriever
    # reads (doc-text lookups, stream position).
    state = RunState()
    pool: list[_Cand] = []
    for cand in candidates:
        doc_text = docs.get(int(cand.doc_id))
        if doc_text is None:
            continue
        dp = DataPoint(
            doc_id=int(cand.doc_id),
            doc_name="",
            doc_text=doc_text,
            query_name=_query_name_for(cand.predicate),
            query_description=cand.predicate,
            ground_truth=cand.expected_answer or "",
        )
        state._doc_text_by_id[int(cand.doc_id)] = doc_text
        pool.append(_Cand(cand=cand, dp=dp))
    _log(f"working pool: {len(pool)} candidates")

    # Large-model answers are cached per candidate so augmentation reuses the
    # same reference y_L without paying for a second large call.
    large_cache: dict[str, str | None] = {}

    diagnostics: dict[str, Any] = {
        "n_candidates_in_pool": len(pool),
        "rounds": [],
        "n_experiences_seeded": 0,
        "n_aug_rollouts": 0,
        "n_main_rollouts": 0,
        "total_seconds": 0.0,
    }
    t_total = time.time()
    router_head: Any | None = None
    k = int(cfg.k_experiences)

    def _score_one(c: _Cand) -> None:
        """Cheap scoring: retrieve from current pool + run M_S with them."""
        try:
            exp_hits = retriever.retrieve(state, c.dp, k)
        except Exception:
            exp_hits = []
        try:
            sf = small_call_fn(c.cand.predicate, c.dp.doc_text,
                               c.dp.query_description, exp_hits)
        except Exception:
            sf = None
        c.exp_hits = exp_hits
        c.small_feats = sf
        c.small_pred = (sf or {}).get("prediction") if sf else None
        c.s_entropy = _s_entropy_from_features(sf)
        c.features = _build_features_with_retriever(c.dp, sf, exp_hits, retriever)

    for r in range(cfg.n_rounds):
        unlabeled = [c for c in pool if not c.labeled]
        if not unlabeled:
            break
        budget = (cfg.budget_per_round[r]
                  if r < len(cfg.budget_per_round)
                  else cfg.budget_per_round[-1])

        # Working set for this round: uniformly sampled subset of remaining.
        if cfg.scoring_pool_size is not None and \
                len(unlabeled) > cfg.scoring_pool_size:
            sel = rng.choice(len(unlabeled), size=cfg.scoring_pool_size,
                             replace=False)
            working = [unlabeled[int(i)] for i in sel]
        else:
            working = unlabeled
        budget = min(budget, len(working))

        # --- Cheap scoring pass (read-only; safe to parallelize). ----------
        t_score = time.time()
        if cfg.workers > 1 and len(working) > 1:
            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                list(ex.map(_score_one, working))
        else:
            for c in working:
                _score_one(c)

        if cfg.acq_conf_cap is not None:
            def _conf(c: _Cand) -> float:
                return float((c.small_feats or {}).get("confidence", 0.0))
            in_cap = [c for c in working if _conf(c) < cfg.acq_conf_cap]
            if len(in_cap) < budget:
                over = sorted(
                    (c for c in working if _conf(c) >= cfg.acq_conf_cap),
                    key=_conf)
                in_cap = in_cap + over[:budget - len(in_cap)]
            working = in_cap
            budget = min(budget, len(working))

        # Uncertainty per candidate.
        if router_head is None:
            for c in working:
                c.uncertainty = c.s_entropy
            acq_name = "s_entropy"
        else:
            X = np.stack([c.features for c in working])
            try:
                p_z1 = router_head.predict_proba(X)[:, 1]
            except Exception:
                p_z1 = np.full(len(working), 0.5)
            # Paper §6.2: uncertainty = p(1-p), maximized at p = 0.5.
            unc = p_z1 * (1.0 - p_z1)
            for c, u in zip(working, unc):
                c.uncertainty = float(u)
            acq_name = "router_p1p"

        # --- Active selection (without replacement, prob prop uncertainty). -
        scores = np.array([c.uncertainty for c in working], dtype=np.float64)
        q_dist = _acquisition_distribution(scores, cfg.epsilon)
        chosen_idx = rng.choice(len(working), size=budget, replace=False,
                                p=q_dist)
        # Inverse-propensity weights for the chosen batch (paper §6.2
        # Step 4): 1/q, mean-normalized within the batch and clipped so a
        # single low-q row can't dominate. Applied both to the utility-
        # estimator updates below and (via the stored q) to the router
        # refit.
        q_chosen = np.array([float(q_dist[int(j)]) for j in chosen_idx])
        raw_ips = 1.0 / np.clip(q_chosen, 1e-9, None)
        ips_batch = np.clip(raw_ips / max(float(raw_ips.mean()), 1e-12),
                            0.1, 10.0)
        _log(f"round {r}: acq={acq_name} working={len(working)} "
             f"budget={budget} score_t={time.time() - t_score:.1f}s "
             f"pool_size={len(getattr(retriever, "experiences", []) or [])}")

        # --- Label + augment + seed. ---------------------------------------
        round_rollouts: list[Rollout] = []
        n_l_ok = n_l_fail = 0
        n_seeded = 0
        n_aug = 0
        for bi, j in enumerate(chosen_idx):
            c = working[int(j)]
            q_sel = float(q_dist[int(j)])
            ips_w = float(ips_batch[bi])
            c.labeled = True

            # Large label (cached per candidate).
            if c.cand.cand_id in large_cache:
                y_L = large_cache[c.cand.cand_id]
            else:
                try:
                    y_L, _raw = large_call_fn(
                        c.cand.predicate, c.dp.doc_text, c.dp.query_description)
                    n_l_ok += 1
                except Exception:
                    y_L = None
                    n_l_fail += 1
                large_cache[c.cand.cand_id] = y_L

            y_S = c.small_pred
            z = _z_label(y_S, y_L)

            round_rollouts.append(_rollout(c, c.features, y_S, y_L, z,
                                           q_sel, r, acq_name, role="main"))
            diagnostics["n_main_rollouts"] += 1

            # Feed the full retrieved set to the utility estimator (online
            # credit-assignment update -- same as serving time).
            if c.exp_hits and y_L is not None and y_S is not None:
                _safe_observe(retriever, c.exp_hits, y_S, y_L, c.dp,
                              c.small_feats, sample_weight=ips_w)

            # --- Leave-out-experience augmentation (Section 6.2). ----------
            if (cfg.augment_leave_out and c.exp_hits and y_L is not None
                    and y_S is not None):
                n_aug += _augment_leave_out(
                    c, y_L, retriever, small_call_fn, round_rollouts,
                    q_sel, r, acq_name, cfg, ips_w=ips_w)

            # --- Seed the experience pool from disagreements (z == 0). -----
            if (generator is not None and z == 0 and y_L is not None
                    and y_S not in (None, "UNKNOWN")):
                exp = _seed_experience(generator, retriever, c.dp, y_S, y_L)
                if exp is not None:
                    n_seeded += 1

        dump_rollouts(out_path, round_rollouts, append=True)
        diagnostics["n_aug_rollouts"] += n_aug
        diagnostics["n_experiences_seeded"] += n_seeded

        # --- Joint refit: router head on cumulative buffer (IPS weights). ----
        all_rollouts = _load_buffer(out_path)
        labeled = [rr for rr in all_rollouts if rr.z is not None]
        fit_summary: dict[str, Any] = {
            "round": r, "acquisition": acq_name, "budget": budget,
            "n_l_ok": n_l_ok, "n_l_fail": n_l_fail,
            "n_main_rollouts_round": len(round_rollouts) - n_aug,
            "n_aug_rollouts_round": n_aug,
            "n_seeded_round": n_seeded,
            "pool_size": len(getattr(retriever, "experiences", []) or []),
            "n_labeled_total": len(labeled),
        }
        if len(labeled) >= cfg.min_for_refit:
            X = np.stack([np.asarray(rr.features, dtype=np.float64)
                          for rr in labeled])
            y = np.array([int(rr.z) for rr in labeled])
            # Inverse-propensity weights (paper Section 6.2 Step 4): 1/q,
            # mean-normalized and clipped so a single low-q row can't
            # dominate the fit.
            raw_w = np.array([1.0 / max(float(rr.q), 1e-9) for rr in labeled],
                             dtype=np.float64)
            w = np.clip(raw_w / max(float(raw_w.mean()), 1e-12), 0.1, 10.0)
            new_head = _fit_router_head(X, y, w, cfg=cfg)
            if new_head is not None:
                router_head = new_head
                fit_summary["fit"] = "ok"
                fit_summary["pos_frac"] = float((y == 1).mean())
            else:
                fit_summary["fit"] = "skipped (single class or error)"
        else:
            fit_summary["fit"] = "below_min_for_refit"
        diagnostics["rounds"].append(fit_summary)
        _log(f"round {r}: labeled_total={len(labeled)} seeded={n_seeded} "
             f"aug={n_aug} fit={fit_summary.get('fit')}")

    diagnostics["total_seconds"] = round(time.time() - t_total, 1)
    diagnostics["router_head_trained"] = router_head is not None
    diagnostics["final_pool_size"] = len(
        getattr(retriever, "experiences", []) or [])
    return diagnostics


# ── internals ────────────────────────────────────────────────────────────


def _rollout(c: _Cand, features: np.ndarray, y_S: str | None,
             y_L: str | None, z: int | None, q: float, r: int,
             acq: str, *, role: str) -> Rollout:
    return Rollout(
        cand_id=c.cand.cand_id,
        predicate=c.cand.predicate,
        doc_id=int(c.cand.doc_id),
        expected_answer=c.cand.expected_answer,
        features=[float(x) for x in np.asarray(features).ravel()],
        small_prediction=y_S,
        small_confidence=(float((c.small_feats or {}).get("confidence", 0.0))
                          if c.small_feats else None),
        s_entropy=float(c.s_entropy),
        large_prediction=y_L,
        z=z,
        q=float(q),
        round_idx=int(r),
        acquisition=acq,
        meta={"role": role, "n_exp": len(c.exp_hits)},
    )


def _safe_observe(retriever: Any, exp_hits: list[Experience],
                  small_pred: str, large_pred: str, point: DataPoint,
                  small_feats: dict | None,
                  sample_weight: float = 1.0) -> None:
    try:
        retriever.observe_escalation(
            exp_hits, small_pred, large_pred,
            query_name=point.query_name,
            small_confidence=(float((small_feats or {}).get("confidence", 0.0))
                              if small_feats else None),
            point=point,
            sample_weight=float(sample_weight),
        )
    except Exception:
        pass


def _augment_leave_out(
    c: _Cand, y_L: str, retriever: Any, small_call_fn: SmallCallFn,
    round_rollouts: list[Rollout], q_sel: float, r: int, acq: str,
    cfg: PaperPretrainConfig, ips_w: float = 1.0,
) -> int:
    """Run the leave-out-experience counterfactuals for one labeled pair.

    Gating (Section 6.2): first run M_S with NO experiences. If that matches
    the with-experience answer the retrieved set didn't change the decision,
    so no augmentation is produced. Otherwise emit each leave-one-out
    subset (and, opt-in, the no-experience variant) -- each compared against
    the same y_L, and each replayed through the utility estimator's
    credit-assignment update.

    Returns the number of augmented rollouts produced.
    """
    exp_hits = c.exp_hits
    k = len(exp_hits)
    pred = c.small_pred

    def _ms(subset: list[Experience]) -> tuple[str | None, dict | None, np.ndarray]:
        # Small-model call + feature build. Read-only on retriever state
        # (set_features_for reads mu/A_inv under locks) so it is safe to run
        # concurrently; the stateful observe_escalation updates are applied
        # sequentially by the caller afterward.
        try:
            sf = small_call_fn(c.cand.predicate, c.dp.doc_text,
                               c.dp.query_description, subset)
        except Exception:
            sf = None
        feats = _build_features_with_retriever(c.dp, sf, subset, retriever)
        return (sf or {}).get("prediction") if sf else None, sf, feats

    # The GATE: run M_S with NO experiences. If it matches the with-experience
    # answer the retrieved set didn't change the decision -> no augmentation.
    # Always run for the decision; only emitted as a rollout when the caller
    # asks for the "noexp" variant.
    y_S0, sf0, feats0 = _ms([])
    if y_S0 == pred:
        return 0

    # Build the requested augmentation variants. Default (cfg.aug_variants =
    # ("loo",)) emits each leave-one-out subset -- the paper's set.
    # "noexp" (drop all) is opt-in.
    variants: list[tuple[str, list[Experience]]] = []
    if "loo" in cfg.aug_variants and k > 1:
        # Budget guard: 1 (gate) + k loo calls.
        if (1 + k) <= cfg.max_aug_small_calls:
            variants += [
                ("aug_loo", [e for jj, e in enumerate(exp_hits) if jj != i])
                for i in range(k)
            ]

    if cfg.workers > 1 and len(variants) > 1:
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            results = list(ex.map(lambda v: _ms(v[1]), variants))
    else:
        results = [_ms(v[1]) for v in variants]

    n_aug = 0
    if "noexp" in cfg.aug_variants:
        z0 = _z_label(y_S0, y_L)
        c0 = _Cand(cand=c.cand, dp=c.dp, small_feats=sf0, s_entropy=c.s_entropy,
                   exp_hits=[])
        round_rollouts.append(_rollout(c0, feats0, y_S0, y_L, z0, q_sel, r, acq,
                                       role="aug_noexp"))
        n_aug += 1

    for (role, subset), (y_v, sf_v, feats_v) in zip(variants, results):
        z_v = _z_label(y_v, y_L)
        cv = _Cand(cand=c.cand, dp=c.dp, small_feats=sf_v,
                   s_entropy=c.s_entropy, exp_hits=subset)
        round_rollouts.append(_rollout(cv, feats_v, y_v, y_L, z_v, q_sel, r,
                                       acq, role=role))
        n_aug += 1
        # Replay the variant through the utility estimator's credit update.
        if y_v is not None and subset:
            _safe_observe(retriever, subset, y_v, y_L, c.dp, sf_v,
                          sample_weight=ips_w)
    return n_aug


def _seed_experience(generator: Any, retriever: Any, point: DataPoint,
                     small_pred: str, large_pred: str) -> Experience | None:
    """Distill one disagreement into an experience and add it to the pool."""
    record = _make_record(point, small_pred, large_pred)
    if not generator.should_generate(point, record):
        return None
    try:
        exp = generator.generate(point, record)
    except Exception:
        exp = None
    if exp is None:
        return None
    try:
        retriever.add(exp)
    except Exception:
        return None
    return exp


def _load_buffer(path: Path) -> list[Rollout]:
    from .types import load_rollouts
    try:
        return load_rollouts(path)
    except Exception:
        return []
