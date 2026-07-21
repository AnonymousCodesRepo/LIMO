"""Offline pretraining of the CARE BLR head.

Replays a labeled rollout JSONL through a CARE retriever instance so the
posterior (μ, A_inv) reflects what the BLR would have learned across the
synthetic queries. The fitted state is saved as a JSON snapshot the
runtime CARE / CARE-PQ retriever can warm-start from.

Why offline replay rather than batch-fit
----------------------------------------
CARE's update is a one-step Newton/Sherman-Morrison move under softmax
credit assignment over the retrieved set. The per-experience "label"
depends on which experiences were retrieved together (softmax
competition), which depends on the current μ — so a closed-form batch
fit would need CARE retrieval re-run at the final μ. Iterative replay is
the natural fitting procedure.

What's snapshotted
------------------
* ``mu``                         — global BLR posterior mean (19-d)
* ``A_inv``                      — global BLR posterior inverse precision (19×19)
* ``n_blr_updates``              — observation count (warm-start clock)
* ``stats``                      — diagnostics dict from CARE

The CARE base class (and CARE-PQ via inheritance) gains a
``load_offline_snapshot()`` method that copies the (μ, A_inv,
biases, n_updates) into the live retriever.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.common.types import DataPoint, RunState
from pipeline.label_synth.types import Rollout, load_rollouts


@dataclass
class CARESnapshot:
    mu: list[float]
    A_inv: list[list[float]]
    n_blr_updates: int = 0
    feature_dim: int = 19
    stats: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


def load_snapshot(path: str | Path) -> CARESnapshot:
    p = Path(path)
    with open(p) as f:
        d = json.load(f)
    if "A_inv" not in d:
        raise RuntimeError(f"snapshot at {p} missing 'A_inv'")
    return CARESnapshot(
        mu=list(d["mu"]),
        A_inv=[list(row) for row in d["A_inv"]],
        n_blr_updates=int(d.get("n_blr_updates", 0)),
        feature_dim=int(d.get("feature_dim", 19)),
        stats=dict(d.get("stats", {})),
        config=dict(d.get("config", {})),
    )


def _build_care(
    *,
    experiences_path: str | Path,
    embed_url: str,
    care_pq: bool = False,
    blend_lambda: float = 20.0,
    prior_precision: float = 1.0,
):
    """Construct a CARE / CARE-PQ retriever pre-loaded with the experience pool."""
    from pipeline import experience as experience_stage
    from pipeline.common.embeddings import EmbeddingClient
    from pipeline.experience_retriever import build as build_exp_retriever

    embed = EmbeddingClient(url=embed_url)
    experiences = experience_stage.load(str(experiences_path))
    name = "care_pq" if care_pq else "care"
    kwargs: dict[str, Any] = dict(
        experiences=experiences,
        embed_client=embed,
        prior_precision=prior_precision,
    )
    if care_pq:
        kwargs["blend_lambda"] = float(blend_lambda)
    return build_exp_retriever(name, **kwargs)


def train_care_offline(
    *,
    rollouts_path: str | Path,
    docs: dict[int, str],
    qdesc_map: dict[str, str] | None,
    experiences_path: str | Path,
    output_path: str | Path,
    embed_url: str = os.environ.get(
        "PIPELINE_EMBED_URL", "http://localhost:8200/embed"
    ),
    care_pq: bool = False,
    blend_lambda: float = 20.0,
    k_experiences: int = 8,
    prior_precision: float = 1.0,
    progress_cb=None,
) -> dict[str, Any]:
    """Replay rollouts through CARE so its BLR posterior reflects the
    synthetic-query supervision.

    Each rollout drives one update:

      1. Build a DataPoint from rollout (predicate, doc_id).
      2. retriever.retrieve(state, point, k) → exp_hits.
      3. retriever.observe_escalation(exp_hits, small_pred, large_pred,
         query_name=predicate, point=point).

    After the loop, snapshot (μ, A_inv, biases) and write to ``output_path``.
    """
    rolls = load_rollouts(rollouts_path)
    rolls = [r for r in rolls if r.z is not None and r.small_prediction is not None
             and r.large_prediction is not None]
    if not rolls:
        raise RuntimeError(f"no labeled rollouts in {rollouts_path}")

    retriever = _build_care(
        experiences_path=experiences_path, embed_url=embed_url,
        care_pq=care_pq, blend_lambda=blend_lambda,
        prior_precision=prior_precision,
    )
    state = RunState()
    t0 = time.time()
    n_obs = 0
    n_with_exp = 0
    for i, r in enumerate(rolls):
        dt = docs.get(int(r.doc_id))
        if dt is None:
            continue
        # Synthetic query_name is unique per rollout; use the predicate as
        # both name and description. Per-predicate biases won't transfer to
        # eval queries (which use canonical names) — only the global μ does.
        qname = f"synthetic_{r.cand_id}"
        qdesc = (qdesc_map or {}).get(qname, r.predicate)
        dp = DataPoint(
            doc_id=int(r.doc_id), doc_name="", doc_text=dt,
            query_name=qname, query_description=qdesc,
            ground_truth=r.expected_answer or "",
        )
        try:
            exp_hits = retriever.retrieve(state, dp, k_experiences)
        except Exception:
            exp_hits = []
        if exp_hits:
            n_with_exp += 1
        try:
            retriever.observe_escalation(
                exp_hits, r.small_prediction, r.large_prediction,
                query_name=qname,
                small_confidence=r.small_confidence,
                point=dp,
            )
            n_obs += 1
        except Exception:
            pass
        # Update RunState so subsequent retrievals see the doc text via
        # state.doc_text(). No full ProcessedRecord is needed here.
        try:
            state._doc_text_by_id[int(r.doc_id)] = dt  # type: ignore[attr-defined]
        except Exception:
            pass
        if progress_cb is not None and (i + 1) % 100 == 0:
            progress_cb(f"replay {i + 1}/{len(rolls)} obs={n_obs} with_exp={n_with_exp}")

    # Snapshot.
    mu = retriever._blr_mu.copy()
    A_inv = retriever._blr_A_inv.copy()
    n_upd = int(retriever._blr_n_updates)
    stats = dict(getattr(retriever, "_care_stats", {}))

    snap = CARESnapshot(
        mu=mu.tolist(),
        A_inv=[row.tolist() for row in A_inv],
        n_blr_updates=n_upd,
        feature_dim=int(getattr(retriever, "_d_care", 19)),
        stats=stats,
        config={
            "care_pq": bool(care_pq),
            "blend_lambda": float(blend_lambda),
            "k_experiences": int(k_experiences),
            "prior_precision": float(prior_precision),
            "experiences_path": str(experiences_path),
            "n_rollouts": len(rolls),
            "n_observations_applied": n_obs,
            "n_rollouts_with_experiences": n_with_exp,
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(snap), indent=2))
    metrics = {
        "n_rollouts": len(rolls),
        "n_observations_applied": n_obs,
        "n_rollouts_with_experiences": n_with_exp,
        "n_blr_updates_final": n_upd,
        "wall_seconds": round(time.time() - t0, 1),
    }
    return metrics
