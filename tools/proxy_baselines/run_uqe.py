"""UQE proxy-cascade baseline (Dai et al., "UQE: A Query Engine for Unstructured
Databases", NeurIPS 2024), adapted to our (query, doc) binary-classification
cascade.

Mechanism (per query, over its document collection):
  1. Seed: label a small random set of docs with the large LLM M_L (70B cache);
     train the first logistic-regression (LR) proxy on those M_L labels.
  2. Active-learning rounds: the proxy scores every not-yet-labeled doc; the
     TOP-CONFIDENCE batch (confidence = max(p, 1-p)) is sent to M_L for labels,
     added to the training set, and the proxy is retrained. Repeat.
  3. When the M_L budget is spent, the remaining docs are labeled by the proxy
     (our enhancement — the paper leaves this unspecified).

Varying the M_L budget traces the cost-accuracy curve: at a budget fraction b,
the first b*N docs (in acquisition order) are served by M_L (= the 70B answer)
and the rest by the proxy snapshot at that budget.

We run each query all the way to budget = 100% once and record a checkpoint
after the seed and after every round; the aggregation script then reads any
budget off these checkpoints. Output: one JSON object per query (JSONL).

Run on GPU server, e.g.:
  MOP_PY=python3
  PYTHONPATH=. $MOP_PY \
    tools/proxy_baselines/run_uqe.py --dataset opp115 \
    --out runs/proxy_baselines/uqe_opp115.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _common import (
    QueryData,
    confusion_from_served,
    fit_proxy_proba,
    load_dataset,
)


def _pick_seed(order: list[int], ml_int: np.ndarray, seed_size: int) -> list[int]:
    """First ``seed_size`` docs in ``order``, extended until both M_L classes
    (0 and 1) appear among the non-UNKNOWN labels, so the first LR can fit."""
    sel = list(order[:seed_size])
    seen = {int(ml_int[i]) for i in sel if ml_int[i] >= 0}
    j = seed_size
    while {0, 1} - seen and j < len(order):
        i = order[j]
        sel.append(i)
        if ml_int[i] >= 0:
            seen.add(int(ml_int[i]))
        j += 1
    return sel


def _served_arrays(qd: QueryData, labeled: set[int], proxy_pred: np.ndarray):
    """Served prediction (int) and UNKNOWN flag per doc: M_L answer for labeled
    docs (may be UNKNOWN), proxy answer otherwise (never UNKNOWN)."""
    n = qd.n
    served = np.empty(n, dtype=np.int64)
    unknown = np.zeros(n, dtype=bool)
    for i in range(n):
        if i in labeled:
            if qd.ml_int[i] < 0:
                unknown[i] = True
                served[i] = 0
            else:
                served[i] = int(qd.ml_int[i])
        else:
            served[i] = int(proxy_pred[i])
    return served, unknown


def run_query(
    qd: QueryData,
    *,
    seed_frac: float,
    round_frac: float,
    rng_seed: int,
    class_weight: str | None = "balanced",
) -> dict:
    n = qd.n
    rng = np.random.default_rng(rng_seed + (hash(qd.query_name) % 100000))
    order = list(rng.permutation(n))
    seed_size = max(2, round(seed_frac * n))
    batch = max(1, round(round_frac * n))

    seed_idx = _pick_seed(order, qd.ml_int, seed_size)
    labeled: set[int] = set(seed_idx)
    acq_order: list[int] = list(seed_idx)

    checkpoints: list[dict] = []
    round_idx = 0
    while True:
        train = [i for i in labeled if qd.ml_int[i] >= 0]
        proba = fit_proxy_proba(qd.emb[train], qd.ml_int[train], qd.emb,
                                class_weight=class_weight)
        proxy_pred = (proba >= 0.5).astype(np.int64)

        served, unknown = _served_arrays(qd, labeled, proxy_pred)
        conf = confusion_from_served(served, unknown, qd.gold)
        checkpoints.append({
            "round_idx": round_idx,
            "n_ml": len(labeled),
            "frac": len(labeled) / n,
            **conf,
        })

        unlabeled = [i for i in range(n) if i not in labeled]
        if not unlabeled:
            break
        confidence = np.maximum(proba, 1.0 - proba)
        unlabeled.sort(key=lambda i: -confidence[i])   # most confident first
        for i in unlabeled[:batch]:
            labeled.add(i)
            acq_order.append(i)
        round_idx += 1

    return {
        "query_name": qd.query_name,
        "n_docs": n,
        "seed_size": len(seed_idx),
        "batch": batch,
        "n_pos_gold": int(qd.gold.sum()),
        "n_pos_ml": int((qd.ml_int == 1).sum()),
        "acq_order": [int(qd.doc_ids[i]) for i in acq_order],
        "checkpoints": checkpoints,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--queries", default="all")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--cache-dir", default=None,
                    help="dir for cached mpnet doc embeddings (.npz)")
    ap.add_argument("--embed-url", default=None)
    ap.add_argument("--ml-cache", action="append", default=None,
                    help="zero-shot csv supplying the large-model (M_L) labels; "
                         "repeatable. Default = Llama-3.1-70B cache. Pass the "
                         "397B canonical csv to run with 397B as the large model.")
    ap.add_argument("--seed-frac", type=float, default=0.2)
    ap.add_argument("--round-frac", type=float, default=0.2)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--class-weight", choices=["balanced", "none"],
                    default="balanced",
                    help="LR class_weight; 'none' = sklearn default (variant A1)")
    args = ap.parse_args()
    cw = None if args.class_weight == "none" else "balanced"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out.parent / "embeds"

    t0 = time.time()
    by_q, stats = load_dataset(
        args.dataset, queries=args.queries,
        cache_dir=cache_dir, embed_url=args.embed_url,
        ml_cache_sources=args.ml_cache,
    )
    print(f"[uqe] loaded {args.dataset}: {stats}", flush=True)

    with open(out, "w") as f:
        for qi, (q, qd) in enumerate(sorted(by_q.items()), 1):
            rec = run_query(
                qd, seed_frac=args.seed_frac,
                round_frac=args.round_frac, rng_seed=args.rng_seed,
                class_weight=cw,
            )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            last = rec["checkpoints"][-1]
            first = rec["checkpoints"][0]
            print(
                f"[uqe] ({qi}/{len(by_q)}) {q}: n={rec['n_docs']} "
                f"seed={rec['seed_size']} rounds={len(rec['checkpoints'])} "
                f"seed-acc={_acc(first):.3f} full-acc={_acc(last):.3f}",
                flush=True,
            )
    print(f"[uqe] done {args.dataset} in {time.time() - t0:.1f}s -> {out}",
          flush=True)


def _acc(cp: dict) -> float:
    d = cp["tp"] + cp["tn"] + cp["fp"] + cp["fn"]
    return (cp["tp"] + cp["tn"]) / d if d else 0.0


if __name__ == "__main__":
    main()
