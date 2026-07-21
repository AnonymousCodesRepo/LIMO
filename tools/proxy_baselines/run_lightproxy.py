"""lightProxy proxy-cascade baseline, adapted to our (query, doc) cascade.

Mechanism (per query):
  1. Draw a small probe sample of docs and label them with the large LLM M_L
     (70B cache).
  2. Train a logistic-regression (LR) proxy on the probe and self-check with
     k-fold cross-validation: estimate the proxy's agreement with M_L on the
     probe. e_q = 1 - agreement is the estimated accuracy gap vs M_L.
  3. Decision (parameterised by a query-level tolerance tau, swept in the
     aggregation step): if e_q <= tau the proxy serves the WHOLE collection
     (probe docs keep their M_L labels, the rest get proxy labels); otherwise
     the query falls back to M_L over the whole corpus.

Sweeping tau across the query set changes the fraction of queries served by the
proxy, tracing the cost-accuracy curve. e_q and both outcomes (proxy vs
full-M_L) are fixed per query, so the tau sweep is a post-hoc pass.

Output: one JSON object per query (JSONL) with e_q, the probe doc ids, and the
confusion counts of BOTH outcomes so the sweep needs no re-computation.

Run on GPU server, e.g.:
  MOP_PY=python3
  PYTHONPATH=. $MOP_PY \
    tools/proxy_baselines/run_lightproxy.py --dataset hoc \
    --out runs/proxy_baselines/lightproxy_hoc.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from _common import (
    QueryData,
    confusion_from_served,
    fit_proxy_proba,
    load_dataset,
)


def _estimate_gap(X: np.ndarray, y_ml: np.ndarray, *, folds: int,
                  class_weight: str | None = "balanced") -> tuple[float, str]:
    """Estimated proxy-vs-M_L gap e_q on the probe via k-fold CV.

    Returns (e_q, note). Single-class probes can't be cross-validated; we treat
    the proxy as trivially agreeing (e_q = 0) and flag it."""
    classes, counts = np.unique(y_ml, return_counts=True)
    if len(classes) < 2:
        return 0.0, "single-class-probe"
    k = int(min(folds, counts.min()))
    if k < 2:
        return 0.0, "too-few-minority"
    lr = LogisticRegression(C=1.0, class_weight=class_weight, max_iter=1000)
    cv_pred = cross_val_predict(lr, X, y_ml, cv=k)
    agreement = float((cv_pred == y_ml).mean())
    return 1.0 - agreement, f"cv{k}"


def run_query(
    qd: QueryData,
    *,
    probe_frac: float,
    folds: int,
    rng_seed: int,
    class_weight: str | None = "balanced",
    proxy_serves_all: bool = False,
) -> dict:
    n = qd.n
    rng = np.random.default_rng(rng_seed + (hash(qd.query_name) % 100000))
    m = min(n, max(20, round(probe_frac * n)))
    probe = np.sort(rng.permutation(n)[:m])
    probe_set = set(int(i) for i in probe)

    # M_L labels on the probe (drop UNKNOWN for training / self-check).
    valid = [i for i in probe if qd.ml_int[i] >= 0]
    Xp = qd.emb[valid]
    yp = qd.ml_int[valid]
    e_q, note = _estimate_gap(Xp, yp, folds=folds, class_weight=class_weight)

    # Final serving proxy: trained on the full (valid) probe, predicts all docs.
    proxy_proba = fit_proxy_proba(Xp, yp, qd.emb, class_weight=class_weight)
    proxy_pred = (proxy_proba >= 0.5).astype(np.int64)

    # Outcome A: proxy serves the collection. With proxy_serves_all the proxy
    # labels EVERY doc (probe M_L calls used only for the self-check, discarded
    # for serving — variant A2); otherwise probe docs keep their M_L answers.
    if proxy_serves_all:
        served_a = proxy_pred.copy()
        unknown_a = np.zeros(n, dtype=bool)
    else:
        served_a = np.array(
            [int(qd.ml_int[i]) if (i in probe_set and qd.ml_int[i] >= 0)
             else int(proxy_pred[i]) for i in range(n)],
            dtype=np.int64,
        )
        unknown_a = np.array(
            [bool(i in probe_set and qd.ml_int[i] < 0) for i in range(n)]
        )
    conf_proxy = confusion_from_served(served_a, unknown_a, qd.gold)

    # Outcome B: M_L serves the whole corpus (= the 70B answer for every doc).
    served_b = np.where(qd.ml_int >= 0, qd.ml_int, 0)
    unknown_b = qd.ml_int < 0
    conf_ml = confusion_from_served(served_b, unknown_b, qd.gold)

    return {
        "query_name": qd.query_name,
        "n_docs": n,
        "m_probe": int(m),
        "e_q": e_q,
        "e_q_note": note,
        "n_pos_gold": int(qd.gold.sum()),
        "n_pos_ml": int((qd.ml_int == 1).sum()),
        "probe_doc_ids": [int(qd.doc_ids[i]) for i in probe],
        "all_doc_ids": [int(d) for d in qd.doc_ids],
        "proxy_outcome": conf_proxy,   # served when e_q <= tau
        "ml_outcome": conf_ml,          # served when e_q >  tau
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--queries", default="all")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--embed-url", default=None)
    ap.add_argument("--ml-cache", action="append", default=None,
                    help="zero-shot csv supplying the large-model (M_L) labels; "
                         "repeatable. Default = Llama-3.1-70B cache. Pass the "
                         "397B canonical csv to run with 397B as the large model.")
    ap.add_argument("--probe-frac", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--class-weight", choices=["balanced", "none"],
                    default="balanced",
                    help="LR class_weight; 'none' = sklearn default (variant A1)")
    ap.add_argument("--proxy-serves-all", action="store_true",
                    help="proxy labels every doc; probe M_L labels used only for "
                         "the self-check, not for serving (variant A2)")
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
    print(f"[lightproxy] loaded {args.dataset}: {stats}", flush=True)

    with open(out, "w") as f:
        for qi, (q, qd) in enumerate(sorted(by_q.items()), 1):
            rec = run_query(
                qd, probe_frac=args.probe_frac,
                folds=args.folds, rng_seed=args.rng_seed,
                class_weight=cw, proxy_serves_all=args.proxy_serves_all,
            )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(
                f"[lightproxy] ({qi}/{len(by_q)}) {q}: n={rec['n_docs']} "
                f"m={rec['m_probe']} e_q={rec['e_q']:.3f} ({rec['e_q_note']}) "
                f"proxy-acc={_acc(rec['proxy_outcome']):.3f} "
                f"ml-acc={_acc(rec['ml_outcome']):.3f}",
                flush=True,
            )
    print(f"[lightproxy] done {args.dataset} in {time.time() - t0:.1f}s -> {out}",
          flush=True)


def _acc(cp: dict) -> float:
    d = cp["tp"] + cp["tn"] + cp["fp"] + cp["fn"]
    return (cp["tp"] + cp["tn"]) / d if d else 0.0


if __name__ == "__main__":
    main()
