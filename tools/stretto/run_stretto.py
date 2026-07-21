"""Stretto baseline -- Track B (optimizer only, no KV-cache compression).

For each (dataset, query) it fits Stretto's asymmetric two-threshold band on the
0.8B log-odds by the gradient optimizer (stretto_opt.fit_plan), against the 70B gold,
for a grid of query-level (precision=recall=q) targets. Each fitted plan is applied to
ALL documents of the query (as Stretto applies its plan to every tuple), producing one
JSON-Lines row with escalation rate + accuracy/F1 vs the human labels, mirroring
tools/scaledoc/run_scaledoc.py.

Run on GPU server:
  MOP_PYTHON=python3 \
  ./run.sh tools/stretto/run_stretto.py --datasets opp115,cuad,contract \
      --out runs/stretto
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stretto_data as sd          # noqa: E402
import stretto_opt as so           # noqa: E402

# per-token USD prices (mop/llm.py). LARGE defaults to
# Llama-3.1-70B; --large-price switches it to 397B ($0.45/$3.00) so the
# optimizer's cost-awareness matches the deployed large model.
SMALL_IN, SMALL_OUT = 0.01e-6, 0.05e-6
LARGE_IN, LARGE_OUT = 0.52e-6, 0.75e-6
# average 70B input tokens per (query, doc) pair, per panel (scaledoc constants)
AVG_IN = {"cuad": 2947.5, "contract": 2947.5, "opp115": 179.2, "hoc": 422.4}

DEFAULT_TARGETS = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.99"


def _binary_metrics(pred, human):
    tp = int(np.sum((pred == 1) & (human == 1)))
    fp = int(np.sum((pred == 1) & (human == 0)))
    fn = int(np.sum((pred == 0) & (human == 1)))
    tn = int(np.sum((pred == 0) & (human == 0)))
    n = len(human)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, acc=acc, f1=f1)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    targets = [float(x) for x in args.targets.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for ds in datasets:
        avg_in = AVG_IN.get(ds, 500.0)
        cost_small = avg_in * args.small_in + args.small_out
        cost_large = avg_in * args.large_in + args.large_out
        fout_path = os.path.join(args.out, f"stretto_{ds}.jsonl")
        try:
            queries = sd.load_dataset(ds, gold_cache=args.gold_cache,
                                      small_conf=args.small_conf)
        except FileNotFoundError as e:
            print(f"[{ds}] SKIP: {e}")
            continue
        print(f"[{ds}] {len(queries)} queries; cost_small={cost_small:.2e} cost_large={cost_large:.2e}")
        with open(fout_path, "w") as fout:
            for q in queries:
                n = len(q.gold)
                tr = sd.stratified_sample(q.gold, args.train_frac, args.seed + q.qid)
                lo_tr, gold_tr = q.log_odds[tr], q.gold[tr]
                for tq in targets:
                    plan = so.fit_plan(
                        [lo_tr], gold_tr, [cost_small], cost_large,
                        target_prec=tq, target_rec=tq,
                        confidence=args.confidence, steps=args.steps,
                        restarts=args.restarts, seed=args.seed + q.qid)
                    pred, esc = so.simulate_hard(
                        [q.log_odds], plan.theta_lower, plan.theta_upper, plan.pick, q.gold)
                    m = _binary_metrics(pred, q.human)
                    row = dict(
                        dataset=ds, query_name=q.query_name, qid=q.qid,
                        target=tq, n_valid=n, n_train=int(len(tr)),
                        esc_rate=float(esc.mean()),
                        acc=m["acc"], f1=m["f1"],
                        tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
                        acc_vs_70b=float(np.mean(pred == q.gold)),
                        theta_lower=float(plan.theta_lower[0]),
                        theta_upper=float(plan.theta_upper[0]),
                        feasible=bool(plan.feasible),
                        status="ok",
                    )
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                print(f"  [{ds}] q{q.qid} {q.query_name}: done ({len(targets)} targets, n={n})")
        print(f"[{ds}] wrote {fout_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="opp115,cuad,contract,hoc")
    ap.add_argument("--out", default="runs/stretto_2026_07_02")
    ap.add_argument("--gold-cache", default=None,
                    help="large-model zero-shot csv to use as the cascade gold "
                         "(default = Llama-3.1-70B). Pass the 397B canonical csv "
                         "to run with 397B as the large model.")
    ap.add_argument("--large-in", type=float, default=LARGE_IN,
                    help="large-model input $/token for the optimizer's cost term")
    ap.add_argument("--large-out", type=float, default=LARGE_OUT,
                    help="large-model output $/token for the optimizer's cost term")
    ap.add_argument("--small-conf", default=None,
                    help="small-model 3-way confidence csv for the log-odds "
                         "(default = 0.8B). Pass the 8B cache to run with 8B small.")
    ap.add_argument("--small-in", type=float, default=SMALL_IN,
                    help="small-model input $/token for the optimizer's cost term")
    ap.add_argument("--small-out", type=float, default=SMALL_OUT,
                    help="small-model output $/token for the optimizer's cost term")
    ap.add_argument("--train-frac", type=float, default=0.15)
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=43)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
