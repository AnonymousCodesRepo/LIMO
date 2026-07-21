"""Stretto baseline -- Track A (KV-cache compression ladder on the 70B side).

Ladder per (dataset, query), cost order:
    qwen08b@r0.0  ->  llama70b@r0.8  ->  llama70b@r0.6  ->  llama70b@r0.3  ->  llama70b@r0.0 (gold)

The small rung is the SAME operator Track B uses (cached vLLM 3-way log-odds), so
Track A vs Track B isolates the contribution of the 70B mid rungs. The 70B mid
rungs are kvpress-profiled on the in_sample70 rows only (tools/stretto/
profile_ladder.py on a GPU node), so fitting AND evaluation happen on those rows.
(Qwen3.5-0.8B is a hybrid linear-attention model: kvpress compression is
architecturally inapplicable to the small side.)

Two cost models, same ladder:
  * token   : the paper's token-USD prices (mid rungs share the gold token
              price).
  * runtime : Stretto's runtime-style tiers (more compression = cheaper);
              operator-usage analysis (their Fig. 6/7 analog) + a
              sample-estimated (esc, acc) curve.

Output: {out}/stretto_full_{ds}.jsonl, one row per (query, cost_model, target):
  esc_rate = fraction reaching ANY 70B rung (mid or gold);
  pick / rung_resolved = operator-usage diagnostics.

Run on GPU server:
  PYTHONPATH=. <python> tools/stretto/run_stretto_full.py \
      --ladder-dir runs/stretto_ladder_out \
      --out runs/stretto_full
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stretto_data as sd          # noqa: E402
import stretto_opt as so           # noqa: E402
from run_stretto import AVG_IN, SMALL_IN, SMALL_OUT, LARGE_IN, LARGE_OUT, \
    DEFAULT_TARGETS, _binary_metrics   # noqa: E402

LLAMA_RUNGS = [0.8, 0.6]           # cost order among the mids (cheapest first)
# Stretto-style runtime tiers (fake_cost pattern from their run_benchmark.py:
# compression tier + a large-model size penalty).
RUNTIME_COST = {("qwen08b", 0.0): 0.05,
                ("llama70b", 0.8): 0.3 + 0.5, ("llama70b", 0.6): 0.6 + 0.5}
RUNTIME_GOLD_COST = 1.0 + 0.5


def _load_rung_csv(path: str) -> dict[tuple[int, str], float]:
    df = pd.read_csv(path, usecols=["document_id", "query_name",
                                    "logprob_true", "logprob_false"])
    lo = np.clip(df["logprob_true"].to_numpy() - df["logprob_false"].to_numpy(),
                 -sd.LOGODDS_CLIP, sd.LOGODDS_CLIP)
    return {(int(d), q): float(x) for d, q, x in
            zip(df["document_id"], df["query_name"], lo)}


def _sample_mask(input_dir: str, ds: str) -> dict[tuple[int, str], bool]:
    p = pd.read_csv(os.path.join(input_dir, f"{ds}_pairs.csv"))
    if p["in_sample70"].dtype != bool:
        p["in_sample70"] = p["in_sample70"].astype(str).str.lower() == "true"
    return {(int(d), q): bool(s) for d, q, s in
            zip(p["document_id"], p["query_name"], p["in_sample70"])}


def load_ladder(ds: str, ladder_dir: str, input_dir: str):
    queries = sd.load_dataset(ds)   # 0.8B@0 log-odds (cache), gold, human
    rung_maps = {}
    for r in LLAMA_RUNGS:
        path = os.path.join(ladder_dir, f"llama70b_r{r}_{ds}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing 70B rung profile: {path}")
        rung_maps[("llama70b", r)] = _load_rung_csv(path)
    smask = _sample_mask(input_dir, ds)

    out = []
    for q in queries:
        keys = [(int(d), q.query_name) for d in q.doc_ids]
        entry = {"q": q,
                 "sample": np.array([smask.get(k, False) for k in keys]),
                 ("qwen08b", 0.0): q.log_odds}
        for rung, m in rung_maps.items():
            entry[rung] = np.array([m.get(k, np.nan) for k in keys])
        out.append(entry)
    return out


def _rung_resolved(lo_list, plan, gold):
    """Per-rung counts of tuples RESOLVED (accept/reject) at that rung."""
    n = len(gold)
    unresolved = np.ones(n, dtype=bool)
    counts = []
    for k in range(len(lo_list)):
        if plan.pick[k] < 0.5:
            counts.append(0)
            continue
        acc = unresolved & (lo_list[k] > plan.theta_upper[k])
        rej = unresolved & (lo_list[k] < plan.theta_lower[k])
        counts.append(int(acc.sum() + rej.sum()))
        unresolved &= ~acc & ~rej
    counts.append(int(unresolved.sum()))   # gold
    return counts


def run(args):
    os.makedirs(args.out, exist_ok=True)
    targets = [float(x) for x in args.targets.split(",")]
    rungs = [("qwen08b", 0.0)] + [("llama70b", r) for r in LLAMA_RUNGS]
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        avg_in = AVG_IN.get(ds, 500.0)
        token_costs = [avg_in * SMALL_IN + SMALL_OUT] + \
                      [avg_in * LARGE_IN + LARGE_OUT] * len(LLAMA_RUNGS)
        token_gold = avg_in * LARGE_IN + LARGE_OUT
        runtime_costs = [RUNTIME_COST[r] for r in rungs]
        try:
            ladder = load_ladder(ds, args.ladder_dir, args.input_dir)
        except FileNotFoundError as e:
            print(f"[{ds}] SKIP: {e}")
            continue
        fout = open(os.path.join(args.out, f"stretto_full_{ds}.jsonl"), "w")
        for e in ladder:
            q = e["q"]
            tr = e["sample"].copy()
            for r in rungs:
                tr &= np.isfinite(e[r])
            if tr.sum() < 20:
                print(f"  [{ds}] q{q.qid} skipped (only {tr.sum()} usable sample rows)")
                continue
            lo = [e[r][tr] for r in rungs]
            gold, human = q.gold[tr], q.human[tr]
            for mode, costs, gcost in (("token", token_costs, token_gold),
                                       ("runtime", runtime_costs, RUNTIME_GOLD_COST)):
                for tq in targets:
                    plan = so.fit_plan(lo, gold, costs, gcost,
                                       target_prec=tq, target_rec=tq,
                                       confidence=args.confidence, steps=args.steps,
                                       restarts=args.restarts, optimize_pick=True,
                                       seed=args.seed + q.qid)
                    pred, esc = so.simulate_hard(lo, plan.theta_lower,
                                                 plan.theta_upper, plan.pick, gold)
                    # escalation = reached ANY 70B rung = not resolved by the small rung
                    resolved = _rung_resolved(lo, plan, gold)
                    n_eval = int(tr.sum())
                    esc_any70 = 1.0 - resolved[0] / n_eval
                    m = _binary_metrics(pred, human)
                    fout.write(json.dumps(dict(
                        dataset=ds, query_name=q.query_name, qid=q.qid,
                        cost_model=mode, target=tq, n_eval=n_eval,
                        esc_rate=esc_any70,
                        esc_gold=float(esc.mean()),
                        acc=m["acc"], f1=m["f1"],
                        tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
                        acc_vs_70b=float(np.mean(pred == gold)),
                        rungs=[f"{a}@{b}" for a, b in rungs] + ["llama70b@0.0(gold)"],
                        pick=[float(x) for x in plan.pick],
                        rung_resolved=resolved,
                        theta_lower=[float(x) for x in plan.theta_lower],
                        theta_upper=[float(x) for x in plan.theta_upper],
                        feasible=bool(plan.feasible), status="ok")) + "\n")
            fout.flush()
            print(f"  [{ds}] q{q.qid} {q.query_name} done (n_eval={int(tr.sum())})")
        fout.close()
        print(f"[{ds}] wrote stretto_full_{ds}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="opp115,cuad,contract,hoc")
    ap.add_argument("--ladder-dir", default="runs/stretto_ladder_out")
    ap.add_argument("--input-dir", default="tools/stretto/ladder_inputs")
    ap.add_argument("--out", default="runs/stretto_full_2026_07_02")
    ap.add_argument("--targets", default=DEFAULT_TARGETS)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=43)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
