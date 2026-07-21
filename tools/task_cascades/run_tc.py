"""Task-Cascades baseline driver: per-query cascade construction + evaluation.

Runs the upstream method (via mop_adapter, zero upstream edits) once per
(query, variant, alpha) and records per-document outcomes so we can report
accuracy/F1 against GOLD labels, escalation rate, and full-price USD cost —
the axes our figures use — rather than upstream's agreement-vs-oracle.

Accounting:
  * dev (train-split) documents are always counted as "escalated to the 70B":
    prediction = the cached 70B zero-shot answer, cost = the full-price tokens
    of the clean 2-way prompt (bit-identical to the zs-70B anchor).
  * test documents are priced along the cascade's actual path, all tokens at
    full price (no prefix-cache discount).
  * the remaining optimization overhead (line-range extraction, surrogate
    evaluation, agent calls) is recorded per phase from the adapter's call
    log into the opt_cost table of summary.json; it is not part of the
    main-figure cost.

Usage (upstream clone required):
  export MOP_TC_UPSTREAM=/path/to/task-cascades
  PYTHONPATH=. $MOP_PYTHON tools/task_cascades/run_tc.py \
      --dataset cuad --queries cuad_governing_law --variants task_cascades_lite \
      --alphas 0.9 --out-dir runs/task_cascades_eval/runs/pilot

  # aggregate after a sweep:
  ... run_tc.py --aggregate --out-dir <same-out-dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Mirror upstream ExperimentRunner.run_method's exact find_surrogates args.
VARIANTS = {
    "task_cascades": dict(num_iterations=3, num_surrogate_requests=5),
    "task_cascades_lite": dict(num_iterations=1, num_surrogate_requests=8,
                               provide_feedback=True, include_selectivity=False,
                               proxy_predictor_only=True),
}


def metrics_vs_gold(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r["pred"] == r["gold"] for r in rows) / n
    tp = sum(r["pred"] == 1 and r["gold"] == 1 for r in rows)
    fp = sum(r["pred"] == 1 and r["gold"] == 0 for r in rows)
    fn = sum(r["pred"] == 0 and r["gold"] == 1 for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    esc = sum(r["stage"] in ("oracle", "dev_oracle") for r in rows) / n
    return {
        "n": n, "acc": acc, "f1": f1, "precision": prec, "recall": rec,
        "tp": tp, "fp": fp, "fn": fn,
        "agree_oracle": sum(r["pred"] == r["oracle_pred"] for r in rows) / n,
        "esc_rate": esc,
        "cost_usd": sum(r["cost_usd"] for r in rows),
    }


def simulate_recorded(ordering, thresholds, exdf, test_docs, prices) -> list[dict]:
    """Per-document replay of upstream simulate_cascade under full-price
    accounting (cache price == input price, so marginal == full cost).

    Unlike upstream apply_cascade we skip its "missing-uuid oracle top-up"
    pass: documents with no matching stage rows fall straight to the oracle
    (prediction = cached label, oracle cost), which is what the top-up
    achieves, without live calls. Predictions are identical.
    """
    (s_in, s_out), (l_in, l_out) = prices["gpt-4o-mini"], prices["gpt-4o"]
    per_model_prices = {"gpt-4o-mini": (s_in, s_out), "gpt-4o": (l_in, l_out)}
    out = []
    have = len(exdf) > 0
    for doc in test_docs:
        rows = exdf[exdf["uuid"] == doc["uuid"]] if have else None
        cost = 0.0
        tok = defaultdict(int)
        stage = None
        pred = None
        for cand in ordering:
            name, model, frac = cand
            if rows is None or not len(rows):
                break
            m = rows[(rows["surrogate_name"] == name)
                     & (rows["surrogate_model"] == model)
                     & (rows["fraction"] == frac)]
            if not len(m):
                continue
            usage = m["surrogate_usage"].values[0]
            cin, cout = per_model_prices[model]
            cost += usage.prompt_tokens * cin + usage.completion_tokens * cout
            key = "small" if model == "gpt-4o-mini" else "large"
            tok[f"{key}_in"] += usage.prompt_tokens
            tok[f"{key}_out"] += usage.completion_tokens
            conf = m["surrogate_confidence"].values[0]
            p = m["surrogate_prediction"].values[0]
            if conf >= thresholds[cand].get(p, float("inf")):
                stage = f"{name}|{model}|{frac}"
                pred = int(p)
                break
        if stage is None:
            stage = "oracle"
            pred = doc["oracle_pred"]
            cost += doc["oracle_prompt_tokens"] * l_in \
                + doc["oracle_output_tokens"] * l_out
            tok["large_in"] += doc["oracle_prompt_tokens"]
            tok["large_out"] += doc["oracle_output_tokens"]
        out.append({
            "document_id": doc["document_id"], "split": "test",
            "stage": stage, "pred": pred, "gold": doc["gold"],
            "oracle_pred": doc["oracle_pred"], "cost_usd": cost, **tok,
        })
    return out


def summarize_call_log(log: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for c in log:
        k = f'{c["phase"]}::{c["requested_model"]}::{c["kind"]}'
        a = agg.setdefault(k, {"calls": 0, "prompt_tokens": 0,
                               "completion_tokens": 0, "cost_usd": 0.0,
                               "litellm_cache_hits": 0})
        a["calls"] += 1
        a["prompt_tokens"] += c["prompt_tokens"]
        a["completion_tokens"] += c["completion_tokens"]
        a["cost_usd"] += c["cost_usd"]
        a["litellm_cache_hits"] += c["litellm_cache_hit"]
    return agg


def run_query(args, adapter, dataset: str, query: str, task_meta: dict,
              meta_rows: dict, out_root: Path) -> None:
    import pandas as pd
    from eval.data import load_points
    from task_cascades.cascade.find_surrogates import find_surrogates
    from task_cascades.experiments.experiment_runner import ExperimentRunner
    from task_cascades.predictors.predictors import (
        PROMPT_TO_TASK_TYPE_DICT, TASK_PROMPT_DICT,
        run_predictor_and_get_row_copies,
    )
    from mop.llm import (
        LARGE_COST_PER_INPUT_TOKEN, LARGE_COST_PER_OUTPUT_TOKEN,
        SMALL_COST_PER_INPUT_TOKEN, SMALL_COST_PER_OUTPUT_TOKEN,
    )
    prices = {"gpt-4o-mini": (SMALL_COST_PER_INPUT_TOKEN, SMALL_COST_PER_OUTPUT_TOKEN),
              "gpt-4o": (LARGE_COST_PER_INPUT_TOKEN, LARGE_COST_PER_OUTPUT_TOKEN)}

    task = f"tc_{query}"
    qdir = out_root / query
    qdir.mkdir(parents=True, exist_ok=True)

    pts = load_points([query], dataset=dataset)
    assert pts, f"no docs for {query}"
    prompt_template = adapter.register_task(
        task, task_meta["prefix"], task_meta["instruction"], task_meta["suffix"])

    df = pd.DataFrame({
        "document_id": [p.doc_id for p in pts],
        "text": [p.doc_text for p in pts],
    })
    adapter.register_dataframe(task, df, [p.doc_text for p in pts])

    for p in pts:
        m = meta_rows[p.doc_id]
        adapter.register_oracle_answer(
            prompt_template.format(text=p.doc_text), m["oracle_pred"],
            m["oracle_prompt_tokens"], m["oracle_output_tokens"])

    n = len(pts)
    dev_n = min(args.max_dev, max(args.min_dev, int(round(args.dev_frac * n))))
    if dev_n >= n:
        print(f"[skip] {query}: dev_n {dev_n} >= n_docs {n}")
        return

    adapter.set_phase("prepare")
    runner = ExperimentRunner(
        task=task, sample_size=n, train_split=(dev_n + 0.5) / n,
        seed=args.seed, cache_dir=str(qdir / "cache"),
        results_dir=str(qdir / "results"))
    runner.prepare()
    prepare_log = adapter.drain_call_log()

    # sanity: oracle labels must have come from the cache, prices registered
    all_docs = pd.concat([runner.train_df, runner.test_df])
    mism = sum(int(r.label) != meta_rows[int(r.document_id)]["oracle_pred"]
               for r in all_docs.itertuples())
    # A cache miss makes upstream's internal labeling fall back to a LIVE oracle
    # call whose answer can differ from the cached zero-shot label (e.g. a
    # co-tenant 397B oracle is not bit-deterministic on a duplicate-text doc).
    # This only affects upstream's filtering-classifier training, NOT the
    # reported cost/accuracy (simulate_recorded serves escalations from
    # meta_rows["oracle_pred"], the zero-shot cache). Tolerate a tiny count;
    # abort only on a systematic mismatch that signals a real wiring break.
    n_all = len(all_docs)
    assert mism <= max(3, int(0.002 * n_all)), \
        f"{mism}/{n_all} oracle labels missed the prediction cache (systematic)"
    if mism:
        print(f"[warn] {query}: {mism}/{n_all} oracle labels differ from the "
              f"zero-shot cache (live-fallback nondeterminism; measurement unaffected)")
    assert all_docs["baseline_cost"].sum() > 0, "baseline cost is 0 — price table broken"
    assert all_docs["baseline_cost"].sum() > 0, "baseline cost is 0 — price table broken"

    # Serve the (unused-as-candidate) s1/oracle/f=1.0 rows from the cache too:
    # same content reordered, same question => same cached answer, no live call.
    for split_df in (runner.train_df_filtered, runner.test_df_filtered):
        for r in split_df[split_df["fraction"] == 1.0].itertuples():
            m = meta_rows[int(r.document_id)]
            adapter.register_oracle_answer(
                prompt_template.format(text=r.filtered_text), m["oracle_pred"],
                m["oracle_prompt_tokens"], m["oracle_output_tokens"])

    dev_rows = []
    l_in, l_out = prices["gpt-4o"]
    for r in runner.train_df.drop_duplicates(subset="uuid").itertuples():
        m = meta_rows[int(r.document_id)]
        dev_rows.append({
            "document_id": int(r.document_id), "split": "dev",
            "stage": "dev_oracle", "pred": m["oracle_pred"],
            "gold": m["gold"], "oracle_pred": m["oracle_pred"],
            "cost_usd": m["oracle_prompt_tokens"] * l_in
            + m["oracle_output_tokens"] * l_out,
            "large_in": m["oracle_prompt_tokens"],
            "large_out": m["oracle_output_tokens"],
        })

    test_docs = []
    for r in runner.test_df.drop_duplicates(subset="uuid").itertuples():
        m = meta_rows[int(r.document_id)]
        test_docs.append({
            "uuid": r.uuid, "document_id": int(r.document_id),
            "gold": m["gold"], "oracle_pred": m["oracle_pred"],
            "oracle_prompt_tokens": m["oracle_prompt_tokens"],
            "oracle_output_tokens": m["oracle_output_tokens"],
        })

    task_type = PROMPT_TO_TASK_TYPE_DICT[task]
    for variant in args.variants:
        for alpha in args.alphas:
            tag = f"{variant}_a{alpha}"
            spath = qdir / f"summary_{tag}.json"
            if spath.exists() and not args.force:
                print(f"[resume] {query} {tag}: summary exists, skip")
                continue
            t0 = time.time()
            adapter.set_phase(f"discover::{tag}")
            cascade = find_surrogates(runner.train_df_filtered, task, alpha,
                                      **VARIANTS[variant])
            discover_log = adapter.drain_call_log()

            ordering = cascade["greedy"]["ordering"]
            thresholds = cascade["greedy"]["thresholds"]
            s2p = cascade["surrogate_to_prompt"]

            adapter.set_phase(f"apply::{tag}")
            all_exec = []
            seen = set()
            for cand in ordering:
                name, model, frac = cand
                if cand in seen or not s2p.get(name):
                    continue
                seen.add(cand)
                sub = runner.test_df_filtered[
                    runner.test_df_filtered["fraction"] == frac
                ].reset_index(drop=True)
                all_exec.extend(run_predictor_and_get_row_copies(
                    model, s2p[name], sub, name, task_type=task_type))
            exdf = pd.DataFrame(all_exec)
            apply_log = adapter.drain_call_log()

            test_rows = simulate_recorded(ordering, thresholds, exdf,
                                          test_docs, prices)
            per_doc = test_rows + dev_rows
            for r in per_doc:
                r.update({"dataset": dataset, "query": query,
                          "variant": variant, "alpha": alpha})
            pd.DataFrame(per_doc).to_csv(
                qdir / f"per_doc_{tag}.csv", index=False)

            summary = {
                "dataset": dataset, "query": query, "variant": variant,
                "alpha": alpha, "n_docs": n, "n_dev": len(dev_rows),
                "n_test": len(test_rows), "seed": args.seed,
                "dev_frac": args.dev_frac,
                "test": metrics_vs_gold(test_rows),
                "full": metrics_vs_gold(per_doc),
                "cascade": {
                    "ordering": [list(c) for c in ordering],
                    "thresholds": {str(k): v for k, v in thresholds.items()},
                    "surrogate_prompts": s2p,
                },
                "opt_cost": {
                    "prepare": summarize_call_log(prepare_log),
                    "discover": summarize_call_log(discover_log),
                    "apply": summarize_call_log(apply_log),
                },
                "wall_s": time.time() - t0,
            }
            with open(spath, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            fm, tm = summary["full"], summary["test"]
            print(f"[done] {query} {tag}: full acc {fm['acc']:.3f} "
                  f"f1 {fm['f1']:.3f} esc {fm['esc_rate']:.3f} "
                  f"${fm['cost_usd']:.4f} | test acc {tm['acc']:.3f} "
                  f"| {summary['wall_s']:.0f}s | stages {len(ordering)}")


def aggregate(out_root: Path) -> None:
    import pandas as pd
    frames = [pd.read_csv(p) for p in sorted(out_root.glob("*/per_doc_*.csv"))]
    if not frames:
        raise SystemExit(f"no per_doc files under {out_root}")
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for (variant, alpha), g in df.groupby(["variant", "alpha"]):
        for scope, gg in (("full", g), ("test", g[g["split"] == "test"])):
            m = metrics_vs_gold(gg.to_dict("records"))
            rows.append({"variant": variant, "alpha": alpha, "scope": scope,
                         "n_queries": g["query"].nunique(), **m})
    out = pd.DataFrame(rows)
    path = out_root / "dataset_summary.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cuad")
    ap.add_argument("--queries", default="all",
                    help="comma-separated query names, or 'all'")
    ap.add_argument("--variants", default="task_cascades_lite",
                    help=f"comma-separated from {sorted(VARIANTS)}")
    ap.add_argument("--alphas", default="0.9",
                    help="comma-separated target accuracies")
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--min-dev", type=int, default=20)
    ap.add_argument("--max-dev", type=int, default=200,
                    help="cap on dev-set size (paper uses 200); binds on "
                         "large rectangular datasets like opp115")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tc-data", default=str(HERE / "tc_data"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--shard", default=None,
                    help="'i/N': run queries i, i+N, i+2N, ... (0-based) for "
                         "parallel workers sharing one --out-dir")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve() / args.dataset
    if args.aggregate:
        aggregate(out_root)
        return

    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    for v in args.variants:
        assert v in VARIANTS, f"unknown variant {v}"

    tc_data = Path(args.tc_data).resolve() / args.dataset
    with open(tc_data / "tasks.json") as f:
        tasks = json.load(f)
    queries = (sorted(tasks) if args.queries == "all"
               else [q.strip() for q in args.queries.split(",")])
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        queries = queries[i::n]
        print(f"shard {i}/{n}: {len(queries)} queries")

    out_root.mkdir(parents=True, exist_ok=True)
    # litellm's disk cache and upstream's relative cache/ both live in CWD.
    os.chdir(out_root)

    upstream = os.environ.get("MOP_TC_UPSTREAM")
    assert upstream and Path(upstream).is_dir(), \
        "set MOP_TC_UPSTREAM to the task-cascades clone"
    sys.path.insert(0, upstream)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(HERE))

    import mop_adapter as adapter
    adapter.init()

    import pandas as pd
    for q in queries:
        assert q in tasks, f"{q} not in {tc_data}/tasks.json"
        meta = pd.read_csv(tc_data / f"{q}.csv")
        meta_rows = {int(r.doc_id): {
            "gold": int(r.gold), "oracle_pred": int(r.oracle_pred),
            "oracle_prompt_tokens": int(r.oracle_prompt_tokens),
            "oracle_output_tokens": int(r.oracle_output_tokens),
        } for r in meta.itertuples()}
        print(f"\n===== {args.dataset} / {q} ({tasks[q]['n_docs']} docs) =====")
        run_query(args, adapter, args.dataset, q, tasks[q], meta_rows, out_root)


if __name__ == "__main__":
    main()
