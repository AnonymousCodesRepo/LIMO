"""Two-phase router-as-detector evaluation.

Goal
----
Measure how well the learned router predicts "the small model will disagree
with the large model" (i.e. this point should escalate), compared against two
training-free signals from the same small-model forward pass:

  * Ours          — the router's calibrated P(z=1); escalate-score = 1 - P(z=1)
  * 3-way conf    — escalate-score = 1 - max(p_yes, p_no, p_unsure)
  * True/False margin — escalate-score = 1 - |p_yes - p_no|

Design
------
1. Randomly split each dataset's (query, doc) points into part1 (default 60%)
   and part2 (40%). part1 builds the online experiences AND trains the router
   online; part2 is held out.
2. After part1 the WHOLE Ours system is frozen: no router refit, no retriever
   counter update, no new experience generation (``runner.frozen = True`` +
   ``exp_gen = None`` + router ``_freeze_online_fit``). This prevents part2
   agreement outcomes from leaking into the router / retriever features.
3. On part2 we record, per point, the small model's 3-way class probabilities
   and the router's P(z=1). The large-model verdict (the oracle for the
   agreement label) is taken from the prediction cache when present and only
   fetched live on a miss.

This script only produces the per-point CSV; scoring (AUPRC / AUROC / best-F1 /
accuracy-at-matched-escalation) is a separate step so it can be re-run without
touching the models.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from pipeline.common.prompts import build_replay_messages

from .cli import _apply_override, _load_file, _to_dataclass
from .config import EvalConfig
from .run import build_pipeline


def _split_indices(n: int, frac_part1: float, seed: int) -> tuple[list[int], list[int]]:
    """Random split of ``range(n)`` into (part1, part2) index lists.

    part1 gets ``round(n * frac_part1)`` points. Both lists are returned in the
    original data order so each phase streams in the configured within-query
    order (only membership is randomized, not the per-phase ordering).
    """
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n1 = round(n * frac_part1)
    part1 = sorted(idx[:n1])
    part2 = sorted(idx[n1:])
    return part1, part2


def _oracle_large_pred(runner, point, rec) -> tuple[str, bool]:
    """Return (large_pred, was_live). Reuse the escalation call when the point
    already escalated; otherwise consult the runner's clean 2-way large path
    (cache hit = free, miss = one live 70B call)."""
    if rec.escalated and rec.prediction not in (None, "UNKNOWN"):
        return rec.prediction, False
    large_msgs = build_replay_messages(
        document_text=point.doc_text,
        query_description=point.query_description,
        retrieved_experiences=None,
        fewshot_demos=None,
        system_base=runner.large_system_prompt,
    )
    pred, _raw, prompt_tokens, _ct = runner._call_large(large_msgs, point=point)
    return pred, prompt_tokens > 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Two-phase router-as-detector evaluation (produces part2 CSV)."
    )
    p.add_argument("--config", required=True, help="Base Ours cascade YAML/JSON.")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="DOTTED.PATH=VALUE",
        help="Override a config field (JSON value), repeatable.",
    )
    p.add_argument("--out-dir", required=True, help="Directory for the CSV + meta.")
    p.add_argument(
        "--split-frac", type=float, default=0.6,
        help="Fraction of points used for part1 (online training). Default 0.6.",
    )
    p.add_argument("--split-seed", type=int, default=0)
    args = p.parse_args(argv)

    raw = _load_file(args.config)
    for spec in args.overrides:
        _apply_override(raw, spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # This driver never uses the single-pass results writer, but EvalConfig
    # still wants a reporting.output; point it at an unused path.
    raw.setdefault("reporting", {})["output"] = str(out_dir / "_unused.json")
    cfg = _to_dataclass(EvalConfig, raw)
    dataset = cfg.data.dataset

    bundle = build_pipeline(cfg)
    runner = bundle.runner
    data = bundle.data
    n = len(data)

    part1_idx, part2_idx = _split_indices(n, args.split_frac, args.split_seed)
    part1 = [data[i] for i in part1_idx]
    part2 = [data[i] for i in part2_idx]
    print(
        f"\n[split] n={n}  part1={len(part1)}  part2={len(part2)}  "
        f"frac_part1={args.split_frac}  seed={args.split_seed}",
        flush=True,
    )

    # ---- Phase 1: online experiences + online router training on part1 ----
    print("\n=== PHASE 1: part1 (build experiences + train router online) ===", flush=True)
    t0 = time.perf_counter()
    runner.run(part1, progress_every=cfg.runner.progress_every)
    t_p1 = time.perf_counter() - t0
    fit_info = getattr(runner.router, "fit_info", lambda: {})()
    n_online = len(runner.online_experiences)
    print(f"[phase1] {t_p1:.1f}s  online_experiences={n_online}", flush=True)
    print(f"[phase1] router fit_info: {json.dumps(fit_info)}", flush=True)
    if not fit_info.get("trained", False):
        print(
            "[phase1][WARN] router never left bootstrap on part1 → router_p_z1 "
            "will be null and 'Ours' degenerates to the confidence rule. "
            "Consider a larger --split-frac or lower bootstrap minima.",
            flush=True,
        )

    # ---- Freeze the whole Ours system before touching part2 ----
    runner.frozen = True
    runner.exp_gen = None
    if hasattr(runner.router, "_freeze_online_fit"):
        runner.router._freeze_online_fit = True
    print(
        "[freeze] router refit OFF · retriever updates OFF · experience-gen OFF",
        flush=True,
    )

    # ---- Phase 2: frozen inference over held-out part2 ----
    print("\n=== PHASE 2: part2 (frozen inference) ===", flush=True)
    t0 = time.perf_counter()
    recs2, _ = runner.run(part2, progress_every=cfg.runner.progress_every)
    t_p2 = time.perf_counter() - t0
    print(f"[phase2] {t_p2:.1f}s", flush=True)

    # ---- Join the large-model oracle label; write per-point rows ----
    rows: list[dict] = []
    n_live_large = 0
    for point, rec in zip(part2, recs2):
        large_pred, was_live = _oracle_large_pred(runner, point, rec)
        n_live_large += int(was_live)
        rows.append({
            "query_name": rec.query_name,
            "doc_id": rec.doc_id,
            "gold": rec.ground_truth,
            "small_pred": rec.small_prediction,
            "large_pred": large_pred,
            "p_yes": rec.small_p_yes,
            "p_no": rec.small_p_no,
            "p_unsure": rec.small_p_unsure,
            "router_p_z1": rec.router_p_z1,
            "router_bootstrap": rec.router_bootstrap,
        })

    csv_path = out_dir / f"{dataset}_part2_detector.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_unsure = sum(1 for r in rows if r["small_pred"] in (None, "UNKNOWN", "Unsure"))
    meta = {
        "dataset": dataset,
        "config": args.config,
        "split_frac_part1": args.split_frac,
        "split_seed": args.split_seed,
        "n_total": n,
        "n_part1": len(part1),
        "n_part2": len(part2),
        "phase1_online_experiences": n_online,
        "phase1_router_fit_info": fit_info,
        "phase2_n_definitive": len(rows) - n_unsure,
        "phase2_n_unsure_or_unknown": n_unsure,
        "phase2_router_p_z1_null": sum(1 for r in rows if r["router_p_z1"] is None),
        "phase2_live_large_calls": n_live_large,
    }
    meta_path = out_dir / f"{dataset}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n[out] {csv_path}", flush=True)
    print(f"[out] {meta_path}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
