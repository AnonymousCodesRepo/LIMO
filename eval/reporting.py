"""Results JSON + console summary + optional sidecar recording."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline.common.types import ProcessedRecord

from .config import EvalConfig, ReportConfig
from .metrics import binary_metrics, frontier_delta


REPO_ROOT = Path(__file__).resolve().parent.parent


def _abs_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def serialise_record(r: ProcessedRecord) -> dict[str, Any]:
    return {
        "doc_id": r.doc_id,
        "query_name": r.query_name,
        "ground_truth": r.ground_truth,
        "prediction": r.prediction,
        "raw": r.raw,
        "routed_to": r.routed_to,
        "retrieved_experiences": r.retrieved_experiences,
        "retrieved_fewshot": r.retrieved_fewshot,
        "latency": r.latency,
        "small_prediction": r.small_prediction,
        "small_confidence": r.small_confidence,
        "escalated": r.escalated,
        "router_p_z1": r.router_p_z1,
        "router_bootstrap": r.router_bootstrap,
        "t_small": r.t_small,
        "t_route_decision": r.t_route_decision,
        "t_large": r.t_large,
        "t_retrieve_exp": r.t_retrieve_exp,
        "t_retrieve_fs": r.t_retrieve_fs,
        "t_router_signals": r.t_router_signals,
        "t_observe": r.t_observe,
        "small_prompt_tokens": r.small_prompt_tokens,
        "small_completion_tokens": r.small_completion_tokens,
        "large_prompt_tokens": r.large_prompt_tokens,
        "large_completion_tokens": r.large_completion_tokens,
        "verify_small_calls": r.verify_small_calls,
        "verify_small_prompt_tokens": r.verify_small_prompt_tokens,
        "verify_small_completion_tokens": r.verify_small_completion_tokens,
        "verify_score": r.verify_score,
    }


def _token_summary(records: list[ProcessedRecord]) -> dict[str, Any]:
    """Per-model token + call totals, metered live. Large token counts are 0
    for cache-served escalations; the offline clean-2-way token table charges
    their cost at aggregation. ``n_large_calls`` counts every escalation (cache
    or live), so cost accounting can charge each one the full large cost.

    AutoMix-style self-verification issues k EXTRA small calls per verified
    point. The ``small_*`` totals (and ``n_small_calls``) below INCLUDE those
    verification calls so the cost axis is fair; the ``verify_small_*`` keys
    break them out separately for auditing. Every non-AutoMix router records
    0 verification calls, so those routers' totals are unchanged."""
    verify_calls = sum(r.verify_small_calls for r in records)
    verify_pt = sum(r.verify_small_prompt_tokens for r in records)
    verify_ct = sum(r.verify_small_completion_tokens for r in records)
    return {
        # every point with a small prediction had a live small call (escalated
        # ones too — they hit the small model first, then escalated), PLUS any
        # extra self-verification calls the router issued.
        "n_small_calls": sum(1 for r in records if r.small_prediction is not None)
        + verify_calls,
        "n_large_calls": sum(1 for r in records if r.escalated or r.routed_to == "large"),
        "small_prompt_tokens": sum(r.small_prompt_tokens for r in records) + verify_pt,
        "small_completion_tokens": sum(r.small_completion_tokens for r in records)
        + verify_ct,
        "large_prompt_tokens_live": sum(r.large_prompt_tokens for r in records),
        "large_completion_tokens_live": sum(r.large_completion_tokens for r in records),
        # Self-verification breakdown (AutoMix); 0 for every other router.
        "verify_small_calls": verify_calls,
        "verify_small_prompt_tokens": verify_pt,
        "verify_small_completion_tokens": verify_ct,
    }


def build_results(
    *,
    config: EvalConfig,
    queries: list[str],
    records: list[ProcessedRecord],
    wall_clock_seconds: float,
    router: Any,
    generator: Any,
    retriever: Any,
    online_experiences: list[dict[str, Any]],
    large_cache: Any,
) -> dict[str, Any]:
    rec_dicts = [serialise_record(r) for r in records]
    n_escalated = sum(1 for r in rec_dicts if r.get("escalated"))
    n_small_only = sum(
        1 for r in rec_dicts
        if r["routed_to"] == "small" and not r.get("escalated")
    )
    overall = binary_metrics(rec_dicts)

    per_query: dict[str, dict[str, Any]] = {}
    for q in queries:
        sub = [r for r in rec_dicts if r["query_name"] == q]
        if not sub:
            continue
        m = binary_metrics(sub)
        n_esc_q = sum(1 for r in sub if r.get("escalated"))
        m["n_escalated"] = n_esc_q
        m["escalation_rate"] = n_esc_q / len(sub)
        per_query[q] = m

    timing_summary = _timing_summary(records, generator)

    out: dict[str, Any] = {
        "config": _config_to_dict(config),
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "n_points": len(rec_dicts),
        "n_small_only": n_small_only,
        "n_escalated": n_escalated,
        "escalation_rate": (
            n_escalated / len(rec_dicts) if rec_dicts else 0.0
        ),
        "timing_summary": timing_summary,
        "token_summary": _token_summary(records),
        "router_fit_info": _safe_call(router, "fit_info"),
        "experience_generator_stats": _safe_call(generator, "stats"),
        "online_retriever_stats": _retriever_stats(retriever),
        "online_experiences": online_experiences,
        "large_prediction_cache_stats": (
            large_cache.stats() if large_cache is not None else None
        ),
        "overall": overall,
        "per_query": per_query,
        "records": rec_dicts,
    }

    if config.reporting.frontier_baseline:
        delta = frontier_delta(
            overall, out["escalation_rate"], config.reporting.frontier_baseline,
        )
        if delta is not None:
            out["frontier_delta"] = delta

    return out


def write_results(blob: dict[str, Any], cfg: ReportConfig) -> Path:
    out_path = _abs_path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False))
    return out_path


def write_recording_sidecar(
    *, generator: Any, retriever: Any, output: Path,
    record_output: str | None,
) -> Path | None:
    """If generator / retriever were wrapped with RecordingDiscrepancyGenerator
    / RecordingExperienceRetriever, dump their recordings to a sidecar."""
    gen_rec = getattr(generator, "recordings", None)
    ret_rec = getattr(retriever, "recordings", None)
    if gen_rec is None and ret_rec is None:
        return None

    sidecar: dict[str, Any] = {
        "main_output": str(output),
        "experiences": [],
        "summary": {},
    }
    if gen_rec is not None:
        triggers = gen_rec()
        if ret_rec is not None:
            rr = ret_rec()
            installs = rr.get("installs", {})
            usage = rr.get("usage", {})
            for t in triggers:
                eid = t.get("experience_id")
                if eid:
                    inst = installs.get(eid)
                    if inst is not None:
                        t["install_order"] = inst.get("install_order")
                    t["usage"] = usage.get(eid, {})
        sidecar["experiences"] = triggers
        n_acc = sum(1 for t in triggers if t.get("accept"))
        n_rej = len(triggers) - n_acc
        sidecar["summary"] = {
            "n_triggers_recorded": len(triggers),
            "n_accepted": n_acc,
            "n_rejected": n_rej,
            "generator_stats": _safe_call(generator, "stats"),
        }
    if ret_rec is not None:
        rr = ret_rec()
        sidecar["retriever_recordings"] = rr

    sc_path = (
        _abs_path(record_output) if record_output
        else Path(str(output) + ".recording.json")
    )
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    sc_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
    return sc_path


def print_summary(
    *,
    blob: dict[str, Any],
    queries: list[str],
    needs_large: bool,
    out_path: Path,
) -> None:
    overall = blob["overall"]
    n_esc = blob["n_escalated"]
    n = len(blob["records"])
    print("\n=== SUMMARY ===")
    print(f"wall_clock = {blob['wall_clock_seconds']:.1f}s,  n={overall['n']}")
    if needs_large and n:
        print(f"escalated: {n_esc}/{n} ({n_esc/n:.2%})")
    print(
        f"overall: acc={overall['accuracy']:.4f} f1={overall['f1']:.4f} "
        f"prec={overall['precision']:.4f} rec={overall['recall']:.4f}"
    )
    if "frontier_delta" in blob:
        d = blob["frontier_delta"]
        print(
            f"vs frontier @ esc={blob['escalation_rate']:.3f}: "
            f"Δacc={d['delta_acc']:+.4f}  Δf1={d['delta_f1']:+.4f}"
        )
    print(f"{'query':48} {'n':>5} {'acc':>7} {'f1':>7} {'esc':>6} {'esc%':>7}")
    for q in queries:
        m = blob["per_query"].get(q)
        if m:
            esc_n = m.get("n_escalated", 0)
            esc_r = m.get("escalation_rate", 0.0)
            print(
                f"{q:48} {m['n']:5d} {m['accuracy']:.3f}  {m['f1']:.3f} "
                f"{esc_n:6d} {esc_r:7.2%}"
            )
    print(f"\n[done] wrote {out_path}")


# ── helpers ──────────────────────────────────────────────────────────────


def _safe_call(obj: Any, method_name: str) -> Any:
    fn = getattr(obj, method_name, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _retriever_stats(retriever: Any) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    snap = _safe_call(retriever, "stats_snapshot")
    if isinstance(snap, dict) and snap:
        n_pos = sum(v.get("pos", 0) for v in snap.values())
        n_neg = sum(v.get("neg", 0) for v in snap.values())
        out["experiences_observed"] = len(snap)
        out["total_pos"] = n_pos
        out["total_neg"] = n_neg
        out["per_experience"] = snap
    elif snap is not None:
        out["stats_snapshot_error"] = snap
    flin = _safe_call(retriever, "feature_linucb_snapshot")
    if flin is not None:
        out["feature_linucb"] = flin
    care = _safe_call(retriever, "care_snapshot")
    if care is not None:
        out["care"] = care
    return out or None


def _timing_summary(
    records: list[ProcessedRecord], generator: Any,
) -> dict[str, Any]:
    def avg(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    t_small = [r.t_small for r in records if r.t_small is not None]
    t_large = [r.t_large for r in records if r.t_large is not None]
    t_route = [
        r.t_route_decision for r in records if r.t_route_decision is not None
    ]
    t_retr_exp = [
        r.t_retrieve_exp for r in records if r.t_retrieve_exp is not None
    ]
    t_retr_fs = [
        r.t_retrieve_fs for r in records if r.t_retrieve_fs is not None
    ]
    t_signals = [
        r.t_router_signals for r in records if r.t_router_signals is not None
    ]
    t_obs = [r.t_observe for r in records if r.t_observe is not None]
    summary: dict[str, Any] = {
        "small_initial_prediction": {
            "avg_seconds": avg(t_small),
            "n_calls": len(t_small),
            "total_seconds": round(sum(t_small), 3),
        },
        "router_decision": {
            "avg_seconds": avg(t_route),
            "n_calls": len(t_route),
            "total_seconds": round(sum(t_route), 4),
        },
        "large_prediction": {
            "avg_seconds": avg(t_large),
            "n_calls": len(t_large),
            "total_seconds": round(sum(t_large), 3),
        },
        "experience_retrieval": {
            "avg_seconds": avg(t_retr_exp),
            "n_calls": len(t_retr_exp),
            "total_seconds": round(sum(t_retr_exp), 3),
            "max_seconds": round(max(t_retr_exp), 4) if t_retr_exp else None,
        },
        "fewshot_retrieval": {
            "avg_seconds": avg(t_retr_fs),
            "n_calls": len(t_retr_fs),
            "total_seconds": round(sum(t_retr_fs), 3),
        },
        "retriever_router_signals": {
            "avg_seconds": avg(t_signals),
            "n_calls": len(t_signals),
            "total_seconds": round(sum(t_signals), 3),
        },
        "online_observe_updates": {
            "avg_seconds": avg(t_obs),
            "n_calls": len(t_obs),
            "total_seconds": round(sum(t_obs), 3),
            "max_seconds": round(max(t_obs), 4) if t_obs else None,
        },
    }
    g_stats = _safe_call(generator, "stats") if generator is not None else None
    if isinstance(g_stats, dict):
        summary["small_reasoning_for_experience"] = {
            "avg_seconds": g_stats.get("avg_t_small_explain"),
            "n_calls": g_stats.get("n_small_explain_calls", 0),
            "total_seconds": g_stats.get("t_small_explain_total", 0.0),
        }
        summary["large_experience_synthesis"] = {
            "avg_seconds": g_stats.get("avg_t_large_synthesize"),
            "n_calls": g_stats.get("n_large_synthesize_calls", 0),
            "total_seconds": g_stats.get("t_large_synthesize_total", 0.0),
        }
    return summary


def _config_to_dict(cfg: EvalConfig) -> dict[str, Any]:
    return asdict(cfg)
