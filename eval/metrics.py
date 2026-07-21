"""Binary-classification metrics + frontier comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def binary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute precision / recall / F1 / accuracy on a Yes/No binary task.

    Records with prediction == ``"UNKNOWN"`` are counted under
    ``parse_fail`` and excluded from the metric denominators.
    """
    tp = fp = tn = fn = parse_fail = 0
    for r in records:
        pred, gt = r["prediction"], r["ground_truth"]
        if pred == "UNKNOWN":
            parse_fail += 1
            continue
        if gt == "Yes" and pred == "Yes":
            tp += 1
        elif gt == "No" and pred == "No":
            tn += 1
        elif gt == "No" and pred == "Yes":
            fp += 1
        elif gt == "Yes" and pred == "No":
            fn += 1
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n": n, "parse_fail": parse_fail, "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def frontier_delta(
    overall: dict[str, Any],
    escalation_rate: float,
    frontier_path: str | Path,
) -> dict[str, Any] | None:
    """Compare a run to the no-pretrain Pareto frontier at its esc%.

    Frontier file layout (JSON):
        {"points": [{"esc": 0.0, "acc": ..., "f1": ...}, ...]}

    Linearly interpolates at ``escalation_rate`` and returns
    ``{"baseline_acc": ..., "delta_acc": ..., "baseline_f1": ...,
       "delta_f1": ...}``. Returns ``None`` if the frontier file is
    missing or malformed.
    """
    p = Path(frontier_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text())
    except Exception:
        return None
    points = sorted(blob.get("points", []), key=lambda d: d["esc"])
    if len(points) < 2:
        return None
    e = float(escalation_rate)
    base_acc = _interp(e, points, "acc")
    base_f1 = _interp(e, points, "f1")
    return {
        "frontier_path": str(p),
        "baseline_acc": base_acc,
        "delta_acc": float(overall["accuracy"]) - base_acc,
        "baseline_f1": base_f1,
        "delta_f1": float(overall["f1"]) - base_f1,
    }


def _interp(e: float, points: list[dict[str, Any]], key: str) -> float:
    if e <= points[0]["esc"]:
        return float(points[0][key])
    if e >= points[-1]["esc"]:
        return float(points[-1][key])
    for a, b in zip(points, points[1:]):
        if a["esc"] <= e <= b["esc"]:
            span = b["esc"] - a["esc"]
            t = 0.0 if span == 0 else (e - a["esc"]) / span
            return float(a[key]) * (1.0 - t) + float(b[key]) * t
    return float(points[-1][key])
