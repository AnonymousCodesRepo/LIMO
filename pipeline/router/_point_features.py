"""Per-point feature vector shared by the trained routers.

The 14-d vector below is the point-level feature block used by
`lightgbm_router` (and by `pipeline/label_synth` when shaping offline
rollout features to match). It covers the paper's small-model output
signals (positions 0-7), two cheap length signals (8-9), and the
data-point / query history signals (10-13): escalation counts and
small==large agreement rates for the current document and the current
query, maintained online by the experience retriever and passed in via
``history``.
"""

from __future__ import annotations

import math

import numpy as np

from pipeline.common.types import DataPoint


FEATURE_NAMES = [
    "p_yes",
    "p_no",
    "confidence",
    "margin",
    "entropy_2way",
    "logprob_yes",
    "logprob_no",
    "pred_is_yes",
    "doc_len_log",
    "qdesc_len",
    "doc_esc_norm",       # log(1+n_esc_d)/log(101) for the current document
    "doc_agree_rate",     # Laplace-smoothed small==large rate on this doc
    "query_esc_norm",     # log(1+n_esc_q)/log(101) for the current query
    "query_agree_rate",   # Laplace-smoothed small==large rate on this query
]

POINT_FEATURE_DIM = len(FEATURE_NAMES)

# Neutral history when no counters are available (offline pretraining before
# any labels, or a retriever that does not track history): zero escalations
# observed, agreement rate at the Laplace prior 0.5.
_NEUTRAL_HISTORY = (0.0, 0.5, 0.0, 0.5)


def _featurize(
    point: DataPoint,
    small_features: dict | None,
    history: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    f = small_features or {}
    p_yes = float(f.get("p_yes", 0.5))
    p_no = float(f.get("p_no", 0.5))
    conf = float(f.get("confidence", max(p_yes, p_no)))
    margin = float(f.get("margin", abs(p_yes - p_no)))
    ent = float(f.get("entropy_2way", math.log(2.0)))
    lp_yes = float(f.get("logprob_yes", -10.0))
    lp_no = float(f.get("logprob_no", -10.0))
    pred = f.get("prediction", "UNKNOWN")
    pred_is_yes = 1.0 if pred == "Yes" else (0.0 if pred == "No" else 0.5)
    doc_len_log = math.log(1.0 + len(point.doc_text))
    qdesc_len = float(len(point.query_description))
    d_esc, d_agree, q_esc, q_agree = history or _NEUTRAL_HISTORY
    return np.asarray([
        p_yes, p_no, conf, margin, ent,
        lp_yes, lp_no, pred_is_yes,
        doc_len_log, qdesc_len,
        float(d_esc), float(d_agree), float(q_esc), float(q_agree),
    ], dtype=np.float64)
