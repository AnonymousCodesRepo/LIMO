"""Extract Yes/No confidence from a chat-completion response with logprobs.

The small model answers with exactly "Yes" or "No". We take the first
decision token (Yes / No and common variants) and, from the top_logprobs
list returned by vLLM's OpenAI-compatible endpoint, derive a 2-way softmax
over P(Yes) and P(No). Confidence is max(p_yes, p_no). On any parse failure
we fall back to `(prediction_from_text, 1.0)`.

Two entry points:
  - `extract_yes_no_confidence(resp)` -> (prediction, confidence, raw_text)
  - `extract_yes_no_features(resp)` -> dict with the full 2-way feature set
    (p_yes, p_no, confidence, margin, entropy_2way, logprob_yes, logprob_no,
    prediction, raw, logprobs_available).
"""

from __future__ import annotations

import math

TRUE_TOKENS = {"yes", "y", "true", "t"}
FALSE_TOKENS = {"no", "n", "false", "f"}

# Neutral filler when logprobs are unavailable, so downstream features stay
# well-defined.
_MISSING_LP = -10.0


def _scan_yes_no_logprobs(logprobs_content) -> tuple[float, float]:
    """Walk the response's token-logprobs list and pull (lp_yes, lp_no) for
    the first decision position. Returns (-inf sentinel, -inf sentinel) if
    nothing decision-like was found.
    """
    lp_yes = -100.0
    lp_no = -100.0
    for token_info in logprobs_content:
        tok = token_info.token.strip().lower()
        is_decision = tok in TRUE_TOKENS or tok in FALSE_TOKENS
        if not is_decision:
            alts = getattr(token_info, "top_logprobs", None) or []
            saw_decision_alt = False
            for alt in alts:
                atok = alt.token.strip().lower()
                if atok in TRUE_TOKENS:
                    lp_yes = max(lp_yes, alt.logprob)
                    saw_decision_alt = True
                elif atok in FALSE_TOKENS:
                    lp_no = max(lp_no, alt.logprob)
                    saw_decision_alt = True
            if saw_decision_alt:
                break
            continue
        if tok in TRUE_TOKENS:
            lp_yes = max(lp_yes, token_info.logprob)
        else:
            lp_no = max(lp_no, token_info.logprob)
        for alt in getattr(token_info, "top_logprobs", None) or []:
            atok = alt.token.strip().lower()
            if atok in TRUE_TOKENS:
                lp_yes = max(lp_yes, alt.logprob)
            elif atok in FALSE_TOKENS:
                lp_no = max(lp_no, alt.logprob)
        break
    return lp_yes, lp_no


def _softmax_2way(lp_yes: float, lp_no: float) -> tuple[float, float]:
    m = max(lp_yes, lp_no)
    e_yes = math.exp(lp_yes - m) if lp_yes > -100.0 else 0.0
    e_no = math.exp(lp_no - m) if lp_no > -100.0 else 0.0
    total = e_yes + e_no or 1.0
    return e_yes / total, e_no / total


def extract_yes_no_features(resp) -> dict:
    """Return a rich per-call feature dict derived from the small model's
    first decision-token logprobs. Keys:

      prediction:        "Yes" | "No" | "UNKNOWN"
      raw:               message content string
      confidence:        max(p_yes, p_no) in [0, 1]
      p_yes, p_no:       2-way softmax probabilities (sum to 1 when available)
      margin:            |p_yes - p_no|
      entropy_2way:      -(p_yes log p_yes + p_no log p_no) nats
      logprob_yes/no:    raw logprobs (or _MISSING_LP = -10.0 if unavailable)
      logprobs_available: bool — False when no decision-token logprobs found
    """
    raw = (resp.choices[0].message.content or "").strip()
    from pipeline.common.prompts import parse_yes_no
    pred_from_text = parse_yes_no(raw)

    logprobs_data = getattr(resp.choices[0], "logprobs", None)
    logprobs_content = (
        getattr(logprobs_data, "content", None) if logprobs_data else None
    )

    def _fallback() -> dict:
        return {
            "prediction": pred_from_text,
            "raw": raw,
            "confidence": 1.0,
            "p_yes": 0.5,
            "p_no": 0.5,
            "margin": 0.0,
            "entropy_2way": math.log(2.0),
            "logprob_yes": _MISSING_LP,
            "logprob_no": _MISSING_LP,
            "logprobs_available": False,
        }

    if not logprobs_content:
        return _fallback()

    lp_yes, lp_no = _scan_yes_no_logprobs(logprobs_content)
    if lp_yes <= -100.0 and lp_no <= -100.0:
        return _fallback()

    p_yes, p_no = _softmax_2way(lp_yes, lp_no)
    pred = "Yes" if p_yes >= p_no else "No"
    conf = max(p_yes, p_no)
    margin = abs(p_yes - p_no)
    ent = 0.0
    if 0.0 < p_yes < 1.0 and 0.0 < p_no < 1.0:
        ent = -(p_yes * math.log(p_yes) + p_no * math.log(p_no))
    return {
        "prediction": pred,
        "raw": raw,
        "confidence": conf,
        "p_yes": p_yes,
        "p_no": p_no,
        "margin": margin,
        "entropy_2way": ent,
        "logprob_yes": lp_yes if lp_yes > -100.0 else _MISSING_LP,
        "logprob_no": lp_no if lp_no > -100.0 else _MISSING_LP,
        "logprobs_available": True,
    }


_TRUE_TOKENS_3W = {"true", "t", "yes", "y"}
_FALSE_TOKENS_3W = {"false", "f", "no", "n"}
# Include the partial-token prefixes the small tokenizer emits for "Unsure":
# it splits "Unsure" into "Un" + "sure" (and "Uncertain" into "Un" +
# "certain"), so the first decision token is "Un" rather than the full word;
# without these prefixes p_unsure stays pinned at 0. "unknown" is treated as
# Unsure (the safe direction: forced escalation).
_UNSURE_TOKENS_3W = {"unsure", "uncertain", "unknown", "un", "uns", "uncert"}


def _scan_3way_logprobs(logprobs_content) -> tuple[float, float, float]:
    """Walk the response's token-logprobs list and pull
    (lp_true, lp_false, lp_unsure) for the first decision position.
    Returns -100.0 sentinel for any class never observed.
    """
    lp_true = -100.0
    lp_false = -100.0
    lp_unsure = -100.0

    def _bucket(tok: str) -> str | None:
        if tok in _TRUE_TOKENS_3W:
            return "true"
        if tok in _FALSE_TOKENS_3W:
            return "false"
        if tok in _UNSURE_TOKENS_3W:
            return "unsure"
        return None

    for token_info in logprobs_content:
        tok = token_info.token.strip().lower()
        cls = _bucket(tok)
        if cls is None:
            saw = False
            for alt in getattr(token_info, "top_logprobs", None) or []:
                acls = _bucket(alt.token.strip().lower())
                if acls is None:
                    continue
                saw = True
                if acls == "true":
                    lp_true = max(lp_true, alt.logprob)
                elif acls == "false":
                    lp_false = max(lp_false, alt.logprob)
                else:
                    lp_unsure = max(lp_unsure, alt.logprob)
            if saw:
                break
            continue
        if cls == "true":
            lp_true = max(lp_true, token_info.logprob)
        elif cls == "false":
            lp_false = max(lp_false, token_info.logprob)
        else:
            lp_unsure = max(lp_unsure, token_info.logprob)
        for alt in getattr(token_info, "top_logprobs", None) or []:
            acls = _bucket(alt.token.strip().lower())
            if acls is None:
                continue
            if acls == "true":
                lp_true = max(lp_true, alt.logprob)
            elif acls == "false":
                lp_false = max(lp_false, alt.logprob)
            else:
                lp_unsure = max(lp_unsure, alt.logprob)
        break
    return lp_true, lp_false, lp_unsure


def _softmax_3way(
    lp_true: float, lp_false: float, lp_unsure: float
) -> tuple[float, float, float]:
    lps = [lp_true, lp_false, lp_unsure]
    m = max(lps)
    exps = [math.exp(lp - m) if lp > -100.0 else 0.0 for lp in lps]
    total = sum(exps) or 1.0
    return exps[0] / total, exps[1] / total, exps[2] / total


def extract_yes_no_features_3way(resp) -> dict:
    """3-way version of `extract_yes_no_features`.

    Same-shaped dict as the 2-way function (plus extra keys), but with
    `"Unsure"` exposed as a first-class top class:

      prediction ∈ {"Yes", "No", "Unsure", "UNKNOWN"} = argmax over
        (p_yes, p_no, p_unsure) when logprobs are available; the parsed
        text class otherwise.

    ``confidence`` is kept as the binary cap ``max(p_yes, p_no)``, NOT the
    top-class probability: when the model puts mass on Unsure the cap drops
    and the confidence-threshold router escalates on top of the explicit
    "prediction == 'Unsure' → escalate" rule.

    Extra fields beyond the 2-way shape: ``p_unsure``, ``logprob_unsure``,
    ``entropy_3way``, ``raw_prediction_3way`` ∈ {Yes, No, Unsure, UNKNOWN}.

    Downstream contract:
      * Few-shot retriever filters to {"Yes","No"} so Unsure points are
        excluded from the demo pool.
      * Experience generator treats Unsure ≠ Yes/No as a discrepancy.
    """
    raw = (resp.choices[0].message.content or "").strip()
    from pipeline.common.prompts import parse_3way
    raw_pred_3w = parse_3way(raw)

    logprobs_data = getattr(resp.choices[0], "logprobs", None)
    logprobs_content = (
        getattr(logprobs_data, "content", None) if logprobs_data else None
    )

    def _fallback() -> dict:
        # Logprobs unavailable: defer to the parsed text class; Unsure is
        # surfaced as-is rather than coerced to Yes/No.
        if raw_pred_3w in ("Yes", "No", "Unsure"):
            pred, conf = raw_pred_3w, 1.0
        else:
            pred, conf = "UNKNOWN", 1.0
        return {
            "prediction": pred,
            "raw": raw,
            "confidence": conf,
            "p_yes": 1.0 if pred == "Yes" else 0.0,
            "p_no": 1.0 if pred == "No" else 0.0,
            "p_unsure": 1.0 if pred == "Unsure" else 0.0,
            "margin": 0.0,
            "entropy_2way": math.log(2.0),
            "entropy_3way": math.log(3.0),
            "logprob_yes": _MISSING_LP,
            "logprob_no": _MISSING_LP,
            "logprob_unsure": _MISSING_LP,
            "logprobs_available": False,
            "raw_prediction_3way": raw_pred_3w,
        }

    if not logprobs_content:
        return _fallback()

    lp_t, lp_f, lp_u = _scan_3way_logprobs(logprobs_content)
    if lp_t <= -100.0 and lp_f <= -100.0 and lp_u <= -100.0:
        return _fallback()

    p_yes, p_no, p_unsure = _softmax_3way(lp_t, lp_f, lp_u)
    # Argmax over all three. When Unsure is the argmax it becomes a
    # first-class prediction (forced escalation). Tie-break: Yes ≥ No > Unsure.
    if p_yes >= p_no and p_yes >= p_unsure:
        pred = "Yes"
    elif p_no >= p_yes and p_no >= p_unsure:
        pred = "No"
    else:
        pred = "Unsure"
    # `confidence` is the binary cap (max p_yes, p_no) — see docstring.
    conf = max(p_yes, p_no)
    margin = abs(p_yes - p_no)
    ent2 = 0.0
    s = p_yes + p_no
    if s > 0:
        py, pn = p_yes / s, p_no / s
        if 0.0 < py < 1.0:
            ent2 = -(py * math.log(py) + pn * math.log(pn))
    ent3 = 0.0
    for p in (p_yes, p_no, p_unsure):
        if p > 1e-10:
            ent3 -= p * math.log(p)

    return {
        "prediction": pred,
        "raw": raw,
        "confidence": conf,
        "p_yes": p_yes,
        "p_no": p_no,
        "p_unsure": p_unsure,
        "margin": margin,
        "entropy_2way": ent2,
        "entropy_3way": ent3,
        "logprob_yes": lp_t if lp_t > -100.0 else _MISSING_LP,
        "logprob_no": lp_f if lp_f > -100.0 else _MISSING_LP,
        "logprob_unsure": lp_u if lp_u > -100.0 else _MISSING_LP,
        "logprobs_available": True,
        "raw_prediction_3way": raw_pred_3w,
    }


def extract_yes_no_confidence(resp) -> tuple[str, float, str]:
    """Return (prediction, confidence, raw_text).

    prediction ∈ {"Yes", "No", "UNKNOWN"}; confidence ∈ [0, 1].
    When logprobs are unavailable confidence falls back to 1.0 (i.e. "no signal
    to escalate on"). Callers should treat 1.0 as "no meaningful confidence
    extracted" if they want stricter behavior.
    """
    f = extract_yes_no_features(resp)
    return f["prediction"], f["confidence"], f["raw"]
