"""Prompt templates + JSON parser used by the online experience generator.

These are the canonical templates (also used to build the offline phase_a
experience pool).
"""

from __future__ import annotations

import json
import re
from typing import Any


# ── Small-model explanation prompt (A) ────────────────────────────────────


def small_model_explain_messages(
    document_text: str,
    query_description: str,
    small_prediction: str,
    domain: str = "legal",
) -> list[dict[str, str]]:
    """Ask the small model to rationalize its already-produced answer.

    `domain` names the classification domain so the framing matches the data
    (e.g. "privacy-policy", "cancer-biology", "fact-verification"); defaults
    to "legal".

    The prior verdict is fed back as a premise so we recover the reasoning
    chain that led to it — the 70B later uses that chain to diagnose the
    misstep. It comes from the 3-way small-LLM prompt and is one of
    {"Yes", "No", "Unsure"}; for "Unsure" the model is asked to explain what
    made the document ambiguous rather than commit to a verdict.
    """
    system = (
        f"You are a {domain} document classifier explaining your own prior verdict. "
        "Given a document, a yes/no question, and the verdict you already gave "
        "(\"Yes\", \"No\", or \"Unsure\"), describe the reasoning that led to "
        "that verdict. If your prior verdict was \"Unsure\", explain which "
        "aspects of the document were ambiguous or insufficient — do not now "
        "commit to Yes or No."
    )
    user = (
        f"Question: {query_description}\n\n"
        f"Document:\n{document_text}\n\n"
        f"Your prior verdict: {small_prediction}\n\n"
        "In 1-3 sentences, explain the reasoning that led you to that verdict. "
        "Point to specific phrases or their absence in the document if possible. "
        "Do not re-evaluate or contradict your prior verdict — just explain it. "
        "Do not output a final Yes / No / Unsure line."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── Large-model synthesis prompt (B) ──────────────────────────────────────


_META_SYSTEM = (
    "You are an expert at analyzing LLM classification errors. You are given a "
    "specific case where a small language model got a yes/no document "
    "classification wrong, along with the small model's own step-by-step "
    "reasoning. Your job is to produce a structured experience record that can "
    "later be retrieved to help the small model avoid similar mistakes. You "
    "must output valid JSON and nothing else."
)


def meta_prompt_messages(
    query_name: str,
    query_description: str,
    document_text: str,
    small_reasoning: str,
    small_prediction: str,
    large_prediction: str,
    domain: str = "legal clause",
) -> list[dict[str, str]]:
    """Build the meta-prompt that asks the large LLM to produce an experience.

    `domain` names the classification domain (e.g. "privacy-policy",
    "cancer-biology", "fact-verification") so the 70B frames its diagnosis to
    the actual task; defaults to "legal clause"."""
    user = f"""\
A small language model (Qwen3.5-0.8B) and a large language model
(Llama-3.1-70B) disagreed on a {domain} classification task. Treat the
large model's answer as the reference. Analyze the discrepancy and produce a
reusable experience.

=== Question ===
Query name: {query_name}
Query description: {query_description}

=== Document ===
{document_text}

=== Small model ===
Small model's step-by-step reasoning:
{small_reasoning.strip()}

Small model's final prediction: {small_prediction}

=== Large model (reference) ===
Large model's prediction: {large_prediction}

=== Your task ===
First, internally identify the specific misstep in the small model's reasoning
(e.g. "demanded literal keyword match", "missed a paraphrased obligation",
"confused two legal concepts"). Use that diagnosis to inform — but not appear
in — the lesson you produce. Then output a JSON object with exactly the
following fields:

  "experience_text":        A concise, directly actionable lesson (1-3 sentences,
                            <= 60 words). Phrase it as guidance that can be
                            appended to the small model's system prompt: start
                            with an imperative verb ("Treat ...", "Recognize
                            that ...", "When you see ..., answer Yes ..."). Do
                            NOT mention this specific document, the model
                            names, or the discrepancy itself.

Output ONLY the JSON object, with no surrounding prose, code fences, or
explanation. Use double quotes for all keys and string values."""

    return [
        {"role": "system", "content": _META_SYSTEM},
        {"role": "user", "content": user},
    ]


# ── JSON parser ───────────────────────────────────────────────────────────


META_REQUIRED_FIELDS = {
    "experience_text",
}

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_llm_experience_json(raw: str) -> dict[str, Any]:
    """Parse the large-LLM JSON output into a dict. Raises ValueError on failure.

    Tolerates code fences and leading/trailing prose around the JSON object.
    Missing required fields raise ValueError.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("` \n")
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")
    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode failed: {e}; body={candidate[:200]!r}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")

    missing = META_REQUIRED_FIELDS - obj.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    obj["experience_text"] = str(obj["experience_text"]).strip()
    return obj
