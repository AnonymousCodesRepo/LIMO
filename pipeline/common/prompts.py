"""Pipeline-owned prompt builders.

Two builders with strict role separation enforced by the runner:

  * ``build_replay_messages`` — 2-way (Yes/No) prompt. Used by the LARGE
    LLM on escalation, with no retrieved experiences and no few-shot demos
    (the runner passes both as ``None``). The large model is the cascade's
    reference oracle and must stay clean.

  * ``build_replay_messages_3way`` — 3-way (True/False/Unsure) prompt. Used
    by the SMALL LLM. Accepts retrieved experiences and few-shot demos;
    "Unsure" is a first-class output that propagates to the cascade
    decision and downstream stages.

Few-shot demos include only the document and its predicted label.
"""

from __future__ import annotations

from pipeline.common.types import Experience, FewShotDemo


FEWSHOT_DOC_EXCERPT_CHARS = 2000000


# Default base system sentence for the 2-way (Yes/No) prompt. This exact
# wording must stay fixed so the precomputed 70B cache is bit-exact against
# live calls. Other datasets pass a domain-neutral base via ``system_base``
# (wired through ``DatasetSpec.large_system_prompt``).
DEFAULT_LARGE_SYSTEM = (
    "You are a careful legal document classifier. Given a document and a "
    "yes/no question, output only `Yes` or `No`."
)


# ---------------------------------------------------------------------------
# Small-model system-prompt styles.
#
# The SMALL model's system prompt is selected by (mode, style). "default"
# reproduces the baseline wording verbatim so baselines stay bit-exact.
# "skeptical" is an anti-Yes-bias fact-checking framing for FEVER, where the
# small model otherwise answers "Yes" to nearly every claim.
# ---------------------------------------------------------------------------
_SMALL_3WAY_DEFAULT = (
    "You are a precise document classifier. "
    "Given a condition and a document, determine whether the document "
    "satisfies the condition. "
    "If you are uncertain, say so rather than guessing.\n"
    "Respond with exactly one word: True, False, or Unsure."
)
_SMALL_3WAY_SKEPTICAL = (
    "You are a rigorous fact-checker. You are given a claim and accompanying "
    "documents. Many claims are subtly FALSE, exaggerated, or only partially "
    "supported. Answer True ONLY if the documents EXPLICITLY and FULLY support "
    "the whole claim. If the documents do not mention the claim, contradict it, "
    "or give only partial or loosely related information, answer False. Do not "
    "assume a claim is true just because it sounds plausible.\n"
    "Respond with exactly one word: True, False, or Unsure."
)
_SMALL_2WAY_SKEPTICAL = (
    "You are a rigorous fact-checker. You are given a claim and accompanying "
    "documents. Many claims are subtly FALSE, exaggerated, or only partially "
    "supported. Answer True ONLY if the documents EXPLICITLY and FULLY support "
    "the whole claim; otherwise answer False. Do not assume a claim is true just "
    "because it sounds plausible."
)

SMALL_3WAY_SYSTEMS = {
    "default": _SMALL_3WAY_DEFAULT,
    "skeptical": _SMALL_3WAY_SKEPTICAL,
}
# 2-way small systems. "default" reuses the shared legal base; the runner
# passes the chosen entry as ``system_base`` to build_replay_messages, so the
# large-model path is unaffected.
SMALL_2WAY_SYSTEMS = {
    "default": DEFAULT_LARGE_SYSTEM,
    "skeptical": _SMALL_2WAY_SKEPTICAL,
}


def build_replay_messages(
    document_text: str,
    query_description: str,
    retrieved_experiences: list[Experience] | None = None,
    fewshot_demos: list[FewShotDemo] | None = None,
    *,
    system_base: str = DEFAULT_LARGE_SYSTEM,
) -> list[dict[str, str]]:
    """Return the chat-completions messages for a small-LLM replay call."""
    system_parts = [system_base]
    if retrieved_experiences:
        exp_block = "\n".join(
            f"- {e.experience_text}"
            for e in retrieved_experiences
            if e.experience_text
        )
        if exp_block:
            system_parts.append(
                "Guidance derived from past mistakes on similar questions:\n"
                f"{exp_block}"
            )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]

    # Few-shot demos as prior user/assistant turns.
    # Each demo gives the document only; the assistant turn gives the label.
    for demo in (fewshot_demos or []):
        excerpt = demo.doc_text[:FEWSHOT_DOC_EXCERPT_CHARS]
        messages.append({
            "role": "user",
            "content": (
                f"Document:\n{excerpt}\n\n"
                "Answer with exactly one word: Yes or No."
            ),
        })
        messages.append({"role": "assistant", "content": demo.prediction})

    messages.append({
        "role": "user",
        "content": (
            f"Question: {query_description}\n\n"
            f"Document:\n{document_text}\n\n"
            "Answer with exactly one word: Yes or No."
        ),
    })
    return messages


def build_replay_messages_3way(
    document_text: str,
    query_description: str,
    retrieved_experiences: list[Experience] | None = None,
    fewshot_demos: list[FewShotDemo] | None = None,
    *,
    system_style: str = "default",
) -> list[dict[str, str]]:
    """3-way (True / False / Unsure) version of `build_replay_messages`.

    Same retrieval / few-shot slots as the 2-way variant; only the system
    prompt and the closing instruction differ. ``system_style`` selects the
    system prompt from ``SMALL_3WAY_SYSTEMS``.
    """
    system_parts = [SMALL_3WAY_SYSTEMS[system_style]]
    if retrieved_experiences:
        exp_block = "\n".join(
            f"- {e.experience_text}"
            for e in retrieved_experiences
            if e.experience_text
        )
        if exp_block:
            system_parts.append(
                "Guidance derived from past mistakes on similar questions:\n"
                f"{exp_block}"
            )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]

    # Few-shot demos: same as 2-way but with True/False/Unsure phrasing.
    # Demo labels come in as {Yes, No}; map to {True, False} so the assistant
    # turn matches the 3-way response space.
    for demo in (fewshot_demos or []):
        excerpt = demo.doc_text[:FEWSHOT_DOC_EXCERPT_CHARS]
        messages.append({
            "role": "user",
            "content": (
                f"Document:\n{excerpt}\n\n"
                "Answer with exactly one word: True, False, or Unsure."
            ),
        })
        label = demo.prediction
        if label == "Yes":
            label = "True"
        elif label == "No":
            label = "False"
        messages.append({"role": "assistant", "content": label})

    messages.append({
        "role": "user",
        "content": (
            f"Condition: {query_description}\n\n"
            f"Document:\n{document_text}\n\n"
            "Answer (True, False, or Unsure):"
        ),
    })
    return messages


def parse_yes_no(raw: str) -> str:
    if not raw:
        return "UNKNOWN"
    text = raw.strip().split("\n")[0].strip().lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    for tok in text.replace(",", " ").replace(".", " ").split():
        if tok in {"yes", "yes."}:
            return "Yes"
        if tok in {"no", "no."}:
            return "No"
    return "UNKNOWN"


def parse_3way(raw: str) -> str:
    """Parse a 3-way response into one of {Yes, No, Unsure, UNKNOWN}.

    Maps True→Yes and False→No to keep the rest of the pipeline (which
    speaks Yes/No) unchanged.
    """
    if not raw:
        return "UNKNOWN"
    text = raw.strip().split("\n")[0].strip().lower()
    if text.startswith("true"):
        return "Yes"
    if text.startswith("false"):
        return "No"
    if text.startswith("unsure"):
        return "Unsure"
    for tok in text.replace(",", " ").replace(".", " ").split():
        if tok in {"true", "true."}:
            return "Yes"
        if tok in {"false", "false."}:
            return "No"
        if tok in {"unsure", "unsure."}:
            return "Unsure"
    return "UNKNOWN"
