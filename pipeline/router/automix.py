"""AutoMix self-verification router (Aggarwal et al., NeurIPS 2024).

Reproduces the *threshold* variant of AutoMix (Automatic Mixing of language
models) as a cascade-routing baseline. The small model answers the predicate
(2-way Yes/No, from the runner's normal small call), then the SAME small
model is asked ``k`` more times, at a higher sampling temperature, to verify
whether that answer is correct given the document and predicate. The fraction
of verifications saying "Correct" is a consistency score in [0, 1]; the
router escalates when the score is at or below ``threshold``.

Faithful to the official implementation (github.com/automix-llm/automix,
``colabs/Step2_SelfVerify.ipynb``):

  * the verbatim 4-shot verification prompt (``VERIFIER_PROMPT`` below);
  * the same fraction-correct aggregation: decided = samples that emit a
    verdict; score = correct / decided, or 0.0 when nothing parses (treated
    as "no confidence" → escalate);
  * the same ``score <= threshold → escalate`` rule.

The POMDP meta-verifier is intentionally NOT used; this router is stateless.

Cost: the ``k`` verification generations are extra small-model calls; their
token usage and call count are returned on the ``EscalationDecision`` so the
runner folds them into the cost accounting.

Entry points (Router contract):
    route(state, point) -> "small"
    should_escalate(state, point, small_pred, small_raw, small_confidence)
        -> EscalationDecision
"""

from __future__ import annotations

from typing import Literal

from openai import OpenAI

from pipeline.common.types import DataPoint, RunState

from ._base import BaseRouter, EscalationDecision


# Verbatim 4-shot self-verification prompt from the official AutoMix repo
# (colabs/Step2_SelfVerify.ipynb). Only {context}, {question}, and
# {generated_answer} are filled; the model emits "Evaluation:" reasoning then
# the fixed phrase "The AI generated answer is Correct." / "... Incorrect.".
VERIFIER_PROMPT = """Context: The manuscript, discovered in 1980 in a dusty attic, turned out to be a lost work of Shakespeare.

Question: Whose lost work was discovered in a dusty attic in 1980?

AI Generated Answer: Shakespeare

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context specifically mentions that a lost work of Shakespeare was discovered in 1980 in a dusty attic.

Verification Decision: The AI generated answer is Correct.

---

Context: The celestial event, known as the Pink Moon, is unique to the month of April and has cultural significance in many indigenous tribes.

Question: In which month does the celestial event, the Pink Moon, occur?

AI Generated Answer: July

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context clearly states that the Pink Moon is unique to the month of April.

Verification Decision: The AI generated answer is Incorrect.

---

Context: The Mona Lisa, housed in the Louvre Museum, is believed to be a portrait of Lisa Gherardini, painted by Leonardo da Vinci in the early 16th century.

Question: Who is believed to have painted the Mona Lisa in the early 16th century?

AI Generated Answer: Vincent van Gogh

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context specifies that the Mona Lisa was painted by Leonardo da Vinci in the early 16th century.

Verification Decision: The AI generated answer is Incorrect.

---

Context: The planet Kepler-442b, located 1,100 light-years away, is one of the most Earth-like planets ever discovered, having a similar size and orbiting within its star's habitable zone.

Question: How far away is the planet Kepler-442b?

AI Generated Answer: 1,100 light-years

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation: The context states that Kepler-442b is located 1,100 light-years away.

Verification Decision: The AI generated answer is Correct.

---

Context: {context}

Question: {question}

AI Generated Answer: {generated_answer}

Instruction: Your task is to evaluate if the AI Generated Answer is correct, based on the provided context and question. Provide the judgement and reasoning for each case. Choose between Correct or Incorrect.

Evaluation:"""

# Substrings the official ``compute_fraction_correct`` looks for (lower-cased).
_DECISION_MARK = "the ai generated answer is"
_CORRECT_MARK = "the ai generated answer is correct"

# Split the verbatim prompt into its 4 demonstration blocks + the trailing
# query template (separated by the "---" delimiter), so a caller can reduce
# the number of few-shot demonstrations for long-document datasets.
_VERIFY_BLOCKS = VERIFIER_PROMPT.split("\n\n---\n\n")
_VERIFY_DEMOS = _VERIFY_BLOCKS[:-1]            # the 4 demonstration blocks
_VERIFY_QUERY_TEMPLATE = _VERIFY_BLOCKS[-1]    # "Context: {context}...Evaluation:"


def _build_verify_prompt(
    n_demos: int, context: str, question: str, generated_answer: str
) -> str:
    """Assemble the verification prompt with the first ``n_demos`` (0..4)
    demonstrations followed by the filled query block."""
    demos = _VERIFY_DEMOS[: max(0, n_demos)]
    query = _VERIFY_QUERY_TEMPLATE.format(
        context=context, question=question, generated_answer=generated_answer
    )
    return "\n\n---\n\n".join(demos + [query])


class AutoMixRouter(BaseRouter):
    """Escalate when the small model's self-verification confidence is low.

    The small answer the runner produced is verified ``k`` times; the
    fraction of "Correct" verdicts is thresholded. ``small_client`` /
    ``small_model`` are injected by the builder (they are runtime objects,
    not YAML config).
    """

    def __init__(
        self,
        *,
        small_client: OpenAI,
        small_model: str,
        threshold: float = 0.5,
        k: int = 8,
        verify_temperature: float = 1.0,
        verify_max_tokens: int = 250,
        verify_stop: str = "---",
        request_timeout: float = 60.0,
        max_doc_chars: int | None = None,
        n_verify_demos: int = 4,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if int(k) < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0 <= int(n_verify_demos) <= len(_VERIFY_DEMOS):
            raise ValueError(
                f"n_verify_demos must be in [0, {len(_VERIFY_DEMOS)}], "
                f"got {n_verify_demos}"
            )
        self.small_client = small_client
        self.small_model = small_model
        self.threshold = float(threshold)
        self.k = int(k)
        self.verify_temperature = float(verify_temperature)
        self.verify_max_tokens = int(verify_max_tokens)
        self.verify_stop = verify_stop
        self.request_timeout = float(request_timeout)
        self.n_verify_demos = int(n_verify_demos)
        # Optional left-truncation of the document text (chars) to keep the
        # verification prompt within the small model's context. None = no
        # truncation. AutoMix truncates the filled prompt from the left; we
        # truncate the document, keeping the 4-shot demonstrations intact.
        self.max_doc_chars = max_doc_chars

    def route(
        self, state: RunState, point: DataPoint
    ) -> Literal["small", "large"]:
        return "small"

    def should_escalate(
        self,
        state: RunState,
        point: DataPoint,
        small_pred: str,
        small_raw: str,
        small_confidence: float,
        **_kwargs,  # accept (and ignore) richer signals like small_features
    ) -> EscalationDecision:
        # No parseable small answer to verify (parse failure / unexpected
        # token under the 2-way prompt) → escalate, no verification issued.
        if small_pred not in ("Yes", "No"):
            return EscalationDecision(escalate=True)

        doc_text = point.doc_text
        if self.max_doc_chars is not None and len(doc_text) > self.max_doc_chars:
            doc_text = doc_text[-self.max_doc_chars:]
        prompt = _build_verify_prompt(
            self.n_verify_demos,
            context=doc_text,
            question=point.query_description,
            generated_answer=small_pred,  # "Yes" / "No"
        )
        score, calls, pt, ct = self._verify(prompt)
        return EscalationDecision(
            escalate=(score <= self.threshold),
            verify_small_calls=calls,
            verify_small_prompt_tokens=pt,
            verify_small_completion_tokens=ct,
            verify_score=score,
        )

    def _verify(self, prompt: str) -> tuple[float, int, int, int]:
        """Run k verification samples and return
        (consistency_score, n_calls, prompt_tokens, completion_tokens).

        One chat request with ``n=k`` (matches AutoMix's batched call: the
        prompt is billed once, the k completions are billed individually).
        On any error the answer is treated as having no verification support
        (score 0.0 → escalate), and the k calls are still counted so the
        cost reflects the issued requests.
        """
        try:
            resp = self.small_client.with_options(
                timeout=self.request_timeout
            ).chat.completions.create(
                model=self.small_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.verify_temperature,
                max_tokens=self.verify_max_tokens,
                n=self.k,
                stop=[self.verify_stop],
            )
        except Exception:
            return 0.0, self.k, 0, 0

        texts = [(c.message.content or "").lower() for c in resp.choices]
        decided = sum(1 for t in texts if _DECISION_MARK in t)
        correct = sum(1 for t in texts if _CORRECT_MARK in t)
        score = (correct / decided) if decided else 0.0

        usage = getattr(resp, "usage", None)
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        return score, self.k, pt, ct
