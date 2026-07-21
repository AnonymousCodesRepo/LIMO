"""Route the upstream `task_cascades` package onto the local vLLM / mpnet stack.

Zero-edit port: upstream files stay byte-identical. Everything is done by
replacing litellm module attributes and mutating upstream module dicts at
runtime (see tools/task_cascades/README.md for the full port notes).

Import-order contract
---------------------
``init()`` MUST run before importing ANY ``task_cascades`` module: upstream
binds ``from litellm import completion, embedding, completion_cost`` at module
import time, so the replacement has to be in place first. ``init()`` raises if
an upstream module is already imported.

Model roles
-----------
  proxy  "gpt-4o-mini"             -> MOP_SMALL_MODEL @ MOP_SMALL_URL (Qwen3.5-0.8B)
  oracle "gpt-4o"                  -> MOP_LARGE_MODEL @ MOP_LARGE_URL (Llama-3.1-70B)
  agent  "o1-mini"                 -> large model (all agent models use the 70B)
  ranges "gpt-4.1"                 -> large model (line-range extraction, json_object)
  embed  "text-embedding-3-small"  -> POST MOP_EMBED_URL/embed (custom mpnet endpoint)

Cost accounting
---------------
DeepInfra prices imported from ``mop.llm`` (single source of truth).
``cache_read_input_token_cost`` is set EQUAL to the full input price: all
tokens are charged at full price (no prefix-cache discount), so upstream's
marginal-cost machinery degenerates to full-price accounting for both the
greedy optimizer and the reported costs.

Oracle answers for the ORIGINAL operation are served from the cached 70B
zero-shot predictions (dev labeling counts as one 70B call, bit-identical to
the zs-70B anchor), charged at the clean 2-way prompt's token count.
Everything else (line ranges, failure-case snippets, agent, fractional-doc
evals) is live.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field

import litellm
import requests

# ── configuration (portable-deployment env vars) ─────────────────

SMALL_URL = os.environ.get("MOP_SMALL_URL", "http://localhost:8105/v1")
SMALL_MODEL = os.environ.get(
    "MOP_SMALL_MODEL",
    "Qwen3.5-0.8B",
)
LARGE_URL = os.environ.get("MOP_LARGE_URL", "http://localhost:8102/v1")
LARGE_MODEL = os.environ.get(
    "MOP_LARGE_MODEL",
    "Meta-Llama-3.1-70B-Instruct",
)
EMBED_URL = os.environ.get("MOP_EMBED_URL", "http://localhost:8200")

# Qwen3.5 emits a <think> block by default; upstream reads the FIRST output
# token as the True/False answer, so thinking must be off.
_SMALL_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# Upstream hardcodes timeout=10 for predictor calls; a 70B full-contract call
# routinely exceeds that, litellm would retry twice and process_doc would then
# SWALLOW the exception and return prediction=0/confidence=0, silently
# corrupting the run. The router therefore owns the timeouts.
_SMALL_TIMEOUT = 120
_LARGE_TIMEOUT = 600

# Escape hatch if the deployed vLLM rejects response_format json_object.
_STRIP_JSON_MODE = os.environ.get("MOP_TC_STRIP_JSON_MODE", "") == "1"


@dataclass(frozen=True)
class _Route:
    served: str        # served-model-name on the vLLM endpoint
    api_base: str
    timeout: int
    extra_body: dict | None = None
    is_large: bool = False


_ROUTES: dict[str, _Route] = {
    "gpt-4o-mini": _Route(SMALL_MODEL, SMALL_URL, _SMALL_TIMEOUT, _SMALL_EXTRA_BODY),
    "gpt-4o": _Route(LARGE_MODEL, LARGE_URL, _LARGE_TIMEOUT, is_large=True),
    "gpt-4.1": _Route(LARGE_MODEL, LARGE_URL, _LARGE_TIMEOUT, is_large=True),
    "o1-mini": _Route(LARGE_MODEL, LARGE_URL, _LARGE_TIMEOUT, is_large=True),
}

# ── call log (per-phase cost accounting) ─────────────────────────

_LOG_LOCK = threading.Lock()
CALL_LOG: list[dict] = []
_PHASE = "unphased"


def set_phase(phase: str) -> None:
    global _PHASE
    _PHASE = phase


def drain_call_log() -> list[dict]:
    with _LOG_LOCK:
        out = list(CALL_LOG)
        CALL_LOG.clear()
    return out


def _log_call(requested: str, kind: str, ptok: int, ctok: int,
              cost: float, litellm_cache_hit: bool = False) -> None:
    with _LOG_LOCK:
        CALL_LOG.append({
            "phase": _PHASE,
            "requested_model": requested,
            "kind": kind,  # live | oracle_cache | embed
            "prompt_tokens": int(ptok),
            "completion_tokens": int(ctok),
            "cost_usd": float(cost),
            "litellm_cache_hit": bool(litellm_cache_hit),
            "ts": time.time(),
        })


# ── pricing ──────────────────────────────────────────────────────

def _register_prices() -> None:
    """Mutate litellm.model_cost in place (upstream holds a reference to it).

    Keys cover every name a cost lookup can see: the routed aliases
    ("gpt-4o", ...) used by upstream's cost_given_token_breakdown, plus the
    served names / basenames vLLM may echo back in response.model.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mop.llm import (
        LARGE_COST_PER_INPUT_TOKEN, LARGE_COST_PER_OUTPUT_TOKEN,
        SMALL_COST_PER_INPUT_TOKEN, SMALL_COST_PER_OUTPUT_TOKEN,
    )

    def entry(cin: float, cout: float) -> dict:
        return {
            "input_cost_per_token": cin,
            "output_cost_per_token": cout,
            # full price for "cached" tokens => no prefix-cache discount
            "cache_read_input_token_cost": cin,
        }

    small = entry(SMALL_COST_PER_INPUT_TOKEN, SMALL_COST_PER_OUTPUT_TOKEN)
    large = entry(LARGE_COST_PER_INPUT_TOKEN, LARGE_COST_PER_OUTPUT_TOKEN)

    keys: dict[str, dict] = {"gpt-4o-mini": small, "gpt-4o": large,
                             "gpt-4.1": large, "o1-mini": large}
    for served, e in ((SMALL_MODEL, small), (LARGE_MODEL, large)):
        for k in (served, os.path.basename(served),
                  f"openai/{served}", f"openai/{os.path.basename(served)}"):
            keys[k] = e
    for k, e in keys.items():
        cur = dict(litellm.model_cost.get(k, {}))
        cur.update(e)
        litellm.model_cost[k] = cur


def _price_of(requested: str) -> tuple[float, float]:
    e = litellm.model_cost[requested]
    return e["input_cost_per_token"], e["output_cost_per_token"]


# ── cached-oracle plumbing ───────────────────────────────────────

class _Usage:
    """Supports both attribute and item access, like litellm's Usage."""

    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def __getitem__(self, key: str):
        return getattr(self, key)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


# sha1(final prompt string) -> (pred_int, prompt_tokens, completion_tokens)
_ORACLE_ANSWERS: dict[str, tuple[int, int, int]] = {}


def register_oracle_answer(final_prompt: str, pred: int,
                           prompt_tokens: int, completion_tokens: int) -> None:
    _ORACLE_ANSWERS[_sha1(final_prompt)] = (
        int(pred), int(prompt_tokens), int(completion_tokens))


def oracle_cache_size() -> int:
    return len(_ORACLE_ANSWERS)


def _fake_oracle_response(pred: int, ptok: int, ctok: int):
    content = "True" if pred == 1 else "False"
    choice = _Obj(
        message=_Obj(content=content),
        logprobs={"content": [{"top_logprobs": [
            {"token": content, "logprob": 0.0}]}]},
    )
    return _Obj(model="gpt-4o", usage=_Usage(ptok, ctok),
                choices=[choice], _mop_fake=True)


# ── routed litellm entry points ──────────────────────────────────

_real_completion = litellm.completion
_real_embedding = litellm.embedding
_real_completion_cost = litellm.completion_cost


def _plain_logprobs(lp):
    """Normalize litellm's typed logprob objects into the plain dicts upstream
    indexes (`logprobs['content'][0]['top_logprobs'][i]['token']`). litellm
    1.90 returns pydantic TopLogprob objects, which are not subscriptable."""
    if lp is None:
        return None

    def get(o, k):
        return o[k] if isinstance(o, dict) else getattr(o, k, None)

    content = get(lp, "content")
    if content is None:
        return lp
    return {"content": [
        {"top_logprobs": [
            {"token": get(t, "token"), "logprob": get(t, "logprob")}
            for t in (get(item, "top_logprobs") or [])
        ]}
        for item in content
    ]}


def _routed_completion(*args, **kwargs):
    assert not args, "adapter expects keyword-only litellm.completion calls"
    requested = kwargs.pop("model")
    route = _ROUTES.get(requested)
    if route is None:
        raise RuntimeError(f"unrouted model {requested!r} — extend _ROUTES")

    messages = kwargs["messages"]
    if route.is_large and len(messages) == 1:
        hit = _ORACLE_ANSWERS.get(_sha1(messages[0]["content"]))
        if hit is not None:
            pred, ptok, ctok = hit
            cin, cout = _price_of("gpt-4o")
            _log_call(requested, "oracle_cache", ptok, ctok,
                      ptok * cin + ctok * cout)
            return _fake_oracle_response(pred, ptok, ctok)

    kwargs["timeout"] = route.timeout
    if route.extra_body:
        kwargs["extra_body"] = {**route.extra_body,
                                **kwargs.get("extra_body", {})}
    if _STRIP_JSON_MODE:
        kwargs.pop("response_format", None)
    if "max_completion_tokens" not in kwargs and "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 4096

    resp = _real_completion(
        model=f"openai/{route.served}", api_base=route.api_base,
        api_key="dummy", **kwargs)

    usage = resp.usage
    cin, cout = _price_of(requested)
    cache_hit = bool(getattr(resp, "_hidden_params", {}).get("cache_hit"))
    _log_call(requested, "live",
              usage["prompt_tokens"], usage["completion_tokens"],
              usage["prompt_tokens"] * cin + usage["completion_tokens"] * cout,
              litellm_cache_hit=cache_hit)
    # Make cost_of_completion's model_cost[response.model] lookup safe no
    # matter what name the server echoes back.
    resp.model = requested
    if kwargs.get("logprobs"):
        resp.choices[0].logprobs = _plain_logprobs(resp.choices[0].logprobs)
    return resp


def _routed_embedding(*args, **kwargs):
    assert not args, "adapter expects keyword-only litellm.embedding calls"
    texts = list(kwargs["input"])
    r = requests.post(f"{EMBED_URL.rstrip('/')}/embed",
                      json={"texts": texts}, timeout=300)
    r.raise_for_status()
    embs = r.json()["embeddings"]
    _log_call("text-embedding-3-small", "embed", 0, 0, 0.0)
    return _Obj(data=[{"embedding": e} for e in embs], _mop_fake=True)


def _routed_completion_cost(response, *args, **kwargs):
    if getattr(response, "_mop_fake", False):
        return 0.0
    return _real_completion_cost(response, *args, **kwargs)


# ── upstream registries ──────────────────────────────────────────

# task_name -> (df, documents) consumed by the wrapped load_dataset
_DATASETS: dict[str, tuple] = {}


def register_dataframe(task: str, df, documents: list[str]) -> None:
    _DATASETS[task] = (df, documents)


def register_task(task: str, prefix: str, instruction: str, suffix: str) -> None:
    """Inject a per-query binary task into upstream's prompt registries."""
    from task_cascades.predictors import predictors as P

    P.TASK_INSTRUCTIONS[task] = instruction
    P.PROMPT_PREFIX_SUFFIX_DICT[task] = (prefix, suffix)
    P.TASK_PROMPT_DICT[task] = f"{prefix}\n\n{instruction}\n\n{suffix}"
    P.binary_tasks.append(task)
    P.PROMPT_TO_TASK_TYPE_DICT[task] = "binary"
    P.PROMPT_TO_CLASSES_DICT[task] = [0, 1]
    return P.TASK_PROMPT_DICT[task]


def init() -> None:
    """Patch litellm, then bend the upstream package onto our stack."""
    already = [m for m in sys.modules if m.startswith("task_cascades")]
    if already:
        raise RuntimeError(
            f"init() must run before importing task_cascades (saw {already})")

    litellm.completion = _routed_completion
    litellm.embedding = _routed_embedding
    litellm.completion_cost = _routed_completion_cost
    _register_prices()

    # Disable the 150-sample dev floor; the driver passes the exact dev
    # fraction via ExperimentRunner(train_split=...).
    from task_cascades.config import consts
    consts.MIN_TRAINING_SAMPLES = 0

    # Two-model setting exactly as the paper: trim the predictor pool so
    # candidate generation only pairs surrogates with {proxy, oracle}.
    from task_cascades.predictors import predictors as P
    for k in list(P.PREDICTORS):
        if k not in ("gpt-4o", "gpt-4o-mini"):
            del P.PREDICTORS[k]

    # Give each worker its own litellm disk cache when sharding: upstream
    # points litellm.cache at CWD/.litellm_cache at import; concurrent shards
    # sharing one sqlite log noisy "unable to open database file" contention.
    cache_dir = os.environ.get("MOP_TC_CACHE_DIR")
    if cache_dir:
        litellm.cache = litellm.Cache(type="disk", disk_cache_dir=cache_dir)

    # Route load_dataset through our registry BEFORE experiment_runner binds it.
    from task_cascades.data import create_dfs
    _orig_load = create_dfs.load_dataset

    def _load_dataset(task: str):
        if task in _DATASETS:
            df, documents = _DATASETS[task]
            return df.copy(), list(documents)
        return _orig_load(task)

    create_dfs.load_dataset = _load_dataset
