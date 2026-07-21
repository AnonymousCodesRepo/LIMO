"""Track A profiling: run KV-cache-compressed physical operators (kvpress
ExpectedAttention) over the ladder inputs and record per-(doc, query) log-odds.

Faithful to Stretto's offline phase: the DOCUMENT part of the prompt is prefilled
under the (query-agnostic) press and the compressed cache is REUSED across all
queries of that document (kvpress truncate-back pattern); the query/instruction
part is decoded afterwards without compression. The 3-way prompt wording and the
decision-token logprob extraction match the small-model 3-way confidence collector.

IMPORTANT: rung log-odds are only compared WITHIN this script's outputs. The r=0.0
rung is profiled through this same code path (never substituted from the vLLM
collector caches) so rung deltas measure compression, not prompt-format drift.

Position bookkeeping (kvpress semantics):
  * position_ids continue from the ORIGINAL context length (RoPE positions are
    preserved for the kept tokens),
  * cache_position continues from the COMPRESSED cache length (buffer space, used
    for causal-mask construction). Conflating the two breaks causality.

Runs on a GPU node inside the pytorch container with the kvpress venv:
  python tools/stretto/profile_ladder.py --model-tag qwen08b \
      --model-path /path/to/Qwen3.5-0.8B \
      --ratios 0.8,0.5,0.0 --scope full --datasets opp115,contract \
      --input-dir tools/stretto/ladder_inputs --out-dir ladder_out

Output: {out-dir}/{model_tag}_r{ratio}_{ds}.csv with columns
  document_id, document_name, query_name, prediction, p_true, p_false, p_unsure,
  confidence, entropy, logprob_true, logprob_false, logprob_unsure,
  ctx_tokens, kept_tokens, q_tokens, raw_response
Resumable: torn tails are repaired, done rows skipped, header schema asserted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import ExpectedAttentionPress

SYSTEM_PROMPT = (
    "You are a precise data filtering assistant. "
    "Determine whether the given document satisfies the condition. "
    "If you are uncertain, it is better to say so rather than guessing.\n"
    'Respond with ONLY one of:\n'
    '  {"match": true}\n'
    '  {"match": false}\n'
    '  {"match": "unsure"}'
)

TRUE_TOKENS = {"true", "true}"}
FALSE_TOKENS = {"false", "false}"}
UNSURE_TOKENS = {"unsure", '"unsure"', 'unsure"', '"unsure', "unsure}"}

SPLIT_SENTINEL = "<<<KVQ_SPLIT_7f3a>>>"
MAX_NEW_TOKENS = 12
MAX_CTX_TOKENS = 16384
LLAMA_DATE = "26 Jul 2024"   # pin the Llama-3.1 template date -> deterministic renders

FIELDNAMES = ["document_id", "document_name", "query_name", "prediction",
              "p_true", "p_false", "p_unsure", "confidence", "entropy",
              "logprob_true", "logprob_false", "logprob_unsure",
              "ctx_tokens", "kept_tokens", "q_tokens", "raw_response"]


def _class_token_ids(tok) -> dict[str, list[int]]:
    """Single-token ids whose decoded form falls in each decision class."""
    out = {"true": set(), "false": set(), "unsure": set()}
    variants = {"true": TRUE_TOKENS, "false": FALSE_TOKENS, "unsure": UNSURE_TOKENS}
    for cls, vs in variants.items():
        for v in vs:
            for prefix in ("", " "):
                ids = tok.encode(prefix + v, add_special_tokens=False)
                if len(ids) == 1:
                    out[cls].add(ids[0])
    return {k: sorted(v) for k, v in out.items()}


def _render_split_prompt(tok, doc_text: str, qdesc: str, model_tag: str):
    """Chat-templated prompt split into (context_str, question_str): context ends
    right after the document; the condition/instruction is the question part."""
    user = f"Document:\n{doc_text}{SPLIT_SENTINEL}\n\nCondition: {qdesc}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    kwargs = {}
    if model_tag.startswith("qwen"):
        kwargs["enable_thinking"] = False
    else:
        kwargs["date_string"] = LLAMA_DATE
    full = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True, **kwargs)
    if full.count(SPLIT_SENTINEL) != 1:
        raise RuntimeError("split sentinel not found exactly once in rendered prompt")
    ctx, question = full.split(SPLIT_SENTINEL)
    return ctx, question


def _parse_prediction(raw: str) -> str:
    try:
        result = json.loads(raw.strip())
        mv = result.get("match", False)
        if isinstance(mv, str) and mv.lower() == "unsure":
            return "unsure"
        return "Yes" if bool(mv) else "No"
    except (json.JSONDecodeError, AttributeError):
        lower = raw.lower()
        if "unsure" in lower:
            return "unsure"
        if "true" in lower and "false" not in lower:
            return "Yes"
        if "false" in lower:
            return "No"
        return "PARSE_FAIL"


# ---------------- version-agnostic DynamicCache helpers ---------------- #
def _cache_layer_lens(cache) -> list[int]:
    if hasattr(cache, "layers"):                      # transformers 5.x
        return [l.keys.shape[2] if l.keys is not None else 0 for l in cache.layers]
    return [k.shape[2] for k in cache.key_cache]      # transformers 4.x


def _truncate_cache(cache, lens: list[int]) -> None:
    """kvpress _remove_answer_from_cache pattern: drop the appended answer KV."""
    if hasattr(cache, "layers"):
        for l, n in zip(cache.layers, lens):
            if l.keys is not None:
                l.keys = l.keys[:, :, :n]
                l.values = l.values[:, :, :n]
    else:
        for i, n in enumerate(lens):
            cache.key_cache[i] = cache.key_cache[i][:, :, :n]
            cache.value_cache[i] = cache.value_cache[i][:, :, :n]


class LadderRunner:
    def __init__(self, model_path: str, model_tag: str):
        self.tag = model_tag
        self.tok = AutoTokenizer.from_pretrained(model_path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map="auto",
                attn_implementation="sdpa")
        except TypeError:                              # transformers < 4.56
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation="sdpa")
        self.model.eval()
        got = next(self.model.parameters()).dtype
        if got != torch.bfloat16:
            raise SystemExit(f"COMPAT FAIL: model loaded in {got}, not bf16 "
                             f"(silent fp32 fallback would spill to CPU)")
        self.cls_ids = _class_token_ids(self.tok)
        self.dev = self.model.device
        # EOS stop set: generation_config may carry a LIST (Llama-3.1)
        gc = getattr(self.model.generation_config, "eos_token_id", None)
        self.eos_ids = set(gc) if isinstance(gc, (list, tuple)) else \
            ({gc} if gc is not None else set())
        if self.tok.eos_token_id is not None:
            self.eos_ids.add(self.tok.eos_token_id)
        self._compat_checks()

    def _compat_checks(self):
        """Fail fast if kvpress would silently mis-handle this architecture."""
        attn = self.model.model.layers[0].self_attn
        attn_cls = type(attn).__name__
        try:
            from kvpress.presses.base_press import SUPPORTED_MODELS
            if not isinstance(self.model, SUPPORTED_MODELS):
                raise SystemExit(
                    f"COMPAT FAIL: {type(self.model).__name__} not in kvpress "
                    f"SUPPORTED_MODELS -- ExpectedAttention scores would be unreliable")
        except ImportError:
            print("[compat] could not import kvpress SUPPORTED_MODELS; relying on"
                  " self-test", flush=True)
        # q_norm models must be dispatched to the right prerope branch
        if hasattr(attn, "q_norm") and attn_cls not in ("Qwen3Attention",
                                                        "Gemma3Attention"):
            raise SystemExit(
                f"COMPAT FAIL: attention class {attn_cls} has q_norm but kvpress "
                f"get_prerope_query_states would skip it -- wrong score space")
        print(f"[compat] {self.tag}: attention={attn_cls} ok", flush=True)

    # ---- document context prefill under the press (query-agnostic) ---- #
    def prefill_context(self, ctx_str: str, press):
        # chat template already contains ALL special tokens (incl. BOS) -- do not
        # let the tokenizer prepend another one (double-BOS corrupts Llama).
        ids = self.tok(ctx_str, return_tensors="pt",
                       add_special_tokens=False).input_ids
        if ids.shape[1] > MAX_CTX_TOKENS:
            ids = ids[:, :MAX_CTX_TOKENS]
        ids = ids.to(self.dev)
        cache = DynamicCache()
        with torch.no_grad():
            if press is None:
                self.model.model(input_ids=ids, past_key_values=cache,
                                 use_cache=True)
            else:
                with press(self.model):
                    self.model.model(input_ids=ids, past_key_values=cache,
                                     use_cache=True)
        kept = cache.get_seq_length()
        return cache, int(ids.shape[1]), int(kept), ids

    # ---- decode question + answer from the shared compressed cache ---- #
    def answer(self, ctx_len: int, cache, question_str: str) -> dict:
        q_ids = self.tok(question_str, add_special_tokens=False,
                         return_tensors="pt").input_ids.to(self.dev)
        q_len = int(q_ids.shape[1])
        past_len = cache.get_seq_length()
        base_lens = _cache_layer_lens(cache)
        gen_ids: list[int] = []
        step_logprobs = None
        with torch.no_grad():
            # RoPE positions continue from the ORIGINAL context length; the cache
            # write/mask positions continue from the COMPRESSED length.
            pos = torch.arange(ctx_len, ctx_len + q_len, device=self.dev)
            cpos = torch.arange(past_len, past_len + q_len, device=self.dev)
            out = self.model(input_ids=q_ids, past_key_values=cache,
                             position_ids=pos.unsqueeze(0), cache_position=cpos,
                             use_cache=True)
            next_pos = ctx_len + q_len
            next_cpos = past_len + q_len
            for _ in range(MAX_NEW_TOKENS):
                logits = out.logits[:, -1, :]
                lp = torch.log_softmax(logits.float(), dim=-1)[0]
                nxt = int(torch.argmax(lp).item())
                tok_txt = self.tok.decode([nxt]).strip().lower()
                gen_ids.append(nxt)
                # the '"' token is decision-ish (collector parity): true/false/"
                # compete at the same branch point before a quoted "unsure".
                if step_logprobs is None and (
                        tok_txt in TRUE_TOKENS or tok_txt in FALSE_TOKENS
                        or tok_txt in UNSURE_TOKENS or tok_txt == '"'):
                    step_logprobs = {
                        cls: max((float(lp[i]) for i in ids_), default=-100.0)
                        for cls, ids_ in self.cls_ids.items()}
                if nxt in self.eos_ids:
                    break
                out = self.model(
                    input_ids=torch.tensor([[nxt]], device=self.dev),
                    past_key_values=out.past_key_values,
                    position_ids=torch.tensor([[next_pos]], device=self.dev),
                    cache_position=torch.tensor([next_cpos], device=self.dev),
                    use_cache=True)
                next_pos += 1
                next_cpos += 1
        _truncate_cache(cache, base_lens)   # drop question+answer KV -> reusable
        raw = self.tok.decode(gen_ids, skip_special_tokens=True)
        lp_t = step_logprobs.get("true", -100.0) if step_logprobs else -100.0
        lp_f = step_logprobs.get("false", -100.0) if step_logprobs else -100.0
        lp_u = step_logprobs.get("unsure", -100.0) if step_logprobs else -100.0
        lps = [lp_t, lp_f, lp_u]
        active = [x for x in lps if x > -100.0]
        if len(active) >= 2:
            mx = max(lps)
            exps = [math.exp(x - mx) if x > -100.0 else 0.0 for x in lps]
            tot = sum(exps)
            p = [e / tot for e in exps]
        elif len(active) == 1:
            p = [1.0 if x > -100.0 else 0.0 for x in lps]
        else:
            p = [1 / 3] * 3
        pred = _parse_prediction(raw)
        conf = {"Yes": p[0], "No": p[1], "unsure": p[2]}.get(pred, max(p))
        ent = -sum(x * math.log2(x) for x in p if x > 1e-10)
        return dict(prediction=pred, raw_response=raw[:200],
                    p_true=p[0], p_false=p[1], p_unsure=p[2],
                    confidence=conf, entropy=ent,
                    logprob_true=lp_t, logprob_false=lp_f, logprob_unsure=lp_u,
                    q_tokens=q_len)


def _repair_torn_tail(path: str) -> None:
    """Drop a torn (newline-less) last line left by a mid-write kill."""
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        # walk back to the last newline and truncate the torn remainder
        pos = size - 1
        while pos > 0:
            f.seek(pos - 1)
            if f.read(1) == b"\n":
                break
            pos -= 1
        f.truncate(pos)
        print(f"[resume] repaired torn tail of {os.path.basename(path)} "
              f"({size - pos} bytes dropped)")


def run_dataset(runner: LadderRunner, ds: str, ratio: float, scope: str,
                input_dir: str, out_dir: str, deadline: float | None) -> bool:
    """Returns False if the time guard tripped (caller should stop)."""
    docs = pd.read_csv(os.path.join(input_dir, f"{ds}_docs.csv"))
    queries = pd.read_csv(os.path.join(input_dir, f"{ds}_queries.csv"))
    pairs = pd.read_csv(os.path.join(input_dir, f"{ds}_pairs.csv"))
    if pairs["in_sample70"].dtype != bool:
        pairs["in_sample70"] = pairs["in_sample70"].astype(str).str.lower() == "true"
    if scope == "sample":
        pairs = pairs[pairs["in_sample70"]]
    qdesc = dict(zip(queries["query_name"], queries["query_description"]))
    text_by_id = dict(zip(docs["document_id"], docs["doc_text"]))
    name_by_id = dict(zip(docs["document_id"], docs["document_name"]))

    out_path = os.path.join(out_dir, f"{runner.tag}_r{ratio}_{ds}.csv")
    done = set()
    has_data = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    if has_data:
        _repair_torn_tail(out_path)
        with open(out_path) as f:
            header = f.readline().strip().split(",")
        if header != FIELDNAMES:
            raise SystemExit(f"schema changed vs existing {out_path} -- move the "
                             f"old file before resuming")
        prev = pd.read_csv(out_path, usecols=["document_id", "query_name"])
        done = set(zip(prev["document_id"], prev["query_name"]))
    todo = pairs[[(r.document_id, r.query_name) not in done
                  for r in pairs.itertuples()]]
    if todo.empty:
        print(f"[{ds} r={ratio}] already complete ({len(pairs)} rows)")
        return True
    print(f"[{ds} r={ratio}] todo={len(todo)}/{len(pairs)} "
          f"(resume skips {len(done)})", flush=True)

    fout = open(out_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if not has_data:
        writer.writeheader()
        fout.flush()                       # header durable before first doc
    press = ExpectedAttentionPress(compression_ratio=ratio) if ratio > 0 else None
    n_done, t0 = 0, time.time()
    for doc_id, grp in todo.groupby("document_id", sort=True):
        if deadline and time.time() > deadline:
            print(f"[{ds} r={ratio}] TIME_GUARD after {n_done} rows")
            fout.close()
            return False
        doc_text = str(text_by_id[doc_id])
        cache = None
        for r in grp.itertuples():
            q = qdesc[r.query_name]
            ctx_str, q_str = _render_split_prompt(runner.tok, doc_text, q, runner.tag)
            if cache is None:   # prefill once per doc (query-agnostic compression)
                cache, ctx_tok, kept, _ = runner.prefill_context(ctx_str, press)
            res = runner.answer(ctx_tok, cache, q_str)
            writer.writerow(dict(document_id=doc_id,
                                 document_name=name_by_id[doc_id],
                                 query_name=r.query_name,
                                 ctx_tokens=ctx_tok, kept_tokens=kept, **res))
            n_done += 1
        del cache
        fout.flush()
        if n_done % 200 < len(grp):
            rate = n_done / max(time.time() - t0, 1e-9)
            print(f"  [{ds} r={ratio}] {n_done}/{len(todo)} rate={rate:.2f}/s "
                  f"eta={(len(todo)-n_done)/max(rate,1e-9)/60:.0f}min", flush=True)
    fout.close()
    print(f"[{ds} r={ratio}] DONE {n_done} rows in {(time.time()-t0)/60:.1f}min")
    return True


def self_test(runner: LadderRunner, ratios: list[float]):
    doc = ("The Supplier shall deliver the Goods within 30 days of the Order. "
           "This Agreement is governed by the laws of Singapore. " * 40)
    q = "Does the clause specify which law governs the agreement?"
    lo_by_ratio = {}
    for ratio in sorted({0.0, *ratios}):
        press = ExpectedAttentionPress(compression_ratio=ratio) if ratio > 0 else None
        ctx_str, q_str = _render_split_prompt(runner.tok, doc, q, runner.tag)
        cache, ctx_tok, kept, _ = runner.prefill_context(ctx_str, press)
        res = runner.answer(ctx_tok, cache, q_str)
        lo = res["logprob_true"] - res["logprob_false"]
        lo_by_ratio[ratio] = lo
        print(f"[self-test r={ratio}] ctx={ctx_tok} kept={kept} "
              f"pred={res['prediction']} lo={lo:+.2f} "
              f"raw={res['raw_response'][:40]!r}", flush=True)
        if ratio > 0 and kept > 0.95 * ctx_tok:
            raise SystemExit(f"COMPAT FAIL: press r={ratio} kept {kept}/{ctx_tok} "
                             f"tokens -- compression did not engage")
        if res["logprob_true"] <= -100.0 and res["logprob_false"] <= -100.0:
            raise SystemExit("COMPAT FAIL: no decision-token logprobs extracted")
    hi = [r for r in ratios if r >= 0.5]
    if hi and all(abs(lo_by_ratio[r] - lo_by_ratio[0.0]) < 1e-9 for r in hi):
        raise SystemExit("COMPAT FAIL: compressed log-odds bit-identical to r=0 -- "
                         "press hooks likely not firing")
    print(f"[self-test] {runner.tag} OK for ratios {sorted(set(ratios))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-tag", required=True, choices=["qwen08b", "llama70b"])
    ap.add_argument("--ratios", default="0.8,0.5,0.0")
    ap.add_argument("--datasets", default="cuad,contract,opp115,hoc")
    ap.add_argument("--scope", choices=["full", "sample"], default="full")
    ap.add_argument("--input-dir", default="tools/stretto/ladder_inputs")
    ap.add_argument("--out-dir", default="ladder_out")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    import transformers            # noqa: E401  (versions into the log)
    from importlib.metadata import version as _v
    print(f"[env] transformers={transformers.__version__} "
          f"kvpress={_v('kvpress')} torch={torch.__version__}", flush=True)

    ratios = [float(x) for x in args.ratios.split(",")]
    runner = LadderRunner(args.model_path, args.model_tag)
    if args.self_test:
        self_test(runner, ratios)
        return
    os.makedirs(args.out_dir, exist_ok=True)
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    self_test(runner, ratios)          # fail fast before burning hours
    for ratio in ratios:
        for ds in args.datasets.split(","):
            ok = run_dataset(runner, ds.strip(), ratio, args.scope,
                             args.input_dir, args.out_dir, deadline)
            if not ok:
                print("stopping on time guard; resubmit to resume")
                return


if __name__ == "__main__":
    main()
